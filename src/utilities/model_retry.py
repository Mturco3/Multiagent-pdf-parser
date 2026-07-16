"""Create reusable model agents and handle temporary provider failures.

The module contains agent configuration and caching helpers, retry-delay
parsing, quota-aware request attempts, and fallback-model orchestration.
"""

import os
import time
from collections.abc import Callable
from typing import Any, TypeVar

import httpx
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError

from .model_config import FALLBACK_MODEL, FALLBACK_MODEL_RPD, FALLBACK_MODEL_RPM
from .rate_limit import DailyQuotaExceededError, RequestPacer

MAX_RETRIES = 3
TIMEOUT_RETRY_DELAY_SECONDS = 30.0
DEFAULT_REQUEST_TIMEOUT_SECONDS = 60.0
TRANSIENT_STATUS_CODES = {429, 500, 503}
DEFAULT_RETRY_DELAY_SECONDS = 30.0
OutputT = TypeVar("OutputT")


def get_request_timeout_seconds() -> float:
    """Return a positive per-request timeout from the environment."""
    configured_timeout = os.getenv(
        "MODEL_REQUEST_TIMEOUT_SECONDS",
        str(DEFAULT_REQUEST_TIMEOUT_SECONDS)
    ).strip()
    try:
        timeout_seconds = float(configured_timeout)
    except ValueError:
        return DEFAULT_REQUEST_TIMEOUT_SECONDS
    return timeout_seconds if timeout_seconds > 0 else DEFAULT_REQUEST_TIMEOUT_SECONDS


def get_agent_model_settings(model_name: str) -> dict[str, float | str]:
    """Return shared request settings with low reasoning for fast slide extraction."""
    settings: dict[str, float | str] = {
        "temperature": 0,
        "timeout": get_request_timeout_seconds()
    }
    if model_name.startswith("groq:openai/gpt-oss-"):
        settings["thinking"] = "low"
    return settings


def get_cached_agent(
    agent_cache: dict[str, Agent],
    model_name: str,
    output_type: type[OutputT],
    instructions: str
) -> Agent:
    """Return an Agent for a model, creating it only when needed."""
    agent = agent_cache.get(model_name)
    if agent is None:
        agent = Agent(
            model_name,
            output_type=output_type,
            instructions=instructions,
            model_settings=get_agent_model_settings(model_name)
        )
        agent_cache[model_name] = agent
    return agent


def run_agent_request(
    pacer: RequestPacer,
    request_name: str,
    model_name: str,
    rpm: int,
    rpd: int,
    agent_cache: dict[str, Agent],
    output_type: type[OutputT],
    instructions: str,
    prompt: str,
    window_seconds: int
) -> OutputT:
    """Run one typed agent request through shared pacing and retry handling."""

    def call_model(active_model: str, request_prompt: str) -> OutputT:
        """Reuse the agent when retry handling selects the same model again."""
        agent = get_cached_agent(
            agent_cache,
            active_model,
            output_type,
            instructions
        )
        return agent.run_sync(request_prompt).output

    return run_with_retry(
        pacer=pacer,
        request_name=request_name,
        model_name=model_name,
        rpm=rpm,
        rpd=rpd,
        runner=call_model,
        prompt=prompt,
        window_seconds=window_seconds
    )


def get_provider_retry_delay(error_body: Any) -> float | None:
    """Read a provider-supplied retry delay from a structured error body."""
    if not isinstance(error_body, dict):
        return None

    error_payload = error_body.get("error")
    if not isinstance(error_payload, dict):
        return None

    details = error_payload.get("details")
    if not isinstance(details, list):
        return None

    for detail in details:
        if not isinstance(detail, dict):
            continue
        retry_text = detail.get("retryDelay")
        if not isinstance(retry_text, str) or not retry_text.endswith("s"):
            continue
        try:
            return float(retry_text[:-1])
        except ValueError:
            continue
    return None


def get_retry_delay(error: ModelHTTPError, window_seconds: int) -> float:
    """Return a provider delay or a safe default for a transient failure."""
    retry_delay = get_provider_retry_delay(error.body)

    if retry_delay is not None:
        return max(retry_delay, 1.0)

    if error.status_code == 429:
        return float(window_seconds)
    return DEFAULT_RETRY_DELAY_SECONDS


def run_model_attempts(
    pacer: RequestPacer,
    request_name: str,
    model_name: str,
    rpm: int,
    rpd: int,
    runner: Callable[[str, str], Any],
    prompt: str,
    window_seconds: int
):
    """Run all retry attempts against one active model."""
    for attempt in range(1, MAX_RETRIES + 1):
        pacer.acquire_request_slot(model_name, rpm, rpd, request_name, prompt)
        started_at = time.monotonic()
        print(
            f"{request_name} request {attempt}/{MAX_RETRIES} "
            f"sent to {model_name}..."
        )
        try:
            result = runner(model_name, prompt)
            elapsed_seconds = time.monotonic() - started_at
            print(f"{request_name} request completed in {elapsed_seconds:.1f}s")
            return result
        except ModelHTTPError as error:
            if error.status_code == 429 and pacer.is_daily_quota_error(error.body):
                pacer.mark_daily_exhausted(model_name, rpd)
                raise DailyQuotaExceededError(
                    f"Provider daily request quota exhausted for {model_name}. "
                    "Resume after reset or switch this stage to another model."
                ) from error
            if error.status_code not in TRANSIENT_STATUS_CODES:
                raise
            if attempt >= MAX_RETRIES:
                raise

            delay = get_retry_delay(error, window_seconds)
            print(
                f"{request_name} transient error {error.status_code} from "
                f"{model_name} - retrying in {delay:.1f}s "
                f"({attempt}/{MAX_RETRIES})..."
            )
            time.sleep(delay)
        except httpx.TimeoutException:
            elapsed_seconds = time.monotonic() - started_at
            if attempt >= MAX_RETRIES:
                print(
                    f"{request_name} request timed out after "
                    f"{elapsed_seconds:.1f}s; no retries remain"
                )
                raise

            print(
                f"{request_name} request timed out after {elapsed_seconds:.1f}s "
                f"from {model_name} - retrying in "
                f"{TIMEOUT_RETRY_DELAY_SECONDS:.1f}s "
                f"({attempt}/{MAX_RETRIES})..."
            )
            time.sleep(TIMEOUT_RETRY_DELAY_SECONDS)


def run_with_retry(
    pacer: RequestPacer,
    request_name: str,
    model_name: str,
    rpm: int,
    rpd: int,
    runner: Callable[[str, str], Any],
    prompt: str,
    window_seconds: int
):
    """Run a model request and switch once to the configured fallback after repeated 503 errors."""
    try:
        return run_model_attempts(
            pacer,
            request_name,
            model_name,
            rpm,
            rpd,
            runner,
            prompt,
            window_seconds
        )
    except ModelHTTPError as error:
        can_fallback = error.status_code == 503 and FALLBACK_MODEL != model_name
        if not can_fallback:
            raise
        print(
            f"{request_name} switching from {model_name} to {FALLBACK_MODEL} "
            "after repeated 503 errors"
        )
        fallback_name = f"{request_name}-fallback"
        return run_model_attempts(
            pacer,
            fallback_name,
            FALLBACK_MODEL,
            FALLBACK_MODEL_RPM,
            FALLBACK_MODEL_RPD,
            runner,
            prompt,
            window_seconds
        )
