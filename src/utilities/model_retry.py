import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError

from .rate_limit import DailyQuotaExceededError, RequestPacer

MAX_MODEL_RETRIES = 3
TIMEOUT_RETRY_DELAY_SECONDS = 30.0
TRANSIENT_STATUS_CODES = {429, 500, 503}


@dataclass(frozen=True)
class ModelRequestCandidate:
    """One model candidate for a provider request."""

    model_name: str
    rpm: int
    rpd: int
    runner: Callable[[str], Any]


def get_cached_agent(
    agent_cache: dict[str, Agent],
    model_name: str,
    output_type,
    instructions: str,
) -> Agent:
    """Return an Agent for a model, creating it only when needed."""
    agent = agent_cache.get(model_name)
    if agent is None:
        agent = Agent(model_name, output_type=output_type, instructions=instructions, model_settings={"temperature": 0})
        agent_cache[model_name] = agent
    return agent


def get_retry_delay(error: ModelHTTPError, window_seconds: int) -> float:
    """Extract a useful retry delay from the provider response when available."""
    retry_delay = None
    body = error.body
    if isinstance(body, dict):
        error_payload = body.get("error")
        if isinstance(error_payload, dict):
            details = error_payload.get("details")
            if isinstance(details, list):
                for detail in details:
                    if not isinstance(detail, dict):
                        continue
                    retry_text = detail.get("retryDelay")
                    if isinstance(retry_text, str) and retry_text.endswith("s"):
                        try:
                            retry_delay = float(retry_text[:-1])
                            break
                        except ValueError:
                            continue

    if retry_delay is not None:
        return max(retry_delay, 1.0)

    if error.status_code == 429:
        return float(window_seconds)
    return 30.0


def run_with_transient_retry_and_fallback(
    pacer: RequestPacer,
    request_name: str,
    candidates: list[ModelRequestCandidate],
    prompt: str,
    window_seconds: int,
):
    """Run a model request, switching model after repeated 503 responses."""
    if not candidates:
        raise ValueError(f"No model candidates configured for {request_name}")

    last_error: Exception | None = None
    for candidate_index, candidate in enumerate(candidates):
        attempt = 0
        consecutive_503s = 0
        while True:
            pacer.acquire_request_slot(candidate.model_name, candidate.rpm, candidate.rpd, request_name)
            try:
                return candidate.runner(prompt)
            except ModelHTTPError as error:
                if error.status_code == 429 and pacer.is_daily_quota_error(error.body):
                    pacer.mark_daily_exhausted(candidate.model_name, candidate.rpd)
                    raise DailyQuotaExceededError(
                        f"Provider daily request quota exhausted for {candidate.model_name}. "
                        "Resume after reset or switch this stage to another model."
                    ) from error
                if error.status_code not in TRANSIENT_STATUS_CODES:
                    raise

                attempt += 1
                if error.status_code == 503:
                    consecutive_503s += 1
                else:
                    consecutive_503s = 0
                last_error = error
                has_fallback = candidate_index < len(candidates) - 1
                if consecutive_503s >= MAX_MODEL_RETRIES and has_fallback:
                    next_model = candidates[candidate_index + 1].model_name
                    print(
                        f"{request_name} received {MAX_MODEL_RETRIES} consecutive 503s from "
                        f"{candidate.model_name}; switching to {next_model}..."
                    )
                    break
                if attempt >= MAX_MODEL_RETRIES:
                    raise

                delay = get_retry_delay(error, window_seconds)
                print(
                    f"{request_name} transient error {error.status_code} from {candidate.model_name} - "
                    f"retrying in {delay:.1f}s ({attempt}/{MAX_MODEL_RETRIES})..."
                )
                time.sleep(delay)
            except httpx.TimeoutException as error:
                attempt += 1
                last_error = error
                if attempt >= MAX_MODEL_RETRIES:
                    raise

                print(
                    f"{request_name} transient timeout from {candidate.model_name} - retrying in "
                    f"{TIMEOUT_RETRY_DELAY_SECONDS:.1f}s ({attempt}/{MAX_MODEL_RETRIES})..."
                )
                time.sleep(TIMEOUT_RETRY_DELAY_SECONDS)

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"All model candidates failed for {request_name}")
