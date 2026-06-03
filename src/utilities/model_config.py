import os
from dataclasses import dataclass

WINDOW_SECONDS = 60

DEFAULT_LIGHT_MODEL = "google:gemini-3.1-flash-lite"
DEFAULT_FULL_TEXT_MODEL = "google:gemini-2.5-flash"
DEFAULT_GEMMA_LIGHT_MODEL = "google:gemma-4-26b"
DEFAULT_MODEL_RPM = 10
DEFAULT_MODEL_RPD = 1000


@dataclass(frozen=True)
class ModelLimit:
    """Default request budgets for a model family."""

    label: str
    aliases: tuple[str, ...]
    max_rpm: int
    max_rpd: int


MODEL_LIMITS = (
    ModelLimit("Gemini 3.1 Flash Lite", ("gemini-3.1-flash-lite",), 15, 500),
    ModelLimit("Gemini 3.5 Flash", ("gemini-3.5-flash",), 5, 20),
    ModelLimit("Gemini 3 Flash", ("gemini-3-flash",), 5, 20),
    ModelLimit("Gemini 2.5 Flash Lite", ("gemini-2.5-flash-lite",), 10, 20),
    ModelLimit("Gemini 2.5 Flash", ("gemini-2.5-flash",), 5, 20),
    ModelLimit("Gemini 3.1 Pro", ("gemini-3.1-pro",), 0, 0),
    ModelLimit("Gemini 2.5 Pro", ("gemini-2.5-pro",), 0, 0),
    ModelLimit("Gemini 2 Flash Lite", ("gemini-2.0-flash-lite", "gemini-2-flash-lite"), 0, 0),
    ModelLimit("Gemini 2 Flash", ("gemini-2.0-flash", "gemini-2-flash"), 0, 0),
    ModelLimit("Gemma 4 31B", ("gemma-4-31b", "gemma-4-31b-it"), 15, 1500),
    ModelLimit("Gemma 4 26B", ("gemma-4-26b", "gemma-4-26b-it"), 15, 1500),
    ModelLimit("Gemma", ("gemma",), 15, 1500),
)


def get_env_text(name: str, default: str) -> str:
    """Return a trimmed environment override when present, otherwise the default."""
    value = os.getenv(name)
    if value is None:
        return default

    value = value.strip()
    return value or default


def get_env_int(name: str, default: int, *, minimum: int = 1) -> int:
    """Return an integer environment override or fall back safely."""
    value = os.getenv(name)
    if value is None:
        return default

    try:
        parsed = int(value)
    except ValueError:
        return default

    if parsed < minimum:
        return default

    return parsed


def normalize_model_name(model_name: str) -> str:
    """Normalize provider-prefixed model identifiers for limit lookup."""
    normalized = model_name.lower().strip()
    if ":" in normalized:
        normalized = normalized.split(":", 1)[1]
    return normalized


def get_default_model_limits(model_name: str) -> tuple[int, int]:
    """Return default rpm/rpd limits for a configured model family."""
    normalized = normalize_model_name(model_name)
    for limit in MODEL_LIMITS:
        if any(alias in normalized for alias in limit.aliases):
            return limit.max_rpm, limit.max_rpd
    return DEFAULT_MODEL_RPM, DEFAULT_MODEL_RPD


LIGHT_MODEL = get_env_text("LIGHT_MODEL", DEFAULT_LIGHT_MODEL)
FULL_TEXT_MODEL = get_env_text("FULL_TEXT_MODEL", DEFAULT_FULL_TEXT_MODEL)

CHECKER_MODEL = get_env_text("CHECKER_MODEL", LIGHT_MODEL)
REVIEWER_MODEL = get_env_text("REVIEWER_MODEL", CHECKER_MODEL)
REWRITER_MODEL = get_env_text("REWRITER_MODEL", CHECKER_MODEL)
MATH_MODEL = get_env_text("MATH_MODEL", CHECKER_MODEL)
TITLE_MODEL = get_env_text("TITLE_MODEL", FULL_TEXT_MODEL)
QUALITY_IDENTIFIER_MODEL = get_env_text("QUALITY_IDENTIFIER_MODEL", FULL_TEXT_MODEL)
QUALITY_FIXER_MODEL = get_env_text("QUALITY_FIXER_MODEL", FULL_TEXT_MODEL)

FULL_TEXT_RPM = get_env_int("FULL_TEXT_RPM", get_default_model_limits(FULL_TEXT_MODEL)[0], minimum=0)


def get_model_rpm(model_name: str, env_name: str) -> int:
    """Return the configured per-minute budget for a model."""
    return get_env_int(env_name, get_default_model_limits(model_name)[0], minimum=0)


def get_model_rpd(model_name: str, env_name: str) -> int:
    """Return the configured daily budget for a model."""
    return get_env_int(env_name, get_default_model_limits(model_name)[1], minimum=0)


CHECKER_MODEL_RPM = get_model_rpm(CHECKER_MODEL, "CHECKER_MODEL_RPM")
REVIEWER_MODEL_RPM = get_model_rpm(REVIEWER_MODEL, "REVIEWER_MODEL_RPM")
REWRITER_MODEL_RPM = get_model_rpm(REWRITER_MODEL, "REWRITER_MODEL_RPM")
MATH_MODEL_RPM = get_model_rpm(MATH_MODEL, "MATH_MODEL_RPM")
TITLE_MODEL_RPM = get_env_int("TITLE_MODEL_RPM", get_default_model_limits(TITLE_MODEL)[0], minimum=0)
QUALITY_IDENTIFIER_MODEL_RPM = get_env_int("QUALITY_IDENTIFIER_MODEL_RPM", get_default_model_limits(QUALITY_IDENTIFIER_MODEL)[0], minimum=0)
QUALITY_FIXER_MODEL_RPM = get_env_int("QUALITY_FIXER_MODEL_RPM", get_default_model_limits(QUALITY_FIXER_MODEL)[0], minimum=0)

CHECKER_MODEL_RPD = get_model_rpd(CHECKER_MODEL, "CHECKER_MODEL_RPD")
REVIEWER_MODEL_RPD = get_model_rpd(REVIEWER_MODEL, "REVIEWER_MODEL_RPD")
REWRITER_MODEL_RPD = get_model_rpd(REWRITER_MODEL, "REWRITER_MODEL_RPD")
MATH_MODEL_RPD = get_model_rpd(MATH_MODEL, "MATH_MODEL_RPD")
TITLE_MODEL_RPD = get_model_rpd(TITLE_MODEL, "TITLE_MODEL_RPD")
QUALITY_IDENTIFIER_MODEL_RPD = get_model_rpd(QUALITY_IDENTIFIER_MODEL, "QUALITY_IDENTIFIER_MODEL_RPD")
QUALITY_FIXER_MODEL_RPD = get_model_rpd(QUALITY_FIXER_MODEL, "QUALITY_FIXER_MODEL_RPD")


def get_model_summary() -> list[tuple[str, str, int, int]]:
    """Return the active stage model configuration in display order."""
    return [
        ("checker", CHECKER_MODEL, CHECKER_MODEL_RPM, CHECKER_MODEL_RPD),
        ("reviewer", REVIEWER_MODEL, REVIEWER_MODEL_RPM, REVIEWER_MODEL_RPD),
        ("rewriter", REWRITER_MODEL, REWRITER_MODEL_RPM, REWRITER_MODEL_RPD),
        ("math", MATH_MODEL, MATH_MODEL_RPM, MATH_MODEL_RPD),
        ("title", TITLE_MODEL, TITLE_MODEL_RPM, TITLE_MODEL_RPD),
        ("quality_identifier", QUALITY_IDENTIFIER_MODEL, QUALITY_IDENTIFIER_MODEL_RPM, QUALITY_IDENTIFIER_MODEL_RPD),
        ("quality_fixer", QUALITY_FIXER_MODEL, QUALITY_FIXER_MODEL_RPM, QUALITY_FIXER_MODEL_RPD),
    ]
