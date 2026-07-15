import os
from dataclasses import dataclass

WINDOW_SECONDS = 60

DEFAULT_MODEL = "google:gemma-4-26b-a4b-it"
DEFAULT_MODEL_RPM = 10
DEFAULT_MODEL_RPD = 1000
FAST_NOTE_MODEL = "groq:openai/gpt-oss-20b"


@dataclass(frozen=True)
class ModelLimit:
    """Default request budgets for a model family."""

    label: str
    aliases: tuple[str, ...]
    max_rpm: int
    max_rpd: int


MODEL_LIMITS = (
    ModelLimit("Groq GPT-OSS 20B", ("openai/gpt-oss-20b", "gpt-oss-20b"), 30, 1000),
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
    ModelLimit("Gemma 4 26B", ("gemma-4-26b", "gemma-4-26b-it", "gemma-4-26b-a4b-it"), 15, 1500),
    ModelLimit("Gemma", ("gemma",), 15, 1500)
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


def get_model_rpm(model_name: str, env_name: str) -> int:
    """Return the configured per-minute budget for a model."""
    return get_env_int(env_name, get_default_model_limits(model_name)[0], minimum=0)


def get_model_rpd(model_name: str, env_name: str) -> int:
    """Return the configured daily budget for a model."""
    return get_env_int(env_name, get_default_model_limits(model_name)[1], minimum=0)


def get_default_note_model() -> str:
    """Prefer the fast Groq free-tier model when its standard key is configured."""
    return FAST_NOTE_MODEL if os.getenv("GROQ_API_KEY", "").strip() else DEFAULT_MODEL


CHECKER_MODEL = get_env_text("CHECKER_MODEL", DEFAULT_MODEL)
REWRITER_MODEL = get_env_text("REWRITER_MODEL", get_default_note_model())
MATH_MODEL = get_env_text("MATH_MODEL", DEFAULT_MODEL)
TITLE_MODEL = get_env_text("TITLE_MODEL", DEFAULT_MODEL)
QUALITY_IDENTIFIER_MODEL = get_env_text("QUALITY_IDENTIFIER_MODEL", DEFAULT_MODEL)
QUALITY_FIXER_MODEL = get_env_text("QUALITY_FIXER_MODEL", DEFAULT_MODEL)
FALLBACK_MODEL = get_env_text("FALLBACK_MODEL", "google:gemma-4-31b-it")

CHECKER_MODEL_RPM = get_model_rpm(CHECKER_MODEL, "CHECKER_MODEL_RPM")
REWRITER_MODEL_RPM = get_model_rpm(REWRITER_MODEL, "REWRITER_MODEL_RPM")
MATH_MODEL_RPM = get_model_rpm(MATH_MODEL, "MATH_MODEL_RPM")
TITLE_MODEL_RPM = get_model_rpm(TITLE_MODEL, "TITLE_MODEL_RPM")
QUALITY_IDENTIFIER_MODEL_RPM = get_model_rpm(QUALITY_IDENTIFIER_MODEL, "QUALITY_IDENTIFIER_MODEL_RPM")
QUALITY_FIXER_MODEL_RPM = get_model_rpm(QUALITY_FIXER_MODEL, "QUALITY_FIXER_MODEL_RPM")
FALLBACK_MODEL_RPM = get_model_rpm(FALLBACK_MODEL, "FALLBACK_MODEL_RPM")

CHECKER_MODEL_RPD = get_model_rpd(CHECKER_MODEL, "CHECKER_MODEL_RPD")
REWRITER_MODEL_RPD = get_model_rpd(REWRITER_MODEL, "REWRITER_MODEL_RPD")
MATH_MODEL_RPD = get_model_rpd(MATH_MODEL, "MATH_MODEL_RPD")
TITLE_MODEL_RPD = get_model_rpd(TITLE_MODEL, "TITLE_MODEL_RPD")
QUALITY_IDENTIFIER_MODEL_RPD = get_model_rpd(QUALITY_IDENTIFIER_MODEL, "QUALITY_IDENTIFIER_MODEL_RPD")
QUALITY_FIXER_MODEL_RPD = get_model_rpd(QUALITY_FIXER_MODEL, "QUALITY_FIXER_MODEL_RPD")
FALLBACK_MODEL_RPD = get_model_rpd(FALLBACK_MODEL, "FALLBACK_MODEL_RPD")


def get_model_summary() -> list[tuple[str, str, int, int]]:
    """Return the active stage model configuration in display order."""
    return [
        ("note_extractor", REWRITER_MODEL, REWRITER_MODEL_RPM, REWRITER_MODEL_RPD),
        ("math", MATH_MODEL, MATH_MODEL_RPM, MATH_MODEL_RPD),
        ("title", TITLE_MODEL, TITLE_MODEL_RPM, TITLE_MODEL_RPD),
        ("quality_identifier", QUALITY_IDENTIFIER_MODEL, QUALITY_IDENTIFIER_MODEL_RPM, QUALITY_IDENTIFIER_MODEL_RPD),
        ("quality_fixer", QUALITY_FIXER_MODEL, QUALITY_FIXER_MODEL_RPM, QUALITY_FIXER_MODEL_RPD),
        ("503_fallback", FALLBACK_MODEL, FALLBACK_MODEL_RPM, FALLBACK_MODEL_RPD)
    ]
