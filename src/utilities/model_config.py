import os

WINDOW_SECONDS = 60

DEFAULT_LIGHT_MODEL = "google:gemini-3.1-flash-lite"
DEFAULT_FULL_TEXT_MODEL = "google:gemini-2.5-flash"
DEFAULT_MODEL_RPD = 1000


def get_env_text(name: str, default: str) -> str:
    """Return a trimmed environment override when present, otherwise the default."""
    value = os.getenv(name)
    if value is None:
        return default

    value = value.strip()
    return value or default


def get_env_int(name: str, default: int) -> int:
    """Return a positive integer environment override or fall back safely."""
    value = os.getenv(name)
    if value is None:
        return default

    try:
        parsed = int(value)
    except ValueError:
        return default

    if parsed < 1:
        return default

    return parsed


LIGHT_MODEL = get_env_text("LIGHT_MODEL", DEFAULT_LIGHT_MODEL)
FULL_TEXT_MODEL = get_env_text("FULL_TEXT_MODEL", DEFAULT_FULL_TEXT_MODEL)
GEMMA_LIGHT_MODEL = get_env_text("GEMMA_LIGHT_MODEL", "")
if GEMMA_LIGHT_MODEL:
    LIGHT_MODEL = GEMMA_LIGHT_MODEL

CHECKER_MODEL = get_env_text("CHECKER_MODEL", LIGHT_MODEL)
REVIEWER_MODEL = get_env_text("REVIEWER_MODEL", CHECKER_MODEL)
REWRITER_MODEL = get_env_text("REWRITER_MODEL", CHECKER_MODEL)
MATH_MODEL = get_env_text("MATH_MODEL", CHECKER_MODEL)
TITLE_MODEL = get_env_text("TITLE_MODEL", FULL_TEXT_MODEL)
QUALITY_IDENTIFIER_MODEL = get_env_text("QUALITY_IDENTIFIER_MODEL", FULL_TEXT_MODEL)
QUALITY_FIXER_MODEL = get_env_text("QUALITY_FIXER_MODEL", FULL_TEXT_MODEL)

CHECKER_MODEL_RPM = get_env_int("CHECKER_MODEL_RPM", 14)
REVIEWER_MODEL_RPM = get_env_int("REVIEWER_MODEL_RPM", CHECKER_MODEL_RPM)
REWRITER_MODEL_RPM = get_env_int("REWRITER_MODEL_RPM", 14)
MATH_MODEL_RPM = get_env_int("MATH_MODEL_RPM", 12)
FULL_TEXT_RPM = get_env_int("FULL_TEXT_RPM", 5)
TITLE_MODEL_RPM = get_env_int("TITLE_MODEL_RPM", FULL_TEXT_RPM)
QUALITY_IDENTIFIER_MODEL_RPM = get_env_int("QUALITY_IDENTIFIER_MODEL_RPM", FULL_TEXT_RPM)
QUALITY_FIXER_MODEL_RPM = get_env_int("QUALITY_FIXER_MODEL_RPM", FULL_TEXT_RPM)


def infer_model_rpd(model_name: str) -> int:
    """Infer a safe default daily request budget from the model family name."""
    normalized = model_name.lower()
    if "gemini-3.1-flash-lite" in normalized:
        return 500
    if "gemini-3.5-flash" in normalized:
        return 20
    if "gemini-2.5-flash" in normalized:
        return 20
    if "gemma" in normalized:
        return 1500
    return DEFAULT_MODEL_RPD


def get_model_rpd(model_name: str, env_name: str) -> int:
    """Return the configured daily budget for a model, falling back to inferred limits."""
    return get_env_int(env_name, infer_model_rpd(model_name))


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
