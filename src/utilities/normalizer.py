import re

PERSONAL_PRONOUNS = re.compile(r"\b(we|you|our|your)\b", re.IGNORECASE)
STANDARD_BULLET = "-"
BULLET_PREFIX = re.compile(r"^\s*(?:[\u2022\-*]|\d+[.)]|[A-Za-z][.)])\s*")
WHITESPACE = re.compile(r"[ \t]+")
MOJIBAKE_REPLACEMENTS = (
    ("\u00c3\u0192\u00c2\u00a2\u00c3\u00a2\u00e2\u20ac\u0161\u00c2\u00ac\u00c3\u201a\u00c2\u00a2", "\u2022"),
    ("\u00e2\u20ac\u00a2", "\u2022"),
    ("\u00e2\u20ac\u0153", '"'),
    ("\u00e2\u20ac\u009d", '"'),
    ("\u00e2\u20ac\u02dc", "'"),
    ("\u00e2\u20ac\u2122", "'"),
    ("\u00e2\u20ac\u00a6", "..."),
    ("\u00e2\u20ac\u201d", "-"),
    ("\u00e2\u20ac\u201c", "-"),
    ("\u00e2\u20ac\u02dd", '"'),
    ("\u00e2\u20ac\u0158", "'"),
    ("\u00e2\u20ac\u2018", "-"),
    ("\u00e2\u20ac\u0090", "-"),
    ("\u00e2\u20ac\u0091", "-"),
    ("\u00e2\u20ac\u0092", "-"),
    ("\u00e2\u20ac\u0093", "-"),
    ("\u00e2\u20ac\u0094", "-"),
    ("\u00e2\u20ac\u0095", "-"),
    ("\u00e2\u201a\u00ac", "EUR"),
    ("\u00c2 ", " "),
    ("\u00c2\u00a0", " "),
    ("\u00c2", ""),
)


def repair_text(text: str) -> str:
    """Repair common mojibake and normalize line endings before further processing."""
    repaired = text.replace("\r\n", "\n").replace("\r", "\n")
    for broken, fixed in MOJIBAKE_REPLACEMENTS:
        repaired = repaired.replace(broken, fixed)
    return repaired


def clean_line(text: str) -> str:
    """Collapse repeated spaces and trim a single line."""
    return WHITESPACE.sub(" ", text).strip()


def is_bullet_line(text: str) -> bool:
    """Return whether the line begins with a bullet-like marker."""
    return bool(BULLET_PREFIX.match(text))


def strip_bullet_prefix(text: str) -> str:
    """Remove one bullet-like marker from the start of the line."""
    return BULLET_PREFIX.sub("", text, count=1).strip()


def looks_like_raw_slide_block(text: str) -> bool:
    """Detect OCR-style slide fragments with many short wrapped lines or bullet glyphs."""
    repaired = repair_text(text)
    lines = [clean_line(line) for line in repaired.splitlines() if clean_line(line)]
    if not lines:
        return False

    bullet_count = sum(1 for line in lines if is_bullet_line(line))
    short_line_count = sum(1 for line in lines if len(line.split()) <= 4)
    stacked_short_lines = short_line_count >= 4 and short_line_count >= len(lines) // 2
    mojibake_bullets = (
        "\u00e2\u20ac\u00a2" in text
        or "\u00c3\u0192\u00c2\u00a2\u00c3\u00a2\u00e2\u20ac\u0161\u00c2\u00ac\u00c3\u201a\u00c2\u00a2" in text
    )
    return mojibake_bullets or bullet_count >= 2 or stacked_short_lines


def normalize(text: str) -> str:
    """Repair text, normalize bullets, and merge wrapped slide lines into readable blocks."""
    lines = [clean_line(line) for line in repair_text(text).splitlines()]
    normalized_blocks: list[str] = []
    current_paragraph: list[str] = []
    current_bullets: list[str] = []

    def flush_paragraph():
        if current_paragraph:
            normalized_blocks.append(" ".join(current_paragraph))
            current_paragraph.clear()

    def flush_bullets():
        if current_bullets:
            normalized_blocks.extend(f"{STANDARD_BULLET} {item}" for item in current_bullets)
            current_bullets.clear()

    for line in lines:
        if not line:
            flush_paragraph()
            flush_bullets()
            continue

        if is_bullet_line(line):
            flush_paragraph()
            current_bullets.append(strip_bullet_prefix(line))
            continue

        if current_bullets:
            current_bullets[-1] = f"{current_bullets[-1]} {line}".strip()
            continue

        current_paragraph.append(line)

    flush_paragraph()
    flush_bullets()
    return "\n".join(block for block in normalized_blocks if block)


def has_personal_pronouns(text: str) -> bool:
    """Check whether the text contains first or second person pronouns."""
    return bool(PERSONAL_PRONOUNS.search(repair_text(text)))


def has_bullet_lines(raw_text: str) -> bool:
    """Check whether any line in the text starts with a bullet character."""
    repaired = repair_text(raw_text)
    return any(is_bullet_line(clean_line(line)) for line in repaired.splitlines())
