import re

PERSONAL_PRONOUNS = re.compile(r"\b(we|you|our|your)\b", re.IGNORECASE)
BULLET_PREFIX = re.compile(r"^\s*[â€¢\-\*\u2013\u2014]\s*")


def normalize(text: str) -> str:
    """Strip bullet prefixes and collapse whitespace into clean lines."""
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        line = BULLET_PREFIX.sub("", line)
        line = re.sub(r"  +", " ", line)
        line = line.strip()
        if line:
            cleaned.append(line)
    return "\n".join(cleaned)


def has_personal_pronouns(text: str) -> bool:
    """Check whether the text contains first or second person pronouns."""
    return bool(PERSONAL_PRONOUNS.search(text))


def has_bullet_lines(raw_text: str) -> bool:
    """Check whether any line in the text starts with a bullet character."""
    return any(BULLET_PREFIX.match(line) for line in raw_text.splitlines())
