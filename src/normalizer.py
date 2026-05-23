import re


# Words that signal the bullet is a standalone concept label (no elaboration)
_PERSONAL_PRONOUNS = re.compile(
    r'\b(we|you|our|your)\b', re.IGNORECASE
)

_BULLET_PREFIX = re.compile(r'^\s*[•\-\*\u2013\u2014]\s*')


def normalize(text: str) -> str:
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        # Remove bullet prefix characters, keep the content
        line = _BULLET_PREFIX.sub('', line)
        # Collapse multiple spaces
        line = re.sub(r'  +', ' ', line)
        line = line.strip()
        if line:
            cleaned.append(line)
    return '\n'.join(cleaned)


def has_personal_pronouns(text: str) -> bool:
    return bool(_PERSONAL_PRONOUNS.search(text))


def has_bullet_lines(raw_text: str) -> bool:
    return any(_BULLET_PREFIX.match(line) for line in raw_text.splitlines())
