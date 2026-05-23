from enum import Enum
from pydantic import BaseModel


class SlideType(str, Enum):
    CONTENT = "content"
    IMAGE_DESCRIPTION = "image_description"
    INTRODUCTION = "introduction"
    COURSE_INFO = "course_info"


class ActionType(str, Enum):
    INSERT_CONNECTIVITY = "insert_connectivity"
    REMOVE_PERSONAL_PRONOUNS = "remove_personal_pronouns"
    REMOVE_USELESS_BULLETS = "remove_useless_bullets"
    DEFINE_ACRONYM = "define_acronym"
    INCOMPLETE_SENTENCE = "incomplete_sentence"


class SuggestedAction(BaseModel):
    action: ActionType
    original_fragment: str


class SlideReview(BaseModel):
    slide_number: int
    slide_type: SlideType
    title: str | None
    is_continuation: bool
    key_concepts: list[str]
    summary: str | None
    actions: list[SuggestedAction]
