from enum import Enum
from pydantic import BaseModel


class SlideType(str, Enum):
    """Classification of a lecture slide by its content purpose."""
    CONTENT = "content"
    IMAGE_DESCRIPTION = "image_description"
    INTRODUCTION = "introduction"
    COURSE_INFO = "course_info"


class ActionType(str, Enum):
    """Types of editing actions the checker can suggest for a slide."""
    INSERT_CONNECTIVITY = "insert_connectivity"
    REMOVE_PERSONAL_PRONOUNS = "remove_personal_pronouns"
    FLATTEN_BULLETS = "flatten_bullets"
    DEFINE_ACRONYM = "define_acronym"
    INCOMPLETE_SENTENCE = "incomplete_sentence"


class SuggestedAction(BaseModel):
    """A single editing action tied to a specific text fragment."""
    action: ActionType
    original_fragment: str


class SlideReviewResponse(BaseModel):
    """LLM-produced review of a slide, without the slide number."""
    slide_type: SlideType
    title: str | None
    is_continuation: bool
    key_concepts: list[str]
    summary: str | None
    actions: list[SuggestedAction]


class SlideReview(BaseModel):
    """Complete review of a slide, including its position in the deck."""
    slide_number: int
    slide_type: SlideType
    title: str | None
    is_continuation: bool
    key_concepts: list[str]
    summary: str | None
    actions: list[SuggestedAction]


class HeadingAction(str, Enum):
    """What to do with a heading in the document."""
    KEEP = "keep"
    REMOVE = "remove"


class HeadingChange(BaseModel):
    """A single heading change identified by the title analysis agent."""
    original_heading: str
    action: HeadingAction
    new_level: int | None
    new_text: str | None


class TitleAnalysis(BaseModel):
    """All heading changes needed in the document."""
    changes: list[HeadingChange]


class IssueType(str, Enum):
    """Types of quality issues that can be flagged in the final document."""
    LIST_COLLAPSED = "list_collapsed"
    EXCESSIVE_INTRODUCTION = "excessive_introduction"
    DANGLING_REFERENCE = "dangling_reference"
    CONTENT_LOST = "content_lost"
    REPETITION = "repetition"
    AWKWARD_FLOW = "awkward_flow"


class QualityIssue(BaseModel):
    """A single quality issue found in the final document."""
    issue_type: IssueType
    problematic_text: str
    explanation: str


class QualityReport(BaseModel):
    """Quality review of the complete markdown document."""
    issues: list[QualityIssue]


class SlideRewrite(BaseModel):
    """Per-slide rewrite result stored as JSON artifact."""
    slide_number: int
    slide_type: str
    title: str | None
    is_continuation: bool
    text: str


class MathReplacement(BaseModel):
    """A single math expression with its LaTeX equivalent."""
    original_text: str
    latex: str
    is_display: bool


class MathReplacementResponse(BaseModel):
    """All math replacements identified for a slide."""
    replacements: list[MathReplacement]
