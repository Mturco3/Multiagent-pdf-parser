"""Convert one normalized slide into structured Markdown-ready notes.

The module contains ``LLMRewriter``, source-coverage validation, and a safe
fallback that preserves source text when model extraction is unreliable.
"""

import re
from collections import Counter

from pydantic_ai import Agent

from .models import SlideRewrite, SlideRewriteResponse, SlideType
from .utilities.model_config import (
    REWRITER_MODEL,
    REWRITER_MODEL_RPD,
    REWRITER_MODEL_RPM,
    WINDOW_SECONDS
)
from .utilities.model_retry import run_agent_request
from .utilities.normalizer import normalize
from .utilities.prompts import REWRITER_SYSTEM_PROMPT
from .utilities.rate_limit import DailyQuotaExceededError, RequestPacer

MINIMUM_SOURCE_WORD_COVERAGE = 0.65
WORD_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9'-]*")
rewriter_agents: dict[str, Agent] = {}


class LLMRewriter:
    """Extract structured notes directly from one source slide."""

    def __init__(self) -> None:
        """Initialize model pacing for rewrite requests."""
        self.pacer = RequestPacer(WINDOW_SECONDS)

    def run_rewriter_request(self, prompt: str) -> SlideRewriteResponse:
        """Call the direct-extraction model with shared rate limiting and retries."""
        return run_agent_request(
            pacer=self.pacer,
            request_name="rewriter",
            model_name=REWRITER_MODEL,
            rpm=REWRITER_MODEL_RPM,
            rpd=REWRITER_MODEL_RPD,
            agent_cache=rewriter_agents,
            output_type=SlideRewriteResponse,
            instructions=REWRITER_SYSTEM_PROMPT,
            prompt=prompt,
            window_seconds=WINDOW_SECONDS
        )

    def extract_body_text(self, slide_text: str, title: str | None) -> str:
        """Remove an isolated title and trailing page number from normalized text."""
        lines = [
            line.strip()
            for line in normalize(slide_text).splitlines()
            if line.strip()
        ]
        if title and lines and lines[0] == title:
            lines = lines[1:]
        if lines and lines[-1].isdigit():
            lines = lines[:-1]
        return "\n".join(lines).strip()

    def get_source_title(self, slide_text: str) -> str | None:
        """Return a plausible verbatim title from the first normalized line."""
        first_line = normalize(slide_text).split("\n", 1)[0].strip()
        return first_line if first_line and len(first_line) <= 160 else None

    def validate_title(self, slide_text: str, proposed_title: str | None) -> str | None:
        """Keep only a title that occurs verbatim in the source slide."""
        normalized_text = normalize(slide_text)
        if proposed_title and proposed_title in normalized_text:
            return proposed_title
        return self.get_source_title(slide_text)

    def build_fallback_output(self, slide_number: int, slide_text: str) -> SlideRewrite:
        """Preserve normalized source text when direct extraction is unavailable."""
        title = self.get_source_title(slide_text)
        body_text = self.extract_body_text(slide_text, title)
        slide_type = SlideType.CONTENT if body_text else SlideType.INTRODUCTION
        return SlideRewrite(
            slide_number=slide_number,
            slide_type=slide_type,
            title=title,
            is_continuation=False,
            text=body_text,
            rewrite_mode="direct_fallback_v4",
        )

    def calculate_source_coverage(self, source_text: str, rewritten_text: str) -> float:
        """Measure how many significant source tokens remain in a proposed rewrite."""
        source_words = [
            word.casefold()
            for word in WORD_PATTERN.findall(source_text)
            if len(word) > 2
        ]
        rewritten_words = [
            word.casefold()
            for word in WORD_PATTERN.findall(rewritten_text)
            if len(word) > 2
        ]
        if not source_words:
            return 1.0
        source_counts = Counter(source_words)
        rewritten_counts = Counter(rewritten_words)
        preserved_count = sum(
            min(count, rewritten_counts[word])
            for word, count in source_counts.items()
        )
        return preserved_count / len(source_words)

    def rewrite_one(self, slide_number: int, slide_text: str) -> SlideRewrite:
        """Extract complete structured notes from one slide in a single model call."""
        fallback = self.build_fallback_output(slide_number, slide_text)
        if len(normalize(slide_text).strip()) < 10:
            print(f"[slide {slide_number:03d}] skipped model request (empty)")
            return fallback

        prompt = f"Slide number: {slide_number}\n\nSlide text:\n\n{normalize(slide_text)}"
        print(f"[slide {slide_number:03d}] extracting notes directly...")

        try:
            response = self.run_rewriter_request(prompt)
        except DailyQuotaExceededError:
            raise
        except Exception as error:
            print(
                f"[warn] Slide {slide_number:03d} extraction failed; "
                f"preserving source text ({error})"
            )
            return fallback

        title = self.validate_title(slide_text, response.title)
        rewritten_text = response.text.strip()
        source_body = self.extract_body_text(slide_text, title)
        source_coverage = self.calculate_source_coverage(source_body, rewritten_text)
        substantive_slide = response.slide_type in {
            SlideType.CONTENT,
            SlideType.IMAGE_DESCRIPTION
        }
        insufficient_coverage = source_coverage < MINIMUM_SOURCE_WORD_COVERAGE
        if substantive_slide and (not rewritten_text or insufficient_coverage):
            print(
                f"[warn] Slide {slide_number:03d} extraction failed source "
                f"coverage ({source_coverage:.0%}); preserving source text"
            )
            return fallback

        empty_introductory_slide = (
            response.slide_type in {SlideType.INTRODUCTION, SlideType.COURSE_INFO}
            and not source_body
        )
        if empty_introductory_slide:
            rewritten_text = ""

        print(f"[slide {slide_number:03d}] extracted - {response.slide_type.value}")
        return SlideRewrite(
            slide_number=slide_number,
            slide_type=response.slide_type,
            title=title,
            is_continuation=response.is_continuation,
            text=rewritten_text,
            rewrite_mode="direct_extraction_v4",
        )
