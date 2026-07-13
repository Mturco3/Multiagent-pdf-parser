import re
from collections import Counter

from pydantic_ai import Agent

from .models import SlideReview, SlideRewrite, SlideType
from .utilities.model_config import REWRITER_MODEL, REWRITER_MODEL_RPD, REWRITER_MODEL_RPM, WINDOW_SECONDS
from .utilities.model_retry import get_cached_agent, run_with_retry
from .utilities.normalizer import normalize
from .utilities.prompts import REWRITER_SYSTEM_PROMPT
from .utilities.rate_limit import DailyQuotaExceededError, RequestPacer

MINIMUM_SOURCE_WORD_COVERAGE = 0.65
WORD_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9'-]*")
rewriter_agents: dict[str, Agent] = {}


class LLMRewriter:
    """Apply validated edit actions while preserving source-slide content."""

    def __init__(self):
        """Initialize model pacing for rewrite requests."""
        self.pacer = RequestPacer(WINDOW_SECONDS)

    def run_rewriter_request(self, prompt: str) -> str:
        """Call the rewrite model with shared rate limiting and retries."""
        runner = lambda active_model, text: get_cached_agent(rewriter_agents, active_model, str, REWRITER_SYSTEM_PROMPT).run_sync(text).output
        return run_with_retry(self.pacer, "rewriter", REWRITER_MODEL, REWRITER_MODEL_RPM, REWRITER_MODEL_RPD, runner, prompt, WINDOW_SECONDS)

    def build_introduction_output(self, review: SlideReview) -> SlideRewrite | None:
        """Return a heading-only output for a titled introduction slide."""
        if not review.title:
            return None
        return SlideRewrite(slide_number=review.slide_number, slide_type=review.slide_type, title=review.title, is_continuation=False, text="", rewrite_mode="introduction_heading_v3")

    def extract_body_text(self, slide_text: str, review: SlideReview) -> str:
        """Remove an isolated title and trailing page number from normalized text."""
        lines = [line.strip() for line in normalize(slide_text).splitlines() if line.strip()]
        if review.title and lines and lines[0] == review.title:
            lines = lines[1:]
        if lines and lines[-1].isdigit():
            lines = lines[:-1]
        return "\n".join(lines).strip()

    def build_passthrough_output(self, slide_text: str, review: SlideReview) -> SlideRewrite:
        """Return deterministic source text when no model rewrite is required."""
        body_text = self.extract_body_text(slide_text, review)
        return SlideRewrite(slide_number=review.slide_number, slide_type=review.slide_type, title=review.title, is_continuation=review.is_continuation, text=body_text, rewrite_mode="deterministic_passthrough_v3")

    def calculate_source_coverage(self, source_text: str, rewritten_text: str) -> float:
        """Measure how many significant source tokens remain in a proposed rewrite."""
        source_words = [word.casefold() for word in WORD_PATTERN.findall(source_text) if len(word) > 2]
        rewritten_words = [word.casefold() for word in WORD_PATTERN.findall(rewritten_text) if len(word) > 2]
        if not source_words:
            return 1.0
        source_counts = Counter(source_words)
        rewritten_counts = Counter(rewritten_words)
        preserved_count = sum(min(count, rewritten_counts[word]) for word, count in source_counts.items())
        return preserved_count / len(source_words)

    def rewrite_one(self, slide_text: str, review: SlideReview) -> SlideRewrite | None:
        """Rewrite one slide only when validated actions require model editing."""
        if review.slide_type == SlideType.COURSE_INFO:
            print(f"[slide {review.slide_number:03d}] skipped ({review.slide_type.value})")
            return None
        if review.slide_type == SlideType.INTRODUCTION:
            print(f"[slide {review.slide_number:03d}] heading only ({review.slide_type.value})")
            return self.build_introduction_output(review)

        fallback = self.build_passthrough_output(slide_text, review)
        if not review.actions:
            print(f"[slide {review.slide_number:03d}] no validated actions; preserving extracted body text")
            return fallback

        actions_text = "\n".join(f'- {action.action.value}: "{action.original_fragment}"' for action in review.actions)
        prompt = f"Slide title: {review.title or '(no title)'}\n\nOriginal slide text:\n\n{fallback.text}\n\nActions to apply:\n{actions_text}"
        print(f"[slide {review.slide_number:03d}] rewriting ({len(review.actions)} validated action(s))...")

        try:
            rewritten_text = self.run_rewriter_request(prompt).strip()
        except DailyQuotaExceededError:
            raise
        except Exception as error:
            print(f"[warn] Slide {review.slide_number:03d} rewrite failed; preserving source text ({error})")
            return fallback

        source_coverage = self.calculate_source_coverage(fallback.text, rewritten_text)
        if not rewritten_text or source_coverage < MINIMUM_SOURCE_WORD_COVERAGE:
            print(f"[warn] Slide {review.slide_number:03d} rewrite failed source coverage ({source_coverage:.0%}); preserving source text")
            return fallback

        print(f"[slide {review.slide_number:03d}] done")
        return SlideRewrite(slide_number=review.slide_number, slide_type=review.slide_type, title=review.title, is_continuation=review.is_continuation, text=rewritten_text, rewrite_mode="validated_rewrite_v3")
