from pydantic_ai import Agent

from .models import RewriteApprovalResponse, SlideReview, SlideRewrite, SlideType
from .utilities.model_config import (
    REVIEWER_MODEL,
    REVIEWER_MODEL_RPD,
    REVIEWER_MODEL_RPM,
    REWRITER_MODEL,
    REWRITER_MODEL_RPD,
    REWRITER_MODEL_RPM,
    WINDOW_SECONDS,
)
from .utilities.model_retry import get_cached_agent, run_with_retry
from .utilities.normalizer import normalize
from .utilities.prompts import REWRITER_SYSTEM_PROMPT, REWRITE_REVIEWER_PROMPT
from .utilities.rate_limit import RequestPacer

MAX_REWRITE_ATTEMPTS = 2
rewriter_agents: dict[str, Agent] = {}
rewrite_reviewer_agents: dict[str, Agent] = {}


class LLMRewriter:
    """Rewrite slide text while preserving structure and content."""

    def __init__(self):
        """Initialize the rewriter with rate-limited model access."""
        self.pacer = RequestPacer(WINDOW_SECONDS)

    def run_rewriter_request(self, prompt: str) -> str:
        """Call the rewrite model with deterministic pacing and transient retries."""
        runner = lambda text: get_cached_agent(rewriter_agents, REWRITER_MODEL, str, REWRITER_SYSTEM_PROMPT).run_sync(text).output
        return run_with_retry(self.pacer, "rewriter", REWRITER_MODEL, REWRITER_MODEL_RPM, REWRITER_MODEL_RPD, runner, prompt, WINDOW_SECONDS)

    def run_rewrite_review_request(self, prompt: str) -> RewriteApprovalResponse:
        """Call the rewrite reviewer model with deterministic pacing and transient retries."""
        runner = lambda text: get_cached_agent(rewrite_reviewer_agents, REVIEWER_MODEL, RewriteApprovalResponse, REWRITE_REVIEWER_PROMPT).run_sync(text).output
        return run_with_retry(self.pacer, "rewrite-reviewer", REVIEWER_MODEL, REVIEWER_MODEL_RPM, REVIEWER_MODEL_RPD, runner, prompt, WINDOW_SECONDS)

    def build_introduction_output(self, review: SlideReview) -> SlideRewrite | None:
        """Return a heading-only output for introduction slides when possible."""
        if not review.title:
            return None

        return SlideRewrite(
            slide_number=review.slide_number,
            slide_type=review.slide_type.value,
            title=review.title,
            is_continuation=False,
            text="",
            rewrite_mode="introduction_heading_v2",
        )

    def extract_body_text(self, slide_text: str, review: SlideReview) -> str:
        """Prepare deterministic slide body text without title or trailing page number noise."""
        lines = [line.strip() for line in normalize(slide_text).splitlines() if line.strip()]

        if review.title and lines and lines[0] == review.title:
            lines = lines[1:]

        if lines and lines[-1].isdigit():
            lines = lines[:-1]

        return "\n".join(lines).strip()

    def build_passthrough_output(self, slide_text: str, review: SlideReview) -> SlideRewrite:
        """Return deterministic output when no approved rewrite should be performed."""
        return SlideRewrite(
            slide_number=review.slide_number,
            slide_type=review.slide_type.value,
            title=review.title,
            is_continuation=review.is_continuation,
            text=self.extract_body_text(slide_text, review),
            rewrite_mode="deterministic_passthrough_v2",
        )

    def review_title_support(self, slide_text: str, title: str | None, rewritten_text: str) -> RewriteApprovalResponse:
        """Ask the reviewer whether the rewritten body is acceptable and whether the title is supported."""
        prompt = (
            f"Original normalized slide text:\n\n{normalize(slide_text)}\n\n"
            f"Proposed title: {title or '(no title)'}\n\n"
            f"Rewritten slide body:\n\n{rewritten_text}"
        )
        return self.run_rewrite_review_request(prompt)

    def finalize_title(self, review: SlideReview, title_review: RewriteApprovalResponse) -> str | None:
        """Drop unsupported titles deterministically after the reviewer verdict."""
        if review.title and not title_review.keep_title:
            print(f"[slide {review.slide_number:03d}] dropping unsupported title")
            return None
        return review.title

    def finalize_passthrough_title(self, slide_text: str, review: SlideReview, fallback: SlideRewrite) -> SlideRewrite:
        """Validate a passthrough title when possible, but keep moving on provider failures."""
        if not fallback.title or not fallback.text:
            return fallback

        try:
            title_review = self.review_title_support(slide_text, review.title, fallback.text)
        except Exception as error:
            print(f"[slide {review.slide_number:03d}] title review failed; keeping passthrough title ({error})")
            return fallback

        return fallback.model_copy(update={"title": self.finalize_title(review, title_review)})

    def rewrite_one(self, slide_text: str, review: SlideReview, previous_paragraph: str | None = None) -> SlideRewrite | None:
        """Rewrite one slide or skip it when the slide should not be sent to the LLM."""
        if review.slide_type in (SlideType.COURSE_INFO, SlideType.IMAGE_DESCRIPTION):
            print(f"[slide {review.slide_number:03d}] skipped ({review.slide_type.value})")
            return None

        if review.slide_type == SlideType.INTRODUCTION:
            print(f"[slide {review.slide_number:03d}] heading only ({review.slide_type.value})")
            return self.build_introduction_output(review)

        if not review.reviewer_approved:
            print(f"[slide {review.slide_number:03d}] review not approved; preserving extracted body text")
            fallback = self.build_passthrough_output(slide_text, review)
            return self.finalize_passthrough_title(slide_text, review, fallback)

        if not review.actions:
            print(f"[slide {review.slide_number:03d}] no approved actions; preserving extracted body text")
            fallback = self.build_passthrough_output(slide_text, review)
            return self.finalize_passthrough_title(slide_text, review, fallback)

        body_text = self.extract_body_text(slide_text, review)
        actions_text = "\n".join(f"- {action.action.value}: \"{action.original_fragment}\"" for action in review.actions)
        retry_instruction: str | None = None

        for attempt in range(1, MAX_REWRITE_ATTEMPTS + 1):
            prompt = f"Slide title: {review.title or '(no title)'}\n\nOriginal slide text:\n\n{body_text}\n\nActions to apply:\n{actions_text}"
            if previous_paragraph:
                prompt += f"\n\nPrevious paragraph (for transition context only - do NOT include this content in your output):\n{previous_paragraph}"
            if retry_instruction:
                prompt += f"\n\nReviewer feedback from the previous attempt:\n{retry_instruction}"

            print(f"[slide {review.slide_number:03d}] rewriting attempt {attempt}/{MAX_REWRITE_ATTEMPTS} ({len(review.actions)} approved action(s))...")
            try:
                rewritten_text = self.run_rewriter_request(prompt)
                title_review = self.review_title_support(slide_text, review.title, rewritten_text)
            except Exception as error:
                print(f"[slide {review.slide_number:03d}] rewrite request failed; falling back to deterministic passthrough ({error})")
                break

            if title_review.approved:
                print(f"[slide {review.slide_number:03d}] done")
                return SlideRewrite(
                    slide_number=review.slide_number,
                    slide_type=review.slide_type.value,
                    title=self.finalize_title(review, title_review),
                    is_continuation=review.is_continuation,
                    text=rewritten_text,
                    rewrite_mode="rewrite_review_v2",
                )

            retry_instruction = title_review.retry_instruction or title_review.reason or "Preserve the original content more faithfully and remove raw slide artifacts."
            print(f"[slide {review.slide_number:03d}] rewrite rejected - {retry_instruction}")

        print(f"[slide {review.slide_number:03d}] rewrite not approved; falling back to deterministic passthrough")
        fallback = self.build_passthrough_output(slide_text, review)
        return self.finalize_passthrough_title(slide_text, review, fallback)
