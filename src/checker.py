"""Classify individual slides and validate model-suggested edit actions.

The module contains ``LLMChecker`` and its shared agent cache. This checker is
retained for compatibility but is not part of the current direct-rewrite flow.
"""

from pydantic_ai import Agent

from .models import SlideReview, SlideReviewResponse, SlideType, SuggestedAction
from .utilities.model_config import (
    CHECKER_MODEL,
    CHECKER_MODEL_RPD,
    CHECKER_MODEL_RPM,
    WINDOW_SECONDS
)
from .utilities.model_retry import run_agent_request
from .utilities.normalizer import looks_like_raw_slide_block, normalize
from .utilities.prompts import CHECKER_SYSTEM_PROMPT
from .utilities.rate_limit import RequestPacer

checker_agents: dict[str, Agent] = {}


class LLMChecker:
    """Classify slides and return deterministically validated edit actions."""

    def __init__(self) -> None:
        """Initialize model pacing for checker requests."""
        self.pacer = RequestPacer(WINDOW_SECONDS)

    def run_checker_request(self, prompt: str) -> SlideReviewResponse:
        """Call the checker model with shared rate limiting and retries."""
        return run_agent_request(
            pacer=self.pacer,
            request_name="checker",
            model_name=CHECKER_MODEL,
            rpm=CHECKER_MODEL_RPM,
            rpd=CHECKER_MODEL_RPD,
            agent_cache=checker_agents,
            output_type=SlideReviewResponse,
            instructions=CHECKER_SYSTEM_PROMPT,
            prompt=prompt,
            window_seconds=WINDOW_SECONDS
        )

    def build_checker_prompt(self, raw_text: str, normalized_text: str) -> str:
        """Build a compact checker prompt with deterministic structural hints."""
        raw_block_hint = "yes" if looks_like_raw_slide_block(raw_text) else "no"
        return (
            f"Slide text:\n\n{normalized_text}\n\n"
            f"Structural hints:\n- raw_slide_block_detected: {raw_block_hint}"
        )

    def validate_title(
        self,
        normalized_text: str,
        proposed_title: str | None
    ) -> str | None:
        """Accept an exact title or fall back to the isolated first text block."""
        if proposed_title and proposed_title in normalized_text:
            return proposed_title
        first_block = normalized_text.split("\n", 1)[0].strip()
        if first_block and len(first_block) <= 160:
            return first_block
        return None

    def canonicalize_review(
        self,
        normalized_text: str,
        review: SlideReview
    ) -> SlideReview:
        """Remove unsupported actions and order valid unique actions by source position."""
        valid_actions: list[SuggestedAction] = []
        seen_actions: set[tuple[str, str]] = set()

        for action in review.actions:
            if action.original_fragment not in normalized_text:
                print(
                    f"[slide {review.slide_number:03d}] dropped action with "
                    "a non-source fragment"
                )
                continue
            action_key = action.action.value, action.original_fragment
            if action_key in seen_actions:
                continue
            seen_actions.add(action_key)
            valid_actions.append(action)

        def action_sort_key(action: SuggestedAction) -> tuple[int, str, str]:
            """Sort actions by source position and stable action metadata."""
            position = normalized_text.find(action.original_fragment)
            return position, action.action.value, action.original_fragment

        ordered_actions = sorted(valid_actions, key=action_sort_key)
        validated_title = self.validate_title(normalized_text, review.title)
        return review.model_copy(update={"title": validated_title, "actions": ordered_actions})

    def check_one(self, slide_number: int, raw_text: str) -> SlideReview:
        """Classify and validate one slide without a redundant reviewer call."""
        normalized_text = normalize(raw_text)
        if len(normalized_text.strip()) < 10:
            print(f"[slide {slide_number:03d}] skipped (empty)")
            return SlideReview(
                slide_number=slide_number,
                slide_type=SlideType.INTRODUCTION,
                title=None,
                is_continuation=False,
                actions=[]
            )

        print(f"[slide {slide_number:03d}] checking...")
        prompt = self.build_checker_prompt(raw_text, normalized_text)
        checker_output = self.run_checker_request(prompt)
        review = SlideReview(slide_number=slide_number, **checker_output.model_dump())
        review = self.canonicalize_review(normalized_text, review)
        title_display = f'"{review.title}"' if review.title else "no title"
        actions_display = (
            f"{len(review.actions)} action(s)"
            if review.actions
            else "no actions"
        )
        print(
            f"[slide {slide_number:03d}] checked - {review.slide_type.value}, "
            f"{title_display}, {actions_display}"
        )
        return review
