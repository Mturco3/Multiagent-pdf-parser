import json

from pydantic_ai import Agent

from .models import ReviewApprovalResponse, SlideReview, SlideReviewResponse, SlideType
from .utilities.model_config import (
    CHECKER_FALLBACK_MODEL,
    CHECKER_FALLBACK_MODEL_RPD,
    CHECKER_FALLBACK_MODEL_RPM,
    CHECKER_MODEL,
    CHECKER_MODEL_RPD,
    CHECKER_MODEL_RPM,
    REVIEWER_FALLBACK_MODEL,
    REVIEWER_FALLBACK_MODEL_RPD,
    REVIEWER_FALLBACK_MODEL_RPM,
    REVIEWER_MODEL,
    REVIEWER_MODEL_RPD,
    REVIEWER_MODEL_RPM,
    WINDOW_SECONDS,
)
from .utilities.model_retry import ModelRequestCandidate, get_cached_agent, run_with_transient_retry_and_fallback
from .utilities.normalizer import looks_like_raw_slide_block, normalize
from .utilities.prompts import CHECKER_REVIEWER_PROMPT, CHECKER_SYSTEM_PROMPT
from .utilities.rate_limit import RequestPacer

MAX_CHECKER_ATTEMPTS = 3

checker_agents: dict[str, Agent] = {}
reviewer_agents: dict[str, Agent] = {}


class LLMChecker:
    """Sends each slide's text to the LLM and collects structured reviews."""

    def __init__(self):
        """Track checker/reviewer model calls using per-model rate buckets."""
        self.pacer = RequestPacer(WINDOW_SECONDS)

    def run_agent_request(self, request_name: str, candidates: list[ModelRequestCandidate], prompt: str):
        """Run a model request with transient retry and 503 fallback handling."""
        return run_with_transient_retry_and_fallback(self.pacer, request_name, candidates, prompt, WINDOW_SECONDS)

    def run_checker_request(self, prompt: str) -> SlideReviewResponse:
        """Call the checker model while respecting the per-minute request budget."""
        candidates = [
            ModelRequestCandidate(
                CHECKER_MODEL,
                CHECKER_MODEL_RPM,
                CHECKER_MODEL_RPD,
                lambda text: get_cached_agent(checker_agents, CHECKER_MODEL, SlideReviewResponse, CHECKER_SYSTEM_PROMPT).run_sync(text).output,
            )
        ]
        if CHECKER_FALLBACK_MODEL:
            candidates.append(
                ModelRequestCandidate(
                    CHECKER_FALLBACK_MODEL,
                    CHECKER_FALLBACK_MODEL_RPM,
                    CHECKER_FALLBACK_MODEL_RPD,
                    lambda text: get_cached_agent(
                        checker_agents,
                        CHECKER_FALLBACK_MODEL,
                        SlideReviewResponse,
                        CHECKER_SYSTEM_PROMPT,
                    ).run_sync(text).output,
                )
            )
        return self.run_agent_request("checker", candidates, prompt)

    def run_reviewer_request(self, prompt: str) -> ReviewApprovalResponse:
        """Call the reviewer model while respecting the per-minute request budget."""
        candidates = [
            ModelRequestCandidate(
                REVIEWER_MODEL,
                REVIEWER_MODEL_RPM,
                REVIEWER_MODEL_RPD,
                lambda text: get_cached_agent(
                    reviewer_agents,
                    REVIEWER_MODEL,
                    ReviewApprovalResponse,
                    CHECKER_REVIEWER_PROMPT,
                ).run_sync(text).output,
            )
        ]
        if REVIEWER_FALLBACK_MODEL:
            candidates.append(
                ModelRequestCandidate(
                    REVIEWER_FALLBACK_MODEL,
                    REVIEWER_FALLBACK_MODEL_RPM,
                    REVIEWER_FALLBACK_MODEL_RPD,
                    lambda text: get_cached_agent(
                        reviewer_agents,
                        REVIEWER_FALLBACK_MODEL,
                        ReviewApprovalResponse,
                        CHECKER_REVIEWER_PROMPT,
                    ).run_sync(text).output,
                )
            )
        return self.run_agent_request("reviewer", candidates, prompt)

    def build_checker_prompt(self, normalized_text: str, retry_instruction: str | None = None) -> str:
        """Build the checker prompt, optionally including reviewer feedback from a failed attempt."""
        raw_block_hint = "yes" if looks_like_raw_slide_block(normalized_text) else "no"
        prompt = f"Slide text:\n\n{normalized_text}\n\nStructural hints:\n- raw_slide_block_detected: {raw_block_hint}"
        if retry_instruction:
            prompt += (
                "\n\nReviewer feedback from the previous attempt:\n"
                f"{retry_instruction}\n\n"
                "Return a corrected structured review. Keep every original_fragment as an exact excerpt from the slide text. "
                "Use only these action types when needed: insert_connectivity, remove_personal_pronouns, flatten_bullets, define_acronym, incomplete_sentence."
            )
        return prompt

    def canonicalize_review(self, normalized_text: str, review: SlideReview) -> SlideReview:
        """Deduplicate and sort actions so later application stays stable across runs."""
        unique_actions = []
        seen_actions: set[tuple[str, str]] = set()

        for action in review.actions:
            key = (action.action.value, action.original_fragment)
            if key in seen_actions:
                continue
            seen_actions.add(key)
            unique_actions.append(action)

        def action_sort_key(action) -> tuple[int, str, str]:
            position = normalized_text.find(action.original_fragment)
            if position < 0:
                position = len(normalized_text)
            return (position, action.action.value, action.original_fragment)

        ordered_actions = sorted(unique_actions, key=action_sort_key)
        return review.model_copy(update={"actions": ordered_actions})

    def review_proposal(self, normalized_text: str, review: SlideReview) -> ReviewApprovalResponse:
        """Ask the reviewer agent to accept or reject the proposed review."""
        proposal = {
            "slide_type": review.slide_type.value,
            "title": review.title,
            "is_continuation": review.is_continuation,
            "key_concepts": review.key_concepts,
            "summary": review.summary,
            "actions": [action.model_dump(mode="json") for action in review.actions],
        }
        prompt = (
            f"Slide text:\n\n{normalized_text}\n\n"
            "Proposed review JSON:\n"
            f"{json.dumps(proposal, indent=2, ensure_ascii=False)}"
        )
        return self.run_reviewer_request(prompt)

    def check_one(self, slide_number: int, raw_text: str) -> SlideReview:
        """Review a single slide's text through the checker agent."""
        normalized_text = normalize(raw_text)

        if len(normalized_text.strip()) < 10:
            print(f"[slide {slide_number:03d}] skipped (empty)")
            return SlideReview(
                slide_number=slide_number,
                slide_type=SlideType.INTRODUCTION,
                title=None,
                is_continuation=False,
                key_concepts=[],
                summary=None,
                actions=[],
            )

        retry_instruction: str | None = None
        last_review: SlideReview | None = None

        for attempt in range(1, MAX_CHECKER_ATTEMPTS + 1):
            print(f"[slide {slide_number:03d}] checker attempt {attempt}/{MAX_CHECKER_ATTEMPTS}...")
            checker_output = self.run_checker_request(self.build_checker_prompt(normalized_text, retry_instruction))
            review = SlideReview(slide_number=slide_number, checker_attempts=attempt, **checker_output.model_dump())
            review = self.canonicalize_review(normalized_text, review)

            approval = self.review_proposal(normalized_text, review)
            review = review.model_copy(
                update={
                    "reviewer_approved": approval.approved,
                    "reviewer_feedback": approval.reason,
                    "checker_attempts": attempt,
                }
            )
            last_review = review

            if approval.approved:
                title_display = f'"{review.title}"' if review.title else "no title"
                actions_display = f"{len(review.actions)} action(s)" if review.actions else "no actions"
                print(f"[slide {slide_number:03d}] approved - {review.slide_type.value}, {title_display}, {actions_display}")
                return review

            retry_instruction = approval.retry_instruction or approval.reason or "Tighten the review and remove unsupported actions."
            print(f"[slide {slide_number:03d}] rejected - {retry_instruction}")

        print(f"[slide {slide_number:03d}] review not approved after {MAX_CHECKER_ATTEMPTS} attempts; preserving original text later")
        return last_review if last_review is not None else SlideReview(
            slide_number=slide_number,
            slide_type=SlideType.CONTENT,
            title=None,
            is_continuation=False,
            key_concepts=[],
            summary=None,
            actions=[],
            reviewer_approved=False,
            reviewer_feedback="Checker did not produce an approved action plan.",
            checker_attempts=MAX_CHECKER_ATTEMPTS,
        )

    def check_all(self, slides: list[tuple[int, str]]) -> list[SlideReview]:
        """Review all slides while pacing real checker and reviewer model calls."""
        reviews = []
        total = len(slides)

        for slide_number, text in slides:
            print(f"[{slide_number}/{total}]", end=" ", flush=True)

            reviews.append(self.check_one(slide_number, text))

        return reviews
