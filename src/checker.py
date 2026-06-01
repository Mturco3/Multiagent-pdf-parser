import json
import time

from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError

from .models import ReviewApprovalResponse, SlideReview, SlideReviewResponse, SlideType
from .utilities.model_config import CHECKER_MODEL, CHECKER_MODEL_RPM, REVIEWER_MODEL, REVIEWER_MODEL_RPM, WINDOW_SECONDS
from .utilities.normalizer import looks_like_raw_slide_block, normalize
from .utilities.prompts import CHECKER_REVIEWER_PROMPT, CHECKER_SYSTEM_PROMPT
from .utilities.rate_limit import RequestPacer

MAX_CHECKER_ATTEMPTS = 3
MAX_MODEL_RETRIES = 3
TRANSIENT_STATUS_CODES = {429, 503}

checker_agent = Agent(
    CHECKER_MODEL,
    output_type=SlideReviewResponse,
    instructions=CHECKER_SYSTEM_PROMPT,
    model_settings={"temperature": 0},
)
reviewer_agent = Agent(
    REVIEWER_MODEL,
    output_type=ReviewApprovalResponse,
    instructions=CHECKER_REVIEWER_PROMPT,
    model_settings={"temperature": 0},
)


class LLMChecker:
    """Sends each slide's text to the LLM and collects structured reviews."""

    def __init__(self):
        """Track checker/reviewer model calls using per-model rate buckets."""
        self.pacer = RequestPacer(WINDOW_SECONDS)

    def get_retry_delay(self, error: ModelHTTPError) -> float:
        """Extract a useful retry delay from the provider response when available."""
        retry_delay = None
        body = error.body
        if isinstance(body, dict):
            error_payload = body.get("error")
            if isinstance(error_payload, dict):
                details = error_payload.get("details")
                if isinstance(details, list):
                    for detail in details:
                        if not isinstance(detail, dict):
                            continue
                        retry_text = detail.get("retryDelay")
                        if isinstance(retry_text, str) and retry_text.endswith("s"):
                            try:
                                retry_delay = float(retry_text[:-1])
                                break
                            except ValueError:
                                continue

        if retry_delay is not None:
            return max(retry_delay, 1.0)

        if error.status_code == 429:
            return float(WINDOW_SECONDS)
        return 30.0

    def run_with_transient_retry(self, request_name: str, model_name: str, rpm: int, runner, prompt: str):
        """Retry transient provider failures with backoff while preserving deterministic prompts."""
        attempt = 0
        while True:
            self.pacer.wait_for_capacity(model_name, rpm)
            try:
                result = runner(prompt)
                self.pacer.mark_request(model_name)
                return result.output
            except ModelHTTPError as error:
                if error.status_code not in TRANSIENT_STATUS_CODES or attempt >= MAX_MODEL_RETRIES - 1:
                    raise

                delay = self.get_retry_delay(error)
                attempt += 1
                print(f"{request_name} transient error {error.status_code} - retrying in {delay:.1f}s ({attempt}/{MAX_MODEL_RETRIES})...")
                time.sleep(delay)
                self.pacer.reset(model_name, rpm)

    def run_checker_request(self, prompt: str) -> SlideReviewResponse:
        """Call the checker model while respecting the per-minute request budget."""
        return self.run_with_transient_retry("checker", CHECKER_MODEL, CHECKER_MODEL_RPM, checker_agent.run_sync, prompt)

    def run_reviewer_request(self, prompt: str) -> ReviewApprovalResponse:
        """Call the reviewer model while respecting the per-minute request budget."""
        return self.run_with_transient_retry("reviewer", REVIEWER_MODEL, REVIEWER_MODEL_RPM, reviewer_agent.run_sync, prompt)

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
