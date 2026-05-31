import time

from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError

from .models import SlideReview, SlideRewrite, SlideType
from .normalizer import normalize
from .utilities.model_config import REWRITER_MODEL, REWRITER_MODEL_RPM, WINDOW_SECONDS
from .utilities.prompts import REWRITER_SYSTEM_PROMPT
from .utilities.rate_limit import RequestPacer

rewriter_agent = Agent(REWRITER_MODEL, output_type=str, instructions=REWRITER_SYSTEM_PROMPT, model_settings={"temperature": 0})
MAX_MODEL_RETRIES = 3
TRANSIENT_STATUS_CODES = {429, 503}


class LLMRewriter:
    """Rewrite slide text while preserving structure and content."""

    def __init__(self):
        self.pacer = RequestPacer(WINDOW_SECONDS)

    def get_retry_delay(self, error: ModelHTTPError) -> float:
        """Extract a provider-suggested retry delay when one is available."""
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

    def run_rewriter_request(self, prompt: str) -> str:
        """Call the rewrite model with deterministic pacing and transient retries."""
        attempt = 0
        while True:
            self.pacer.wait_for_capacity(REWRITER_MODEL, REWRITER_MODEL_RPM)
            try:
                result = rewriter_agent.run_sync(prompt)
                self.pacer.mark_request(REWRITER_MODEL)
                return result.output
            except ModelHTTPError as error:
                if error.status_code not in TRANSIENT_STATUS_CODES or attempt >= MAX_MODEL_RETRIES - 1:
                    raise

                delay = self.get_retry_delay(error)
                attempt += 1
                print(f"rewriter transient error {error.status_code} - retrying in {delay:.1f}s ({attempt}/{MAX_MODEL_RETRIES})...")
                time.sleep(delay)
                self.pacer.reset(REWRITER_MODEL, REWRITER_MODEL_RPM)

    def build_introduction_output(self, review: SlideReview) -> SlideRewrite | None:
        """Return a heading-only output for introduction slides when possible."""
        if not review.title:
            return None

        return SlideRewrite(slide_number=review.slide_number, slide_type=review.slide_type.value, title=review.title, is_continuation=False, text="")

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
        )

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
            return self.build_passthrough_output(slide_text, review)

        if not review.actions:
            print(f"[slide {review.slide_number:03d}] no approved actions; preserving extracted body text")
            return self.build_passthrough_output(slide_text, review)

        body_text = self.extract_body_text(slide_text, review)
        actions_text = "\n".join(f"- {action.action.value}: \"{action.original_fragment}\"" for action in review.actions)

        prompt = f"Slide title: {review.title or '(no title)'}\n\nOriginal slide text:\n\n{body_text}\n\nActions to apply:\n{actions_text}"
        if previous_paragraph:
            prompt += f"\n\nPrevious paragraph (for transition context only - do NOT include this content in your output):\n{previous_paragraph}"

        print(f"[slide {review.slide_number:03d}] rewriting ({len(review.actions)} approved action(s))...")
        rewritten_text = self.run_rewriter_request(prompt)
        print(f"[slide {review.slide_number:03d}] done")

        return SlideRewrite(
            slide_number=review.slide_number,
            slide_type=review.slide_type.value,
            title=review.title,
            is_continuation=review.is_continuation,
            text=rewritten_text,
        )

    def rewrite_all(self, slides: list[tuple[int, str]], reviews: list[SlideReview]) -> list[SlideRewrite]:
        """Rewrite all slides that should appear in the final document."""
        review_by_number = {review.slide_number: review for review in reviews}
        results: list[SlideRewrite] = []
        previous_paragraph: str | None = None
        current_section_title: str | None = None
        total = len(slides)

        for slide_number, text in slides:
            review = review_by_number[slide_number]
            print(f"[{slide_number}/{total}]", end=" ", flush=True)

            transition_paragraph = previous_paragraph
            starts_new_section = bool(review.title and review.title != current_section_title and not review.is_continuation)
            if starts_new_section:
                transition_paragraph = None

            should_call_llm = (
                review.slide_type not in (SlideType.COURSE_INFO, SlideType.IMAGE_DESCRIPTION, SlideType.INTRODUCTION)
                and review.reviewer_approved
                and bool(review.actions)
            )

            rewrite = self.rewrite_one(text, review, transition_paragraph if should_call_llm else None)

            if rewrite is None:
                continue

            results.append(rewrite)

            if rewrite.title and not rewrite.is_continuation:
                current_section_title = rewrite.title

            if rewrite.text:
                previous_paragraph = rewrite.text
            else:
                previous_paragraph = None

        return results
