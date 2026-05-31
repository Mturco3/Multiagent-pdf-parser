import re
import time

from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError

from .checker import MODEL, WINDOW_SECONDS
from .models import MathReplacementResponse, SlideRewrite
from .utilities.prompts import MATH_FORMATTER_PROMPT

math_agent = Agent(MODEL, output_type=MathReplacementResponse, instructions=MATH_FORMATTER_PROMPT, model_settings={"temperature": 0})
MATH_MODEL_RPM = 12
MAX_MODEL_RETRIES = 3
TRANSIENT_STATUS_CODES = {429, 503}


class MathFormatter:
    """Identify mathematical expressions and replace them with LaTeX."""

    def __init__(self):
        """Track math-model calls so long runs can pace and retry safely."""
        self.request_count = 0
        self.batch_start = time.time()

    def wait_for_capacity(self):
        """Pause before the next math-model call when the current quota window is exhausted."""
        elapsed = time.time() - self.batch_start
        if elapsed >= WINDOW_SECONDS:
            self.request_count = 0
            self.batch_start = time.time()
            return

        if self.request_count >= MATH_MODEL_RPM:
            remaining = WINDOW_SECONDS - elapsed
            print(f"Rate limit reached - waiting {remaining:.1f}s...")
            time.sleep(remaining)
            self.request_count = 0
            self.batch_start = time.time()

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

    def run_math_request(self, prompt: str) -> MathReplacementResponse:
        """Call the math model with pacing and retries for transient provider failures."""
        attempt = 0
        while True:
            self.wait_for_capacity()
            try:
                result = math_agent.run_sync(prompt)
                self.request_count += 1
                return result.output
            except ModelHTTPError as error:
                if error.status_code not in TRANSIENT_STATUS_CODES or attempt >= MAX_MODEL_RETRIES - 1:
                    raise

                delay = self.get_retry_delay(error)
                attempt += 1
                print(f"math transient error {error.status_code} - retrying in {delay:.1f}s ({attempt}/{MAX_MODEL_RETRIES})...")
                time.sleep(delay)
                self.request_count = 0
                self.batch_start = time.time()

    def apply_replacement(self, text: str, original_text: str, latex: str) -> tuple[str, bool]:
        """Replace one standalone math fragment without interpreting LaTeX escapes."""
        pattern = r"(?<!\w)" + re.escape(original_text) + r"(?!\w)"
        if not re.search(pattern, text):
            return text, False

        updated_text = re.sub(pattern, lambda _: latex, text, count=1)
        return updated_text, True

    def format_slide(self, slide: SlideRewrite) -> tuple[SlideRewrite, MathReplacementResponse]:
        """Identify math in one slide and apply the requested replacements."""
        if not slide.text.strip():
            print(f"[slide {slide.slide_number:03d}] skipped (no body text)")
            return slide, MathReplacementResponse(replacements=[])

        print(f"[slide {slide.slide_number:03d}] identifying math...")
        response = self.run_math_request(f"Slide text:\n\n{slide.text}")

        if not response.replacements:
            print(f"[slide {slide.slide_number:03d}] no math found")
            return slide, response

        updated_text = slide.text
        applied_count = 0
        for replacement in response.replacements:
            updated_text, was_applied = self.apply_replacement(updated_text, replacement.original_text, replacement.latex)
            if was_applied:
                applied_count += 1
                continue

            print(f"[slide {slide.slide_number:03d}] skipped replacement: \"{replacement.original_text}\" (no standalone match)")

        print(f"[slide {slide.slide_number:03d}] applied {applied_count} replacement(s)")
        updated_slide = SlideRewrite(
            slide_number=slide.slide_number,
            slide_type=slide.slide_type,
            title=slide.title,
            is_continuation=slide.is_continuation,
            text=updated_text,
            rewrite_mode=slide.rewrite_mode,
        )
        return updated_slide, response

    def format_all(self, slides: list[SlideRewrite]) -> tuple[list[SlideRewrite], list[MathReplacementResponse]]:
        """Process all slides for math formatting while respecting rate limits."""
        updated_slides: list[SlideRewrite] = []
        all_responses: list[MathReplacementResponse] = []

        for slide in slides:
            updated_slide, response = self.format_slide(slide)

            updated_slides.append(updated_slide)
            all_responses.append(response)

        return updated_slides, all_responses
