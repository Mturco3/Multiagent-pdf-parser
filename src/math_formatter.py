import re
import time

from pydantic_ai import Agent

from .checker import MODEL, WINDOW_SECONDS
from .models import MathReplacementResponse, SlideRewrite
from .utilities.prompts import MATH_FORMATTER_PROMPT

math_agent = Agent(MODEL, output_type=MathReplacementResponse, instructions=MATH_FORMATTER_PROMPT, model_settings={"temperature": 0})


class MathFormatter:
    """Identify mathematical expressions and replace them with LaTeX."""

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
        result = math_agent.run_sync(f"Slide text:\n\n{slide.text}")
        response = result.output

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
        updated_slide = SlideRewrite(slide_number=slide.slide_number, slide_type=slide.slide_type, title=slide.title, is_continuation=slide.is_continuation, text=updated_text)
        return updated_slide, response

    def format_all(self, slides: list[SlideRewrite]) -> tuple[list[SlideRewrite], list[MathReplacementResponse]]:
        """Process all slides for math formatting while respecting rate limits."""
        updated_slides: list[SlideRewrite] = []
        all_responses: list[MathReplacementResponse] = []
        request_count = 0
        batch_start = time.time()

        for slide in slides:
            if slide.text.strip():
                if request_count >= 14 and time.time() - batch_start < WINDOW_SECONDS:
                    remaining = WINDOW_SECONDS - (time.time() - batch_start)
                    print(f"Rate limit reached - waiting {remaining:.1f}s...")
                    time.sleep(remaining)
                    request_count = 0
                    batch_start = time.time()

            updated_slide, response = self.format_slide(slide)

            if slide.text.strip():
                request_count += 1

            updated_slides.append(updated_slide)
            all_responses.append(response)

        return updated_slides, all_responses
