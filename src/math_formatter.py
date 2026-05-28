import re
import time

from pydantic_ai import Agent

from .checker import MODEL, WINDOW_SECONDS
from .models import MathReplacementResponse, SlideRewrite
from .utilities.prompts import MATH_FORMATTER_PROMPT

# Per-slide agent: identifies math expressions and provides LaTeX equivalents
math_agent = Agent(MODEL, output_type=MathReplacementResponse, instructions=MATH_FORMATTER_PROMPT, model_settings={"temperature": 0})


class MathFormatter:
    """Identifies math expressions per slide and applies LaTeX replacements programmatically."""

    def format_slide(self, slide: SlideRewrite) -> tuple[SlideRewrite, MathReplacementResponse]:
        """Identify math in a single slide and apply LaTeX replacements."""
        print(f"[slide {slide.slide_number:03d}] identifying math...")
        result = math_agent.run_sync(f"Slide text:\n\n{slide.text}")
        response = result.output

        if not response.replacements:
            print(f"[slide {slide.slide_number:03d}] no math found")
            return slide, response

        # Apply replacements programmatically using word boundaries to avoid corrupting words
        updated_text = slide.text
        for replacement in response.replacements:
            pattern = r'(?<!\w)' + re.escape(replacement.original_text) + r'(?!\w)'
            if re.search(pattern, updated_text):
                updated_text = re.sub(pattern, replacement.latex, updated_text, count=1)
            else:
                print(f"[slide {slide.slide_number:03d}] skipped replacement: \"{replacement.original_text}\" (no standalone match)")

        count = len(response.replacements)
        print(f"[slide {slide.slide_number:03d}] applied {count} replacement(s)")

        updated_slide = SlideRewrite(
            slide_number=slide.slide_number,
            slide_type=slide.slide_type,
            title=slide.title,
            is_continuation=slide.is_continuation,
            text=updated_text
        )
        return updated_slide, response

    def format_all(self, slides: list[SlideRewrite]) -> tuple[list[SlideRewrite], list[MathReplacementResponse]]:
        """Process all slides for math formatting with rate limiting."""
        updated_slides: list[SlideRewrite] = []
        all_responses: list[MathReplacementResponse] = []
        request_count = 0
        batch_start = time.time()

        for slide in slides:
            # Pause when the batch limit is reached within the rate window
            if request_count >= 14 and time.time() - batch_start < WINDOW_SECONDS:
                remaining = WINDOW_SECONDS - (time.time() - batch_start)
                print(f"Rate limit reached — waiting {remaining:.1f}s...")
                time.sleep(remaining)
                request_count = 0
                batch_start = time.time()

            updated_slide, response = self.format_slide(slide)
            request_count += 1
            updated_slides.append(updated_slide)
            all_responses.append(response)

        return updated_slides, all_responses
