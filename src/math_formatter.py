import re

from pydantic_ai import Agent

from .models import MathReplacementResponse, SlideRewrite
from .utilities.model_config import MATH_MODEL, MATH_MODEL_RPD, MATH_MODEL_RPM, WINDOW_SECONDS
from .utilities.model_retry import get_cached_agent, run_with_retry
from .utilities.prompts import MATH_FORMATTER_PROMPT
from .utilities.rate_limit import RequestPacer

math_agents: dict[str, Agent] = {}
MATH_CANDIDATE_PATTERN = re.compile(r"[=<>±×÷∑∫√^_]|\b(?:alpha|beta|gamma|delta|epsilon|lambda|mu|sigma|theta)\b|\d+\s*[+*/-]\s*\d+", re.IGNORECASE)


class MathFormatter:
    """Identify mathematical expressions and replace them with LaTeX."""

    def __init__(self):
        """Track math-model calls so long runs can pace and retry safely."""
        self.pacer = RequestPacer(WINDOW_SECONDS)

    def run_math_request(self, prompt: str) -> MathReplacementResponse:
        """Call the math model with pacing and retries for transient provider failures."""
        runner = lambda active_model, text: get_cached_agent(math_agents, active_model, MathReplacementResponse, MATH_FORMATTER_PROMPT).run_sync(text).output
        return run_with_retry(self.pacer, "math", MATH_MODEL, MATH_MODEL_RPM, MATH_MODEL_RPD, runner, prompt, WINDOW_SECONDS)

    def has_math_candidate(self, text: str) -> bool:
        """Return whether deterministic syntax indicates a possible math expression."""
        return bool(MATH_CANDIDATE_PATTERN.search(text))

    def apply_replacement(self, text: str, original_text: str, latex: str) -> tuple[str, bool]:
        """Replace one standalone math fragment outside existing math spans."""
        pattern = r"(?<![\w.])" + re.escape(original_text) + r"(?![\w.])"
        for match in re.finditer(pattern, text):
            if self.is_inside_math_span(text, match.start()):
                continue

            updated_text = text[:match.start()] + latex + text[match.end():]
            return updated_text, True

        return text, False

    def is_inside_math_span(self, text: str, position: int) -> bool:
        """Return whether a character offset is inside a markdown math span."""
        prefix = text[:position]
        # Check display math ($$) first by removing matched $$ pairs
        outside_display = re.sub(r"\$\$[^$]*\$\$", "", prefix)
        # An unmatched $$ means we are inside a display math block
        if outside_display.count("$$") % 2 == 1:
            return True
        # Remove display math pairs, then check inline math ($)
        no_display = re.sub(r"\$\$", "", prefix)
        return no_display.count("$") % 2 == 1

    def normalize_latex(self, latex: str, is_display: bool) -> str:
        """Normalize model-produced LaTeX delimiters before inserting it."""
        stripped = latex.strip()
        body = stripped
        if body.startswith("$$") and body.endswith("$$") and len(body) >= 4:
            body = body[2:-2].strip()
        elif body.startswith("$") and body.endswith("$") and len(body) >= 2:
            body = body[1:-1].strip()

        if not body:
            return stripped

        display_markers = ("=", "<", ">", r"\sum", r"\int", r"\frac", r"\begin")
        should_display = is_display and any(marker in body for marker in display_markers)
        if should_display:
            return f"$${body}$$"
        return f"${body}$"

    def format_slide(self, slide: SlideRewrite) -> tuple[SlideRewrite, MathReplacementResponse]:
        """Identify math in one slide and apply the requested replacements."""
        if not slide.text.strip():
            print(f"[slide {slide.slide_number:03d}] skipped (no body text)")
            return slide, MathReplacementResponse(replacements=[])

        if not self.has_math_candidate(slide.text):
            print(f"[slide {slide.slide_number:03d}] skipped (no math candidates)")
            return slide, MathReplacementResponse(replacements=[])

        print(f"[slide {slide.slide_number:03d}] identifying math...")
        response = self.run_math_request(f"Slide text:\n\n{slide.text}")

        if not response.replacements:
            print(f"[slide {slide.slide_number:03d}] no math found")
            return slide, response

        updated_text = slide.text
        applied_count = 0
        replacements = sorted(response.replacements, key=lambda item: len(item.original_text), reverse=True)
        for replacement in replacements:
            latex = self.normalize_latex(replacement.latex, replacement.is_display)
            updated_text, was_applied = self.apply_replacement(updated_text, replacement.original_text, latex)
            if was_applied:
                applied_count += 1
                continue

            print(f"[slide {slide.slide_number:03d}] skipped replacement: \"{replacement.original_text}\" (no standalone match)")

        print(f"[slide {slide.slide_number:03d}] applied {applied_count} replacement(s)")
        updated_slide = SlideRewrite(slide_number=slide.slide_number, slide_type=slide.slide_type, title=slide.title, is_continuation=slide.is_continuation, text=updated_text, rewrite_mode=slide.rewrite_mode)
        return updated_slide, response
