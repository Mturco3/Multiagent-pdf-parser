"""Detect mathematical expressions and convert them to safe LaTeX markup.

The module contains ``MathFormatter``, its lightweight candidate filter, and
helpers that apply model-proposed replacements without touching existing math.
"""

import re

from pydantic_ai import Agent

from .models import MathReplacementResponse, SlideRewrite
from .utilities.model_config import (
    MATH_MODEL,
    MATH_MODEL_RPD,
    MATH_MODEL_RPM,
    WINDOW_SECONDS
)
from .utilities.model_retry import run_agent_request
from .utilities.prompts import MATH_FORMATTER_PROMPT
from .utilities.rate_limit import RequestPacer

math_agents: dict[str, Agent] = {}
MATH_CANDIDATE_PATTERN = re.compile(
    r"[=<>±×÷∑∫√^_]"
    r"|\b(?:alpha|beta|gamma|delta|epsilon|lambda|mu|sigma|theta)\b"
    r"|\d+\s*[+*/-]\s*\d+",
    re.IGNORECASE
)


class MathFormatter:
    """Identify mathematical expressions and replace them with LaTeX."""

    def __init__(self):
        """Track math-model calls so long runs can pace and retry safely."""
        self.pacer = RequestPacer(WINDOW_SECONDS)

    def run_math_request(self, prompt: str) -> MathReplacementResponse:
        """Call the math model with shared pacing and transient retries."""
        return run_agent_request(
            pacer=self.pacer,
            request_name="math",
            model_name=MATH_MODEL,
            rpm=MATH_MODEL_RPM,
            rpd=MATH_MODEL_RPD,
            agent_cache=math_agents,
            output_type=MathReplacementResponse,
            instructions=MATH_FORMATTER_PROMPT,
            prompt=prompt,
            window_seconds=WINDOW_SECONDS
        )

    def has_math_candidate(self, text: str) -> bool:
        """Return whether deterministic syntax indicates possible mathematics."""
        return bool(MATH_CANDIDATE_PATTERN.search(text))

    def apply_replacement(
        self,
        text: str,
        original_text: str,
        latex: str
    ) -> tuple[str, bool]:
        """Replace one standalone math fragment outside existing math spans."""
        pattern = r"(?<![\w.])" + re.escape(original_text) + r"(?![\w.])"
        for match in re.finditer(pattern, text):
            if self.is_inside_math_span(text, match.start()):
                continue

            updated_text = text[:match.start()] + latex + text[match.end():]
            return updated_text, True

        return text, False

    def is_inside_math_span(self, text: str, position: int) -> bool:
        """Return whether a character offset is inside a Markdown math span."""
        prefix = text[:position]
        # Paired display spans must not affect the unmatched-delimiter check.
        outside_display = re.sub(r"\$\$[^$]*\$\$", "", prefix)
        if outside_display.count("$$") % 2 == 1:
            return True

        no_display = re.sub(r"\$\$", "", prefix)
        return no_display.count("$") % 2 == 1

    def normalize_latex(self, latex: str, is_display: bool) -> str:
        """Normalize model-produced LaTeX delimiters before inserting it."""
        stripped_latex = latex.strip()
        latex_body = stripped_latex

        if latex_body.startswith("$$") and latex_body.endswith("$$") and len(latex_body) >= 4:
            latex_body = latex_body[2:-2].strip()
        elif latex_body.startswith("$") and latex_body.endswith("$") and len(latex_body) >= 2:
            latex_body = latex_body[1:-1].strip()

        if not latex_body:
            return stripped_latex

        display_markers = ("=", "<", ">", r"\sum", r"\int", r"\frac", r"\begin")
        should_display = is_display and any(
            marker in latex_body for marker in display_markers
        )
        if should_display:
            return f"$${latex_body}$$"
        return f"${latex_body}$"

    def format_slide(
        self,
        slide: SlideRewrite
    ) -> tuple[SlideRewrite, MathReplacementResponse]:
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
        replacements = sorted(
            response.replacements,
            key=lambda item: len(item.original_text),
            reverse=True
        )
        for replacement in replacements:
            latex = self.normalize_latex(replacement.latex, replacement.is_display)
            updated_text, was_applied = self.apply_replacement(
                updated_text,
                replacement.original_text,
                latex
            )
            if was_applied:
                applied_count += 1
                continue

            print(
                f"[slide {slide.slide_number:03d}] skipped replacement: "
                f"\"{replacement.original_text}\" (no standalone match)"
            )

        print(f"[slide {slide.slide_number:03d}] applied {applied_count} replacement(s)")
        updated_slide = SlideRewrite(
            slide_number=slide.slide_number,
            slide_type=slide.slide_type,
            title=slide.title,
            is_continuation=slide.is_continuation,
            text=updated_text,
            rewrite_mode=slide.rewrite_mode
        )
        return updated_slide, response
