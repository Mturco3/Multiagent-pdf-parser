import re

from pydantic_ai import Agent

from .models import TitleAnalysis, HeadingAction
from .utilities.model_config import (
    TITLE_FALLBACK_MODEL,
    TITLE_FALLBACK_MODEL_RPD,
    TITLE_FALLBACK_MODEL_RPM,
    TITLE_MODEL,
    TITLE_MODEL_RPD,
    TITLE_MODEL_RPM,
    WINDOW_SECONDS,
)
from .utilities.model_retry import ModelRequestCandidate, get_cached_agent, run_with_transient_retry_and_fallback
from .utilities.prompts import TITLE_IDENTIFIER_PROMPT
from .utilities.rate_limit import RequestPacer

title_identifier_agents: dict[str, Agent] = {}


class TitleEditor:
    """Identifies heading changes via LLM, then applies them programmatically."""

    def __init__(self):
        """Initialize the title editor with rate-limited model access."""
        self.pacer = RequestPacer(WINDOW_SECONDS)

    def run_identifier_request(self, prompt: str) -> TitleAnalysis:
        """Call the title model with pacing and transient retry handling."""
        candidates = [
            ModelRequestCandidate(
                TITLE_MODEL,
                TITLE_MODEL_RPM,
                TITLE_MODEL_RPD,
                lambda text: get_cached_agent(
                    title_identifier_agents,
                    TITLE_MODEL,
                    TitleAnalysis,
                    TITLE_IDENTIFIER_PROMPT,
                ).run_sync(text).output,
            )
        ]
        if TITLE_FALLBACK_MODEL:
            candidates.append(
                ModelRequestCandidate(
                    TITLE_FALLBACK_MODEL,
                    TITLE_FALLBACK_MODEL_RPM,
                    TITLE_FALLBACK_MODEL_RPD,
                    lambda text: get_cached_agent(
                        title_identifier_agents,
                        TITLE_FALLBACK_MODEL,
                        TitleAnalysis,
                        TITLE_IDENTIFIER_PROMPT,
                    ).run_sync(text).output,
                )
            )
        return run_with_transient_retry_and_fallback(self.pacer, "title", candidates, prompt, WINDOW_SECONDS)

    def identify(self, document: str) -> TitleAnalysis:
        """Send the document to the LLM to identify all heading changes."""
        print("Identifying heading changes...")
        analysis = self.run_identifier_request(f"Document:\n\n{document}")
        print(f"Found {len(analysis.changes)} heading change(s).")
        return analysis

    def apply(self, document: str, analysis: TitleAnalysis) -> str:
        """Apply heading changes programmatically without LLM."""
        for change in analysis.changes:
            if change.action == HeadingAction.REMOVE:
                # Remove the heading line and the blank line after it
                pattern = re.escape(change.original_heading) + r"\n?"
                document = re.sub(pattern, "", document, count=1)
                print(f"[removed] {change.original_heading}")
            elif change.action == HeadingAction.KEEP and change.new_level is not None:
                # Build the new heading
                heading_text = change.new_text
                if heading_text is None:
                    heading_text = re.sub(r"^#+\s*", "", change.original_heading)
                else:
                    heading_text = re.sub(r"^#+\s*", "", heading_text)
                new_heading = "#" * change.new_level + " " + heading_text
                if new_heading != change.original_heading:
                    document = document.replace(change.original_heading, new_heading, 1)
                    print(f"[changed] {change.original_heading} -> {new_heading}")
        return document

    def edit(self, document: str) -> TitleAnalysis:
        """Run the full two-step title editing process. Returns the analysis for caching."""
        analysis = self.identify(document)
        return analysis
