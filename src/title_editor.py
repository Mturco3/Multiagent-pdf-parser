import re

from pydantic_ai import Agent

from .models import TitleAnalysis, HeadingAction
from .utilities.model_config import (
    TITLE_MODEL,
    TITLE_MODEL_RPD,
    TITLE_MODEL_RPM,
    WINDOW_SECONDS,
)
from .utilities.model_retry import get_cached_agent, run_with_retry
from .utilities.prompts import TITLE_IDENTIFIER_PROMPT
from .utilities.rate_limit import RequestPacer

title_identifier_agents: dict[str, Agent] = {}


class TitleEditor:
    """Identifies heading changes via LLM, then applies them programmatically."""

    def __init__(self):
        """Initialize the title editor with rate-limited model access."""
        self.pacer = RequestPacer(WINDOW_SECONDS)

    def identify(self, document: str) -> TitleAnalysis:
        """Send the document to the LLM to identify all heading changes."""
        print("Identifying heading changes...")
        runner = lambda text: get_cached_agent(title_identifier_agents, TITLE_MODEL, TitleAnalysis, TITLE_IDENTIFIER_PROMPT).run_sync(text).output
        analysis = run_with_retry(self.pacer, "title", TITLE_MODEL, TITLE_MODEL_RPM, TITLE_MODEL_RPD, runner, f"Document:\n\n{document}", WINDOW_SECONDS)
        print(f"Found {len(analysis.changes)} heading change(s).")
        return analysis

    def apply(self, document: str, analysis: TitleAnalysis) -> str:
        """Apply heading changes programmatically without LLM."""
        for change in analysis.changes:
            if change.action == HeadingAction.REMOVE:
                # Remove the heading line and any trailing blank line
                pattern = re.escape(change.original_heading) + r"\n{1,2}"
                document = re.sub(pattern, "", document, count=1)
                print(f"[removed] {change.original_heading}")
            elif change.action == HeadingAction.KEEP and change.new_level is not None:
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
