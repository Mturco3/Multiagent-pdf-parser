import json
import re

from pydantic_ai import Agent

from .checker import FULL_TEXT_MODEL
from .models import TitleAnalysis, HeadingAction
from .utilities.prompts import TITLE_IDENTIFIER_PROMPT

# Identifies heading changes needed in the document
title_identifier_agent = Agent(
    FULL_TEXT_MODEL,
    output_type=TitleAnalysis,
    instructions=TITLE_IDENTIFIER_PROMPT,
    model_settings={"temperature": 0}
)


class TitleEditor:
    """Identifies heading changes via LLM, then applies them programmatically."""

    def identify(self, document: str) -> TitleAnalysis:
        """Send the document to the LLM to identify all heading changes."""
        print("Identifying heading changes...")
        result = title_identifier_agent.run_sync(f"Document:\n\n{document}")
        analysis = result.output
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
                    # Extract text from original heading (strip # symbols)
                    heading_text = re.sub(r"^#+\s*", "", change.original_heading)
                new_heading = "#" * change.new_level + " " + heading_text
                if new_heading != change.original_heading:
                    document = document.replace(change.original_heading, new_heading, 1)
                    print(f"[changed] {change.original_heading} -> {new_heading}")
        return document

    def edit(self, document: str) -> TitleAnalysis:
        """Run the full two-step title editing process. Returns the analysis for caching."""
        analysis = self.identify(document)
        return analysis
