from pydantic_ai import Agent

from .checker import MODEL
from .utilities.prompts import TITLE_HIERARCHY_PROMPT

# Receives the full document and returns it with corrected heading hierarchy
title_agent = Agent(MODEL, output_type=str, instructions=TITLE_HIERARCHY_PROMPT, model_settings={"temperature": 0})


class TitleEditor:
    """Adjusts heading hierarchy and removes redundant titles."""

    def edit(self, document: str) -> str:
        """Send the full document to the LLM for title restructuring."""
        print("Restructuring title hierarchy...")
        result = title_agent.run_sync(f"Document:\n\n{document}")
        print("Title hierarchy done.")
        return result.output
