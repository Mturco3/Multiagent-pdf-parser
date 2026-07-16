"""Review and apply heading changes to the assembled Markdown document.

The module contains ``TitleEditor``, which asks an LLM for indexed heading
changes and applies only changes that still match the original document.
"""

import re

from pydantic_ai import Agent

from .models import HeadingAction, TitleAnalysis
from .utilities.model_config import (
    TITLE_MODEL,
    TITLE_MODEL_RPD,
    TITLE_MODEL_RPM,
    WINDOW_SECONDS
)
from .utilities.model_retry import run_agent_request
from .utilities.prompts import TITLE_IDENTIFIER_PROMPT
from .utilities.rate_limit import RequestPacer

title_identifier_agents: dict[str, Agent] = {}
HEADING_PATTERN = re.compile(r"^#{1,6}\s+\S")


class TitleEditor:
    """Identifies heading changes via LLM, then applies them programmatically."""

    def __init__(self) -> None:
        """Initialize the title editor with rate-limited model access."""
        self.pacer = RequestPacer(WINDOW_SECONDS)

    def identify(self, document: str) -> TitleAnalysis:
        """Send the document to the LLM to identify all heading changes."""
        print("Identifying heading changes...")
        headings = [
            line
            for line in document.splitlines()
            if HEADING_PATTERN.match(line)
        ]
        heading_inventory = "\n".join(
            f"[heading {index}] {heading}"
            for index, heading in enumerate(headings, start=1)
        )
        prompt = f"Heading inventory:\n{heading_inventory}\n\nDocument:\n\n{document}"
        analysis = run_agent_request(
            pacer=self.pacer,
            request_name="title",
            model_name=TITLE_MODEL,
            rpm=TITLE_MODEL_RPM,
            rpd=TITLE_MODEL_RPD,
            agent_cache=title_identifier_agents,
            output_type=TitleAnalysis,
            instructions=TITLE_IDENTIFIER_PROMPT,
            prompt=prompt,
            window_seconds=WINDOW_SECONDS
        )
        print(f"Found {len(analysis.changes)} heading change(s).")
        return analysis

    def apply(self, document: str, analysis: TitleAnalysis) -> str:
        """Apply heading changes by stable heading index without touching body text."""
        changes_by_index = {change.heading_index: change for change in analysis.changes}
        output_lines: list[str] = []
        heading_index = 0

        for line in document.splitlines():
            if not HEADING_PATTERN.match(line):
                output_lines.append(line)
                continue

            heading_index += 1
            change = changes_by_index.get(heading_index)
            if change is None or change.original_heading != line:
                output_lines.append(line)
                continue
            if change.action == HeadingAction.REMOVE:
                print(f"[removed] {line}")
                continue

            original_level = len(line) - len(line.lstrip("#"))
            new_level = change.new_level or original_level
            heading_text = change.new_text or re.sub(r"^#+\s*", "", line)
            heading_text = re.sub(r"^#+\s*", "", heading_text).strip()
            new_heading = "#" * new_level + " " + heading_text
            output_lines.append(new_heading)
            if new_heading != line:
                print(f"[changed] {line} -> {new_heading}")

        updated_document = "\n".join(output_lines)
        updated_document = re.sub(r"\n{3,}", "\n\n", updated_document)
        return updated_document.strip() + "\n"
