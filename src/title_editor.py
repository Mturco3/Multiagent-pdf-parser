import re
import time

from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError

from .models import TitleAnalysis, HeadingAction
from .utilities.model_config import TITLE_MODEL, TITLE_MODEL_RPM, WINDOW_SECONDS
from .utilities.prompts import TITLE_IDENTIFIER_PROMPT
from .utilities.rate_limit import RequestPacer

# Identifies heading changes needed in the document
title_identifier_agent = Agent(
    TITLE_MODEL,
    output_type=TitleAnalysis,
    instructions=TITLE_IDENTIFIER_PROMPT,
    model_settings={"temperature": 0}
)
MAX_MODEL_RETRIES = 3
TRANSIENT_STATUS_CODES = {429, 503}


class TitleEditor:
    """Identifies heading changes via LLM, then applies them programmatically."""

    def __init__(self):
        self.pacer = RequestPacer(WINDOW_SECONDS)

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

    def run_identifier_request(self, prompt: str) -> TitleAnalysis:
        """Call the title model with pacing and transient retry handling."""
        attempt = 0
        while True:
            self.pacer.wait_for_capacity(TITLE_MODEL, TITLE_MODEL_RPM)
            try:
                result = title_identifier_agent.run_sync(prompt)
                self.pacer.mark_request(TITLE_MODEL)
                return result.output
            except ModelHTTPError as error:
                if error.status_code not in TRANSIENT_STATUS_CODES or attempt >= MAX_MODEL_RETRIES - 1:
                    raise

                delay = self.get_retry_delay(error)
                attempt += 1
                print(f"title transient error {error.status_code} - retrying in {delay:.1f}s ({attempt}/{MAX_MODEL_RETRIES})...")
                time.sleep(delay)
                self.pacer.reset(TITLE_MODEL, TITLE_MODEL_RPM)

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
