import re
import time

import httpx
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError

from .models import QualityReport, IssueType
from .utilities.model_config import (
    QUALITY_FIXER_MODEL,
    QUALITY_FIXER_MODEL_RPD,
    QUALITY_FIXER_MODEL_RPM,
    QUALITY_IDENTIFIER_MODEL,
    QUALITY_IDENTIFIER_MODEL_RPD,
    QUALITY_IDENTIFIER_MODEL_RPM,
    WINDOW_SECONDS,
)
from .utilities.prompts import QUALITY_CHECKER_PROMPT, QUALITY_FIXER_PROMPT
from .utilities.rate_limit import DailyQuotaExceededError, RequestPacer

MAX_MODEL_RETRIES = 3
TIMEOUT_RETRY_DELAY_SECONDS = 30.0
TRANSIENT_STATUS_CODES = {429, 500, 503}

# Identifies quality issues in the document
quality_identifier_agent = Agent(
    QUALITY_IDENTIFIER_MODEL,
    output_type=QualityReport,
    instructions=QUALITY_CHECKER_PROMPT,
    model_settings={"temperature": 0}
)

# Fixes a single text fragment based on an identified issue
quality_fixer_agent = Agent(
    QUALITY_FIXER_MODEL,
    output_type=str,
    instructions=QUALITY_FIXER_PROMPT,
    model_settings={"temperature": 0}
)


class QualityChecker:
    """Identifies quality issues via LLM, then fixes them with targeted LLM calls."""

    def __init__(self):
        """Initialize the quality checker with rate-limited model access."""
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

    def run_with_transient_retry(self, request_name: str, model_name: str, rpm: int, rpd: int, runner, prompt: str):
        """Retry transient provider failures with backoff while keeping prompts deterministic."""
        attempt = 0
        while True:
            self.pacer.acquire_request_slot(model_name, rpm, rpd, request_name)
            try:
                result = runner(prompt)
                return result
            except ModelHTTPError as error:
                if error.status_code == 429 and self.pacer.is_daily_quota_error(error.body):
                    self.pacer.mark_daily_exhausted(model_name, rpd)
                    raise DailyQuotaExceededError(
                        f"Provider daily request quota exhausted for {model_name}. "
                        "Resume after reset or switch this stage to another model."
                    ) from error
                if error.status_code not in TRANSIENT_STATUS_CODES or attempt >= MAX_MODEL_RETRIES - 1:
                    raise

                delay = self.get_retry_delay(error)
                attempt += 1
                print(f"{request_name} transient error {error.status_code} - retrying in {delay:.1f}s ({attempt}/{MAX_MODEL_RETRIES})...")
                time.sleep(delay)
            except httpx.TimeoutException as error:
                if attempt >= MAX_MODEL_RETRIES - 1:
                    raise

                attempt += 1
                print(
                    f"{request_name} transient timeout - retrying in "
                    f"{TIMEOUT_RETRY_DELAY_SECONDS:.1f}s ({attempt}/{MAX_MODEL_RETRIES})..."
                )
                time.sleep(TIMEOUT_RETRY_DELAY_SECONDS)

    def identify(self, document: str) -> QualityReport:
        """Send the document to the LLM for quality review."""
        print("Identifying quality issues...")
        result = self.run_with_transient_retry(
            "quality-identifier",
            QUALITY_IDENTIFIER_MODEL,
            QUALITY_IDENTIFIER_MODEL_RPM,
            QUALITY_IDENTIFIER_MODEL_RPD,
            quality_identifier_agent.run_sync,
            f"Document:\n\n{document}",
        )
        report = result.output
        if report.issues:
            for issue in report.issues:
                print(f"[{issue.issue_type.value}] {issue.explanation}")
        else:
            print("No issues found.")
        return report

    def replace_first_non_heading_occurrence(self, document: str, target: str, replacement: str) -> tuple[str, bool]:
        """Replace the first target occurrence that is not part of a markdown heading."""
        start = 0
        while True:
            index = document.find(target, start)
            if index < 0:
                return document, False

            line_start = document.rfind("\n", 0, index) + 1
            prefix = document[line_start:index].strip()
            if not prefix.startswith("#"):
                updated = document[:index] + replacement + document[index + len(target):]
                return updated, True

            start = index + len(target)

    def remove_raw_slide_metadata(self, text: str) -> str:
        """Remove repeated slide footer/source artifacts that are not lecture-note content."""
        metadata_patterns = [
            r"\bData Visualisation\s*/\s*Blerina Sinaimeri\b",
            r"\bAdvanced Visualisations\s*/\s*Blerina Sinaimeri\b",
            r"\bData Visualisation\b",
            r"\bAdvanced Visualisations\b",
            r"\bBlerina Sinaimeri\b",
            r"\bNetworkX\b",
            r"[- ]*L\.\s*Barab\S*\s*,\s*Network Science:\s*Communities\.?",
        ]

        cleaned = text
        for pattern in metadata_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        return cleaned

    def remove_validation_artifacts(self, text: str) -> str:
        """Remove leaked provider/retry text that can appear when an LLM returns diagnostics."""
        validation_patterns = [
            r"(?im)^[ \t]*Validation feedback:[ \t]*\n?",
            r"(?im)^[ \t]*Please return text\.[ \t]*\n?",
            r"(?im)^[ \t]*Fix the errors and try again\.[ \t]*\n?",
            r"\bValidation feedback:\b",
            r"\bPlease return text\.\b",
            r"\bFix the errors and try again\.\b",
        ]

        cleaned = text
        for pattern in validation_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        return cleaned

    def normalize_question_form(self, text: str) -> str:
        """Convert common rhetorical slide questions into declarative note fragments."""
        replacements = {
            "Can these extreme cases be reached?": "These extreme cases are rarely reached in practice.",
            "Whether these extreme cases can be reached.": "These extreme cases are rarely reached in practice.",
        }
        cleaned = text
        for question, statement in replacements.items():
            cleaned = cleaned.replace(question, statement)

        cleaned = re.sub(r"^(#+\s*)Why\s+(.+?)\?\s*$", r"\1Reasons for \2", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"^(#+\s*)What is\s+(.+?)\?\s*$", r"\1Definition of \2", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"^(#+\s*)How\s+(.+?)\?\s*$", r"\1How \2", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(
            r"\bCan ([^.?\n]+?) be ([^.?\n]+?)\?",
            lambda match: f"Whether {match.group(1)} can be {match.group(2)}.",
            cleaned,
        )
        cleaned = re.sub(r"\?(\s|$)", r".\1", cleaned)
        return cleaned

    def bullet_item_text(self, line: str) -> str:
        """Return the text of a markdown bullet line without its marker."""
        return re.sub(r"^\s*[-*]\s+", "", line).strip()

    def looks_like_true_enumeration(self, bullet_items: list[str], previous_heading: str | None) -> bool:
        """Heuristically preserve lists that are genuine enumerations rather than slide fragments."""
        if not bullet_items:
            return False

        heading = (previous_heading or "").lower()
        named_items = sum(1 for item in bullet_items if re.match(r"^[A-Z][A-Za-z0-9 /-]{1,45}:\s+\S+", item))
        explicitly_structural_heading = any(cue in heading for cue in ("types", "categories", "options", "criteria"))

        return explicitly_structural_heading and named_items >= 2

    def collapse_bullet_group(self, bullet_items: list[str]) -> str:
        """Collapse a slide-fragment bullet group into a single prose paragraph."""
        cleaned_items: list[str] = []
        for item in bullet_items:
            item = self.remove_raw_slide_metadata(item)
            item = self.remove_validation_artifacts(item)
            item = re.sub(r"\s+", " ", item).strip()
            if item:
                cleaned_items.append(item)

        return " ".join(cleaned_items).strip()

    def collapse_fragment_bullets(self, document: str, force: bool = False) -> str:
        """Collapse residual markdown bullet groups that are slide fragments."""
        lines = document.splitlines()
        output: list[str] = []
        bullet_buffer: list[str] = []
        previous_heading: str | None = None

        def flush_bullets():
            if not bullet_buffer:
                return

            if not force and self.looks_like_true_enumeration(bullet_buffer, previous_heading):
                output.extend(f"- {item}" for item in bullet_buffer)
            else:
                paragraph = self.collapse_bullet_group(bullet_buffer)
                if paragraph:
                    output.append(paragraph)
            bullet_buffer.clear()

        for line in lines:
            stripped = line.strip()
            if re.match(r"^\s*[-*]\s+\S+", line):
                bullet_buffer.append(self.bullet_item_text(line))
                continue

            flush_bullets()
            output.append(line.rstrip())
            if stripped.startswith("#"):
                previous_heading = stripped

        flush_bullets()
        return "\n".join(output)

    def sanitize_document(self, document: str) -> str:
        """Apply deterministic final cleanup for known slide-transcription artifacts."""
        cleaned = document.replace("\r\n", "\n").replace("\r", "\n")
        cleaned = self.remove_validation_artifacts(cleaned)
        cleaned = self.remove_raw_slide_metadata(cleaned)
        cleaned = self.normalize_question_form(cleaned)
        cleaned = self.collapse_fragment_bullets(cleaned)
        cleaned = re.sub(r"\n#{6,}\s+", "\n#### ", cleaned)
        cleaned = re.sub(r"^#{1,6}\s*$\n?", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"^(#+\s+)#+\s+", r"\1", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip() + "\n"

    def fix(self, document: str, report: QualityReport) -> str:
        """Fix each identified issue by sending the problematic fragment to the LLM."""
        # Filter out content_lost issues since they cannot be fixed without originals
        fixable_issues = [i for i in report.issues if i.issue_type != IssueType.CONTENT_LOST]

        if not fixable_issues:
            print("No fixable issues.")
            return document

        for issue in fixable_issues:
            # Check that the problematic text exists in the document
            if issue.problematic_text not in document:
                print(f"[skip] Could not find problematic text for {issue.issue_type.value}")
                continue

            if issue.issue_type == IssueType.RAW_SLIDE_METADATA:
                replacement = self.remove_raw_slide_metadata(issue.problematic_text).strip()
                document = document.replace(issue.problematic_text, replacement, 1)
                print(f"[fixed] {issue.issue_type.value} (removed metadata)")
                continue

            if issue.issue_type == IssueType.QUESTION_FORM:
                replacement = self.normalize_question_form(issue.problematic_text).strip()
                document = document.replace(issue.problematic_text, replacement, 1)
                print(f"[fixed] {issue.issue_type.value} (declarative form)")
                continue

            if issue.issue_type == IssueType.BULLET_LIST_SHOULD_BE_COLLAPSED:
                replacement = self.collapse_fragment_bullets(issue.problematic_text, force=True).strip()
                document = document.replace(issue.problematic_text, replacement, 1)
                print(f"[fixed] {issue.issue_type.value} (collapsed bullets)")
                continue

            # Handle repetition programmatically by removing the duplicate
            if issue.issue_type == IssueType.REPETITION:
                document, fixed = self.replace_first_non_heading_occurrence(document, issue.problematic_text, "")
                if not fixed:
                    print(f"[skip] Could not find non-heading duplicate for {issue.issue_type.value}")
                    continue
                # Clean up leftover blank lines from removal
                while "\n\n\n" in document:
                    document = document.replace("\n\n\n", "\n\n")
                print(f"[fixed] {issue.issue_type.value} (removed duplicate)")
                continue

            print(f"[fixing] {issue.issue_type.value}...")
            prompt = f"Issue type: {issue.issue_type.value}\nExplanation: {issue.explanation}\n\nText to fix:\n{issue.problematic_text}"
            try:
                result = self.run_with_transient_retry(
                    "quality-fixer",
                    QUALITY_FIXER_MODEL,
                    QUALITY_FIXER_MODEL_RPM,
                    QUALITY_FIXER_MODEL_RPD,
                    quality_fixer_agent.run_sync,
                    prompt,
                )
                document = document.replace(issue.problematic_text, result.output, 1)
                print(f"[fixed] {issue.issue_type.value}")
            except Exception as error:
                print(f"[error] Failed to fix {issue.issue_type.value}: {error}")

        return self.sanitize_document(document)

    def check(self, document: str) -> QualityReport:
        """Run identification step. Returns the report for caching."""
        return self.identify(document)
