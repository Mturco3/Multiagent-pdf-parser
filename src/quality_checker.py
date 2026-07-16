"""Review the completed notes and apply targeted quality corrections.

The module contains ``QualityChecker``, deterministic cleanup helpers, and
model-backed issue identification and replacement generation.
"""

import re

from pydantic_ai import Agent

from .models import IssueType, QualityIssue, QualityReport
from .utilities.model_config import (
    QUALITY_FIXER_MODEL,
    QUALITY_FIXER_MODEL_RPD,
    QUALITY_FIXER_MODEL_RPM,
    QUALITY_IDENTIFIER_MODEL,
    QUALITY_IDENTIFIER_MODEL_RPD,
    QUALITY_IDENTIFIER_MODEL_RPM,
    WINDOW_SECONDS
)
from .utilities.model_retry import run_agent_request
from .utilities.normalizer import repair_text
from .utilities.prompts import QUALITY_CHECKER_PROMPT, QUALITY_FIXER_PROMPT
from .utilities.rate_limit import DailyQuotaExceededError, RequestPacer

quality_identifier_agents: dict[str, Agent] = {}
quality_fixer_agents: dict[str, Agent] = {}
VALIDATION_ARTIFACT_PATTERNS = (
    r"(?im)^[ \t]*Validation feedback:[ \t]*\n?",
    r"(?im)^[ \t]*Please return text\.[ \t]*\n?",
    r"(?im)^[ \t]*Fix the errors and try again\.[ \t]*\n?",
    r"\bValidation feedback:\b",
    r"\bPlease return text\.\b",
    r"\bFix the errors and try again\.\b"
)


class QualityChecker:
    """Identifies quality issues via LLM, then fixes them with targeted LLM calls."""

    def __init__(self) -> None:
        """Initialize the quality checker with rate-limited model access."""
        self.pacer = RequestPacer(WINDOW_SECONDS)

    def run_identifier_request(self, prompt: str) -> QualityReport:
        """Ask the model to identify issues, retrying only temporary failures."""
        return run_agent_request(
            pacer=self.pacer,
            request_name="quality-identifier",
            model_name=QUALITY_IDENTIFIER_MODEL,
            rpm=QUALITY_IDENTIFIER_MODEL_RPM,
            rpd=QUALITY_IDENTIFIER_MODEL_RPD,
            agent_cache=quality_identifier_agents,
            output_type=QualityReport,
            instructions=QUALITY_CHECKER_PROMPT,
            prompt=prompt,
            window_seconds=WINDOW_SECONDS
        )

    def run_fixer_request(self, prompt: str) -> str:
        """Ask the model for replacement text, retrying temporary failures."""
        return run_agent_request(
            pacer=self.pacer,
            request_name="quality-fixer",
            model_name=QUALITY_FIXER_MODEL,
            rpm=QUALITY_FIXER_MODEL_RPM,
            rpd=QUALITY_FIXER_MODEL_RPD,
            agent_cache=quality_fixer_agents,
            output_type=str,
            instructions=QUALITY_FIXER_PROMPT,
            prompt=prompt,
            window_seconds=WINDOW_SECONDS
        )

    def identify(self, source_document: str, document: str) -> QualityReport:
        """Compare generated notes with their source slides during quality review."""
        print("Identifying quality issues...")
        prompt = (
            f"Source slides:\n\n{source_document}\n\n"
            f"Generated document:\n\n{document}"
        )
        report = self.run_identifier_request(prompt)
        if report.issues:
            for issue in report.issues:
                print(f"[{issue.issue_type.value}] {issue.explanation}")
        else:
            print("No issues found.")
        return report

    def replace_last_non_heading_occurrence(
        self,
        document: str,
        target: str,
        replacement: str
    ) -> tuple[str, bool]:
        """Replace the last target occurrence that is not part of a Markdown heading."""
        end = len(document)
        while True:
            index = document.rfind(target, 0, end)
            if index < 0:
                return document, False

            line_start = document.rfind("\n", 0, index) + 1
            prefix = document[line_start:index].strip()
            if not prefix.startswith("#"):
                updated = (
                    document[:index]
                    + replacement
                    + document[index + len(target):]
                )
                return updated, True

            end = index

    def remove_validation_artifacts(self, text: str) -> str:
        """Remove leaked provider/retry text that can appear when an LLM returns diagnostics."""
        cleaned = text
        for pattern in VALIDATION_ARTIFACT_PATTERNS:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        return cleaned

    def repair_math_delimiters(self, text: str) -> str:
        """Repair common malformed inline math delimiters left by model replacements."""
        cleaned = text
        cleaned = re.sub(
            r"\$\$([A-Za-z](?:_\{?[A-Za-z0-9]+\}?|\^\{?[A-Za-z0-9]+\}?)?)\$\$",
            r"$\1$",
            cleaned
        )
        cleaned = re.sub(r"\$\$([^$\n]{1,20})\$", r"$\1$", cleaned)
        cleaned = re.sub(r"\$([A-Za-z])\$\s*\\times\s*([A-Za-z])\$", r"$\1 \\times \2$", cleaned)
        return cleaned

    def bullet_item_text(self, line: str) -> str:
        """Return the text of a markdown bullet line without its marker."""
        return re.sub(r"^\s*[-*]\s+", "", line).strip()

    def collapse_bullet_group(self, bullet_items: list[str]) -> str:
        """Collapse a slide-fragment bullet group into a single prose paragraph."""
        cleaned_items: list[str] = []
        for item in bullet_items:
            item = self.remove_validation_artifacts(item)
            item = re.sub(r"\s+", " ", item).strip()
            if item:
                cleaned_items.append(item)

        return " ".join(cleaned_items).strip()

    def collapse_fragment_bullets(self, document: str) -> str:
        """Collapse one model-identified fragment list into a prose paragraph."""
        lines = document.splitlines()
        output: list[str] = []
        bullet_buffer: list[str] = []

        def flush_bullets():
            """Append the buffered bullet group as a single paragraph."""
            if not bullet_buffer:
                return
            paragraph = self.collapse_bullet_group(bullet_buffer)
            if paragraph:
                output.append(paragraph)
            bullet_buffer.clear()

        for line in lines:
            if re.match(r"^\s*[-*]\s+\S+", line):
                bullet_buffer.append(self.bullet_item_text(line))
                continue

            flush_bullets()
            output.append(line.rstrip())

        flush_bullets()
        return "\n".join(output)

    def sanitize_document(self, document: str) -> str:
        """Apply deterministic final cleanup for known slide-transcription artifacts."""
        cleaned = repair_text(document).replace("\r\n", "\n").replace("\r", "\n")
        cleaned = self.remove_validation_artifacts(cleaned)
        cleaned = self.repair_math_delimiters(cleaned)
        cleaned = re.sub(r"\n#{6,}\s+", "\n#### ", cleaned)
        cleaned = re.sub(r"^#{1,6}\s*$\n?", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"^(#+\s+)#+\s+", r"\1", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip() + "\n"

    def fix_collapsed_bullets(
        self,
        document: str,
        issue: QualityIssue
    ) -> str:
        """Replace one targeted fragment-bullet group with prose."""
        replacement = self.collapse_fragment_bullets(issue.problematic_text).strip()
        updated_document = document.replace(issue.problematic_text, replacement, 1)
        print(f"[fixed] {issue.issue_type.value} (collapsed bullets)")
        return updated_document

    def fix_repetition(self, document: str, issue: QualityIssue) -> str:
        """Remove the final non-heading occurrence of a repeated fragment."""
        if document.count(issue.problematic_text) < 2:
            print("[skip] Repetition target does not occur twice")
            return document

        updated_document, was_fixed = self.replace_last_non_heading_occurrence(
            document,
            issue.problematic_text,
            ""
        )
        if not was_fixed:
            print(
                "[skip] Could not find non-heading duplicate for "
                f"{issue.issue_type.value}"
            )
            return document

        while "\n\n\n" in updated_document:
            updated_document = updated_document.replace("\n\n\n", "\n\n")
        print(f"[fixed] {issue.issue_type.value} (removed duplicate)")
        return updated_document

    def fix_with_model(
        self,
        source_document: str,
        document: str,
        issue: QualityIssue
    ) -> str:
        """Generate and apply replacement text for one remaining issue."""
        print(f"[fixing] {issue.issue_type.value}...")
        prompt = (
            f"Issue type: {issue.issue_type.value}\n"
            f"Explanation: {issue.explanation}\n\n"
            f"Source slides for reference:\n{source_document}\n\n"
            f"Text to fix:\n{issue.problematic_text}"
        )
        try:
            replacement = self.run_fixer_request(prompt)
        except DailyQuotaExceededError:
            raise
        except Exception as error:
            raise RuntimeError(
                f"Failed to fix {issue.issue_type.value}: {error}"
            ) from error

        print(f"[fixed] {issue.issue_type.value}")
        return document.replace(issue.problematic_text, replacement, 1)

    def fix_issue(
        self,
        source_document: str,
        document: str,
        issue: QualityIssue
    ) -> str:
        """Dispatch one issue to its deterministic or model-backed fixer."""
        if issue.issue_type == IssueType.BULLET_LIST_SHOULD_BE_COLLAPSED:
            return self.fix_collapsed_bullets(document, issue)
        if issue.issue_type == IssueType.REPETITION:
            return self.fix_repetition(document, issue)
        return self.fix_with_model(source_document, document, issue)

    def fix(self, source_document: str, document: str, report: QualityReport) -> str:
        """Fix each identified issue with access to the original included slides."""
        fixable_issues = [
            issue
            for issue in report.issues
            if issue.issue_type != IssueType.CONTENT_LOST
        ]

        if not fixable_issues:
            print("No fixable issues.")
            return document

        for issue in fixable_issues:
            if issue.problematic_text not in document:
                print(
                    "[skip] Could not find problematic text for "
                    f"{issue.issue_type.value}"
                )
                continue

            document = self.fix_issue(source_document, document, issue)

        return self.sanitize_document(document)

    def check(self, source_document: str, document: str) -> QualityReport:
        """Run identification step. Returns the report for caching."""
        return self.identify(source_document, document)
