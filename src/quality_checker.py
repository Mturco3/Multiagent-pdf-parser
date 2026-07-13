import re

from pydantic_ai import Agent

from .models import QualityReport, IssueType
from .utilities.normalizer import repair_text
from .utilities.model_config import QUALITY_FIXER_MODEL, QUALITY_FIXER_MODEL_RPD, QUALITY_FIXER_MODEL_RPM, QUALITY_IDENTIFIER_MODEL, QUALITY_IDENTIFIER_MODEL_RPD, QUALITY_IDENTIFIER_MODEL_RPM, WINDOW_SECONDS
from .utilities.model_retry import get_cached_agent, run_with_retry
from .utilities.prompts import QUALITY_CHECKER_PROMPT, QUALITY_FIXER_PROMPT
from .utilities.rate_limit import DailyQuotaExceededError, RequestPacer

quality_identifier_agents: dict[str, Agent] = {}
quality_fixer_agents: dict[str, Agent] = {}


class QualityChecker:
    """Identifies quality issues via LLM, then fixes them with targeted LLM calls."""

    def __init__(self):
        """Initialize the quality checker with rate-limited model access."""
        self.pacer = RequestPacer(WINDOW_SECONDS)

    def run_identifier_request(self, prompt: str) -> QualityReport:
        """Run the quality identifier with transient retry handling."""
        runner = lambda active_model, text: get_cached_agent(quality_identifier_agents, active_model, QualityReport, QUALITY_CHECKER_PROMPT).run_sync(text).output
        return run_with_retry(self.pacer, "quality-identifier", QUALITY_IDENTIFIER_MODEL, QUALITY_IDENTIFIER_MODEL_RPM, QUALITY_IDENTIFIER_MODEL_RPD, runner, prompt, WINDOW_SECONDS)

    def run_fixer_request(self, prompt: str) -> str:
        """Run the quality fixer with transient retry handling."""
        runner = lambda active_model, text: get_cached_agent(quality_fixer_agents, active_model, str, QUALITY_FIXER_PROMPT).run_sync(text).output
        return run_with_retry(self.pacer, "quality-fixer", QUALITY_FIXER_MODEL, QUALITY_FIXER_MODEL_RPM, QUALITY_FIXER_MODEL_RPD, runner, prompt, WINDOW_SECONDS)

    def identify(self, source_document: str, document: str) -> QualityReport:
        """Compare generated notes with their source slides during quality review."""
        print("Identifying quality issues...")
        prompt = f"Source slides:\n\n{source_document}\n\nGenerated document:\n\n{document}"
        report = self.run_identifier_request(prompt)
        if report.issues:
            for issue in report.issues:
                print(f"[{issue.issue_type.value}] {issue.explanation}")
        else:
            print("No issues found.")
        return report

    def replace_last_non_heading_occurrence(self, document: str, target: str, replacement: str) -> tuple[str, bool]:
        """Replace the last target occurrence that is not part of a Markdown heading."""
        end = len(document)
        while True:
            index = document.rfind(target, 0, end)
            if index < 0:
                return document, False

            line_start = document.rfind("\n", 0, index) + 1
            prefix = document[line_start:index].strip()
            if not prefix.startswith("#"):
                updated = document[:index] + replacement + document[index + len(target):]
                return updated, True

            end = index

    def remove_validation_artifacts(self, text: str) -> str:
        """Remove leaked provider/retry text that can appear when an LLM returns diagnostics."""
        validation_patterns = [
            r"(?im)^[ \t]*Validation feedback:[ \t]*\n?",
            r"(?im)^[ \t]*Please return text\.[ \t]*\n?",
            r"(?im)^[ \t]*Fix the errors and try again\.[ \t]*\n?",
            r"\bValidation feedback:\b",
            r"\bPlease return text\.\b",
            r"\bFix the errors and try again\.\b"
        ]

        cleaned = text
        for pattern in validation_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        return cleaned

    def repair_math_delimiters(self, text: str) -> str:
        """Repair common malformed inline math delimiters left by model replacements."""
        cleaned = text
        cleaned = re.sub(r"\$\$([A-Za-z](?:_\{?[A-Za-z0-9]+\}?|\^\{?[A-Za-z0-9]+\}?)?)\$\$", r"$\1$", cleaned)
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

    def fix(self, source_document: str, document: str, report: QualityReport) -> str:
        """Fix each identified issue with access to the original included slides."""
        fixable_issues = [i for i in report.issues if i.issue_type != IssueType.CONTENT_LOST]

        if not fixable_issues:
            print("No fixable issues.")
            return document

        for issue in fixable_issues:
            if issue.problematic_text not in document:
                print(f"[skip] Could not find problematic text for {issue.issue_type.value}")
                continue

            if issue.issue_type == IssueType.BULLET_LIST_SHOULD_BE_COLLAPSED:
                replacement = self.collapse_fragment_bullets(issue.problematic_text).strip()
                document = document.replace(issue.problematic_text, replacement, 1)
                print(f"[fixed] {issue.issue_type.value} (collapsed bullets)")
                continue
            elif issue.issue_type == IssueType.REPETITION:
                if document.count(issue.problematic_text) < 2:
                    print(f"[skip] Repetition target does not occur twice")
                    continue
                document, fixed = self.replace_last_non_heading_occurrence(document, issue.problematic_text, "")
                if not fixed:
                    print(f"[skip] Could not find non-heading duplicate for {issue.issue_type.value}")
                    continue
                while "\n\n\n" in document:
                    document = document.replace("\n\n\n", "\n\n")
                print(f"[fixed] {issue.issue_type.value} (removed duplicate)")
                continue

            print(f"[fixing] {issue.issue_type.value}...")
            prompt = f"Issue type: {issue.issue_type.value}\nExplanation: {issue.explanation}\n\nSource slides for reference:\n{source_document}\n\nText to fix:\n{issue.problematic_text}"
            try:
                replacement = self.run_fixer_request(prompt)
                document = document.replace(issue.problematic_text, replacement, 1)
                print(f"[fixed] {issue.issue_type.value}")
            except DailyQuotaExceededError:
                raise
            except Exception as error:
                raise RuntimeError(f"Failed to fix {issue.issue_type.value}: {error}") from error

        return self.sanitize_document(document)

    def check(self, source_document: str, document: str) -> QualityReport:
        """Run identification step. Returns the report for caching."""
        return self.identify(source_document, document)
