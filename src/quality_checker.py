import time

from pydantic_ai import Agent

from .checker import FULL_TEXT_MODEL, FULL_TEXT_RPM, WINDOW_SECONDS
from .models import QualityReport, IssueType
from .utilities.prompts import QUALITY_CHECKER_PROMPT, QUALITY_FIXER_PROMPT

# Identifies quality issues in the document
quality_identifier_agent = Agent(
    FULL_TEXT_MODEL,
    output_type=QualityReport,
    instructions=QUALITY_CHECKER_PROMPT,
    model_settings={"temperature": 0}
)

# Fixes a single text fragment based on an identified issue
quality_fixer_agent = Agent(
    FULL_TEXT_MODEL,
    output_type=str,
    instructions=QUALITY_FIXER_PROMPT,
    model_settings={"temperature": 0}
)


class QualityChecker:
    """Identifies quality issues via LLM, then fixes them with targeted LLM calls."""

    def identify(self, document: str) -> QualityReport:
        """Send the document to the LLM for quality review."""
        print("Identifying quality issues...")
        result = quality_identifier_agent.run_sync(f"Document:\n\n{document}")
        report = result.output
        if report.issues:
            for issue in report.issues:
                print(f"[{issue.issue_type.value}] {issue.explanation}")
        else:
            print("No issues found.")
        return report

    def fix(self, document: str, report: QualityReport) -> str:
        """Fix each identified issue by sending the problematic fragment to the LLM."""
        # Filter out content_lost issues since they cannot be fixed without originals
        fixable_issues = [i for i in report.issues if i.issue_type != IssueType.CONTENT_LOST]

        if not fixable_issues:
            print("No fixable issues.")
            return document

        request_count = 0
        batch_start = time.time()

        for issue in fixable_issues:
            # Check that the problematic text exists in the document
            if issue.problematic_text not in document:
                print(f"[skip] Could not find problematic text for {issue.issue_type.value}")
                continue

            # Handle repetition programmatically by removing the duplicate
            if issue.issue_type == IssueType.REPETITION:
                document = document.replace(issue.problematic_text, "", 1)
                # Clean up leftover blank lines from removal
                while "\n\n\n" in document:
                    document = document.replace("\n\n\n", "\n\n")
                print(f"[fixed] {issue.issue_type.value} (removed duplicate)")
                continue

            # Rate limiting for full-text model
            if request_count >= FULL_TEXT_RPM - 1 and time.time() - batch_start < WINDOW_SECONDS:
                remaining = WINDOW_SECONDS - (time.time() - batch_start)
                print(f"Rate limit reached - waiting {remaining:.1f}s...")
                time.sleep(remaining)
                request_count = 0
                batch_start = time.time()

            print(f"[fixing] {issue.issue_type.value}...")
            prompt = f"Issue type: {issue.issue_type.value}\nExplanation: {issue.explanation}\n\nText to fix:\n{issue.problematic_text}"
            try:
                result = quality_fixer_agent.run_sync(prompt)
                request_count += 1
                document = document.replace(issue.problematic_text, result.output, 1)
                print(f"[fixed] {issue.issue_type.value}")
            except Exception as error:
                print(f"[error] Failed to fix {issue.issue_type.value}: {error}")

        return document

    def check(self, document: str) -> QualityReport:
        """Run identification step. Returns the report for caching."""
        return self.identify(document)
