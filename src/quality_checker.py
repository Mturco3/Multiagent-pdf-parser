from pydantic_ai import Agent

from .checker import MODEL
from .models import QualityReport
from .utilities.prompts import QUALITY_CHECKER_PROMPT

quality_agent = Agent(MODEL, output_type=QualityReport, instructions=QUALITY_CHECKER_PROMPT, model_settings={"temperature": 0})


class QualityChecker:
    """Reviews the final markdown document and flags quality issues."""

    def check(self, document: str) -> QualityReport:
        """Send the full document to the LLM for quality review."""
        print("Sending document for quality review...")
        result = quality_agent.run_sync(f"Document:\n\n{document}")
        report = result.output
        if report.issues:
            for issue in report.issues:
                print(f"[{issue.issue_type.value}] {issue.explanation}")
        else:
            print("No issues found.")
        return report
