from pydantic_ai import Agent

from .checker import MODEL
from .models import MathIdentifierResponse
from .utilities.prompts import MATH_IDENTIFIER_PROMPT, LATEX_WRITER_PROMPT

# Identifies math expressions in the document
math_identifier_agent = Agent(MODEL, output_type=MathIdentifierResponse, instructions=MATH_IDENTIFIER_PROMPT, model_settings={"temperature": 0})

# Rewrites the document with LaTeX notation applied
latex_writer_agent = Agent(MODEL, output_type=str, instructions=LATEX_WRITER_PROMPT, model_settings={"temperature": 0})


class MathFormatter:
    """Detects math expressions and converts them to LaTeX notation."""

    def format_document(self, document: str) -> str:
        """Identify all math in the document, then apply LaTeX formatting."""
        # Step 1: identify math expressions
        print("Identifying math expressions...")
        id_result = math_identifier_agent.run_sync(f"Document text:\n\n{document}")
        expressions = id_result.output.expressions

        if not expressions:
            print("No math expressions found.")
            return document

        print(f"Found {len(expressions)} expression(s).")

        # Step 2: apply LaTeX formatting
        print("Applying LaTeX formatting...")
        expressions_text = "\n".join(f"- \"{expr.original_text}\" (display={expr.is_display})" for expr in expressions)
        prompt = f"Full text:\n\n{document}\n\nIdentified expressions:\n{expressions_text}"
        latex_result = latex_writer_agent.run_sync(prompt)
        print("LaTeX formatting done.")
        return latex_result.output
