# University Notes Transcript Agent

Converts PDF lecture slides into polished, readable university notes as a single Markdown document. Uses a multi-step LLM pipeline powered by Pydantic AI and Google models.

## Pipeline

1. **Transcriber** - extracts text from each PDF page using PyMuPDF.
2. **Checker** - classifies each slide and identifies a structured action plan.
3. **Reviewer** - approves or rejects the proposed action plan before any slide rewrite is allowed.
4. **Rewriter** - rewrites each slide only from the approved actions, preserving all original content. Passes previous paragraph for flow continuity.
5. **Math Formatter** - identifies mathematical expressions and converts them to LaTeX (`$inline$` and `$$display$$`).
6. **Title Editor** - assigns heading hierarchy (`##`, `###`, `####`) and removes redundant titles.
7. **Quality Checker** - reviews the final document and flags issues (collapsed lists, dangling references, content loss, repetition).

Each step's output is cached so the pipeline can resume from where it left off if a step fails. Reviews and math artifacts are also checkpointed per slide.

## Project Structure

```text
main.py                  Entry point
config.yaml              Input PDF path
src/
  pipeline.py            Orchestrates all steps with rate limiting and caching
  transcriber.py         PDF to per-slide text files
  checker.py             LLM slide classifier plus reviewer approval loop
  rewriter.py            LLM rewriter and document concatenator
  math_formatter.py      Math detection and LaTeX conversion
  title_editor.py        Heading hierarchy editor
  quality_checker.py     Final document quality reviewer
  models.py              Pydantic models for structured LLM output
  utilities/
    model_config.py      Stage-specific model and RPM configuration
    normalizer.py        Shared text normalization helpers
    prompts.py           All LLM system prompts
    rate_limit.py        Shared per-model pacing helper
cache/                   Intermediate and final outputs per PDF
```

## Setup

1. Create and activate a virtual environment:
   ```sh
   python -m venv pdf_parser
   pdf_parser\Scripts\activate  # Windows
   source pdf_parser/bin/activate  # Linux/Mac
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your API key:
   ```env
   GOOGLE_API_KEY=your_google_api_key
   ```

Optional stage-specific model overrides can also go in `.env`:

```env
FULL_TEXT_MODEL=google:gemini-2.5-flash
CHECKER_MODEL=google:gemini-3.1-flash-lite
REVIEWER_MODEL=google:gemini-3.1-flash-lite
REWRITER_MODEL=google:gemini-3.1-flash-lite
MATH_MODEL=google:gemini-3.1-flash-lite
TITLE_MODEL=google:gemini-2.5-flash
QUALITY_IDENTIFIER_MODEL=google:gemini-2.5-flash
QUALITY_FIXER_MODEL=google:gemini-2.5-flash
```

If you want to try Gemma for cheaper structured stages, set the relevant stage variables to a Gemma model ID that is available in your Google account. The pipeline prints the active stage-to-model mapping at startup so you can confirm exactly what will be used.

## Usage

Set the input PDF path in `config.yaml`:

```yaml
input_pdf: "path/to/your/slides.pdf"
```

Then run:

```sh
python main.py
```

Or pass the path directly:

```sh
python main.py path/to/slides.pdf
```

Output is saved to `cache/<pdf_name>/<pdf_name>.md`.

## Tools and Technologies

- Python 3.13
- [Pydantic AI](https://github.com/pydantic/pydantic-ai) for LLM agents with structured output
- Google AI Studio models via Pydantic AI
- [PyMuPDF](https://pymupdf.readthedocs.io/) for PDF text extraction
