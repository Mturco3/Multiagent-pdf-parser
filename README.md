# University Notes Transcript Agent

Converts PDF lecture slides into polished, readable university notes as a single Markdown document. Uses a multi-step LLM pipeline powered by Pydantic AI and Google models.

## Pipeline

```
PDF file
  |
  v
+------------------+     cache/<name>/transcriptions/slide_NNN.txt
|   Transcriber    |---> (one .txt per page, raw text extraction via PyMuPDF)
+------------------+
  |
  v
+------------------+     cache/<name>/reviews/slide_NNN_review.json
|   Checker        |---> (LLM classifies slide type, extracts title, suggests actions)
|   + Reviewer     |     (reviewer approves/rejects plan; up to 3 attempts)
+------------------+     skips: course_info, image_description, outline slides
  |
  v
+------------------+     cache/<name>/rewrites/slide_NNN.json
|   Rewriter       |---> (LLM rewrites text applying approved actions)
|   + Reviewer     |     (receives previous paragraph for flow continuity)
+------------------+     unapproved slides fall back to deterministic passthrough
  |
  v
+------------------+     cache/<name>/math/slide_NNN.json
|  Math Formatter  |---> (LLM finds math expressions, converts to LaTeX)
+------------------+
  |
  v
+------------------+
|    Assemble      |---> concatenate slides into one markdown document
+------------------+     (deduplicate titles, handle continuations)
  |
  v
+------------------+     cache/<name>/title_analysis.json
|  Title Editor    |---> (LLM analyzes headings, assigns hierarchy ##/###/####)
+------------------+
  |
  v
+------------------+     cache/<name>/quality_report.json
| Quality Checker  |---> (LLM spots issues, applies targeted fixes)
+------------------+
  |
  v
cache/<name>/<name>.md   (final output)
```

Each step's output is cached with source-hash fingerprinting so the pipeline can resume from where it left off. Changing a prompt or model schema automatically invalidates stale cache entries. Reviews, rewrites, and math artifacts are checkpointed per slide.

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
    model_config.py      Stage-specific model, RPM, and RPD configuration
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

All stages default to `google:gemma-4-26b` (15 RPM, unlimited tokens, 1500 RPD on the free tier). After 3 consecutive HTTP 503 responses, the pipeline automatically falls back to `google:gemma-4-31b`. Override any stage via `.env`:

```env
CHECKER_MODEL=google:gemma-4-26b
REWRITER_MODEL=google:gemma-4-31b
TITLE_MODEL=google:gemini-2.5-flash
```

The pipeline tracks a local per-model daily request budget in `cache/_model_usage.json`, writes a readable request log to `request_logs/model_requests.csv`, prints both `rpm` and `rpd` at startup, and stops cleanly before a configured budget is exceeded. Known model families and their default limits are defined in `src/utilities/model_config.py`.

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

For Windows paths with spaces, quote the PDF path:

```sh
python main.py "C:\Users\miche\University\Material\Data Science\Data Visualization\Slides\13_Dimensionality_Reduction.pdf"
```

To force a fresh run for that PDF and ignore existing cache:

```sh
python main.py --clear-cache "C:\Users\miche\University\Material\Data Science\Data Visualization\Slides\13_Dimensionality_Reduction.pdf"
```

Output is saved to `cache/<pdf_name>/<pdf_name>.md`.

## Tools and Technologies

- Python 3.13
- [Pydantic AI](https://github.com/pydantic/pydantic-ai) for LLM agents with structured output
- Google AI Studio models via Pydantic AI
- [PyMuPDF](https://pymupdf.readthedocs.io/) for PDF text extraction
