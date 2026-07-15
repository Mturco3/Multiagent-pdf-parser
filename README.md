# University Notes Transcript Agent

Converts PDF lecture slides into source-aware Markdown notes. The pipeline preserves PDF layout long enough to remove repeated deck chrome, infer titles, retain genuine lists, and minimize unnecessary model requests.

## Pipeline

```text
PDF
  -> structured text, font, position, and image extraction
  -> deck-wide boilerplate and page-number removal
  -> slide classification and validated edit actions
  -> selective rewriting
  -> math-candidate detection and LaTeX conversion
  -> Markdown assembly
  -> indexed heading analysis
  -> source-aware quality review
  -> final Markdown
```

The transcriber stores both readable `.txt` files and structured `.json` artifacts. Cache directories include the PDF content hash, so unrelated PDFs with the same filename cannot collide. Prompt, schema, model, source, and pipeline-version changes invalidate dependent artifacts automatically.

## Setup

Create and activate a virtual environment, then install dependencies:

```sh
python -m venv pdf_parser
pdf_parser\Scripts\activate
pip install -r requirements.txt
```

For development and tests:

```sh
pip install -r requirements-dev.txt
python -m pytest
```

Create a `.env` file containing the Google API key:

```env
GOOGLE_API_KEY=your_google_api_key
```

## Usage

Run against an explicit PDF:

```sh
python main.py "path/to/slides.pdf"
```

Or configure `input_pdf` in the local `config.yaml` and run:

```sh
python main.py
```

Force a new content-addressed run:

```sh
python main.py --clear-cache "path/to/slides.pdf"
```

Launch the isolated-upload Streamlit interface:

```sh
python -m streamlit run app.py
```

## Models and quotas

When `GROQ_API_KEY` is configured, direct per-slide note extraction defaults to the fast `groq:openai/gpt-oss-20b` model. Longer document-wide stages continue to use `google:gemma-4-26b-a4b-it`. Without a Groq key, note extraction also uses the Google default. Repeated HTTP 503 responses switch the affected request to `google:gemma-4-31b-it`. Override stages or the fallback in `.env`:

```env
GROQ_API_KEY=your-groq-key
REWRITER_MODEL=groq:openai/gpt-oss-20b
MATH_MODEL=google:gemma-4-26b-a4b-it
TITLE_MODEL=google:gemma-4-26b-a4b-it
QUALITY_IDENTIFIER_MODEL=google:gemma-4-26b-a4b-it
QUALITY_FIXER_MODEL=google:gemma-4-26b-a4b-it
FALLBACK_MODEL=google:gemma-4-31b-it
MODEL_REQUEST_TIMEOUT_SECONDS=60
```

The rewriter is the direct per-slide note extractor; there is no separate per-slide checker request. RPM and RPD values can be overridden with the corresponding `*_RPM` and `*_RPD` variables. Set `MODEL_INPUT_TPM` to the active project input-token-per-minute quota when local TPM pacing is desired. Provider limits can change, so copy stricter limits from the active provider dashboard when necessary.

## Output and privacy

CLI output is stored under `cache/<pdf-name>-<content-hash>/`. The Streamlit interface uses an isolated temporary cache and deletes uploaded PDF files after processing. PDF content is sent to the configured Google model during LLM stages.
