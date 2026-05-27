# University Notes Transcript Agent

Converts PDF lecture slides into polished, readable university notes as a single Markdown document. Uses a multi-step LLM pipeline powered by Pydantic AI and Google Gemini.

## Pipeline

1. **Transcriber** — extracts text from each PDF page using PyMuPDF.
2. **Checker** — classifies each slide (content, course_info, image_description, introduction) and suggests editing actions (insert connectivity, flatten bullets, define acronyms, etc.).
3. **Rewriter** — rewrites each slide applying the checker's actions, preserving all original content. Passes previous paragraph for flow continuity.
4. **Math Formatter** — identifies mathematical expressions and converts them to LaTeX (`$inline$` and `$$display$$`).
5. **Title Editor** — assigns heading hierarchy (`##`, `###`, `####`) and removes redundant titles.
6. **Quality Checker** — reviews the final document and flags issues (collapsed lists, dangling references, content loss, repetition).

Each step's output is cached so the pipeline can resume from where it left off if a step fails.

## Project Structure

```
main.py                  Entry point
config.yaml              Input PDF path
src/
  pipeline.py            Orchestrates all steps with rate limiting and caching
  transcriber.py         PDF to per-slide text files
  checker.py             LLM slide classifier and action suggester
  rewriter.py            LLM rewriter and document concatenator
  math_formatter.py      Math detection and LaTeX conversion
  title_editor.py        Heading hierarchy editor
  quality_checker.py     Final document quality reviewer
  models.py              Pydantic models for structured LLM output
  normalizer.py          Text normalization utilities
  utilities/
    prompts.py           All LLM system prompts
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
   ```
   GOOGLE_API_KEY=your_google_api_key
   ```

## Usage

Set the input PDF path in `config.yaml`:
```yaml
input_pdf: 'path/to/your/slides.pdf'
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
- Google Gemini (`gemini-3.1-flash-lite`) via the free tier
- [PyMuPDF](https://pymupdf.readthedocs.io/) for PDF text extraction
