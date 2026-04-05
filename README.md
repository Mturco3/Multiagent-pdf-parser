# University Notes Transcript Agent

This project is an agent-based system that processes PDF files (such as lecture notes or academic papers) and generates clean, Obsidian-ready Markdown notes. It uses multiple specialized agents coordinated by an orchestrator, leveraging Groq for LLM-powered tasks.

## Features
- **Text Extractor Agent:** Extracts text from PDF files.
- **Connectivity & Syntactic Improver Agent:** Improves sentence flow and fixes awkward phrasing.
- **Content Checker Agent:** Ensures all words (except stopwords) from the PDF are present in the Markdown output.
- **Image Extractor Agent:** Extracts images, diagrams, and tables from the PDF.
- **Markdown Writer Agent:** Assembles the extracted and improved content into a Markdown file suitable for Obsidian.
- **Orchestrator:** Coordinates the agents and manages the workflow.

## Workflow
1. **Input:** User provides a PDF file.
2. **Extraction:** Text and images/diagrams/tables are extracted using dedicated agents and tools.
3. **Improvement:** Text is improved for readability and connectivity.
4. **Content Check:** Ensures completeness of Markdown notes.
5. **Markdown Generation:** All content is assembled into a Markdown file for your Obsidian vault.

## Tools & Technologies
- Python 3.10+
- [Groq](https://groq.com/) for LLM tasks
- PDF parsing libraries (e.g., PyMuPDF, pdfplumber, pdf2image, camelot, etc.)
- Agent-based architecture (e.g., using asyncio, multiprocessing, or frameworks like langchain/crewAI)

## Setup
1. Create and activate a virtual environment:
   ```sh
   python -m venv pdf_parser
   pdf_parser\Scripts\activate  # On Windows
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your API keys:
   ```
   GROQ_API_KEY=your_groq_api_key
   GOOGLE_API_KEY=your_google_api_key
   ```

## Usage

**Option 1 — using `config.yaml`:**

Create a `config.yaml` in the project root:
```yaml
input_pdf: 'path/to/your/slides.pdf'
output_dir: 'path/to/output/directory'
```
Then run:
```sh
python main.py
```

**Option 2 — passing paths directly:**
```sh
python main.py path/to/slides.pdf path/to/output.md
```

The output Markdown file will be generated and ready for import into Obsidian.

## Customization
- You can extend or modify agents for additional processing (e.g., citation extraction, advanced diagram parsing, etc.).
