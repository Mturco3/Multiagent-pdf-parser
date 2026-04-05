import fitz  # PyMuPDF
import os


class TextExtractorAgent:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def extract(self):
        pdf_name = os.path.basename(self.pdf_path)
        print(f"\n[TextExtractor] Extracting text from '{pdf_name}'...")
        doc = fitz.open(self.pdf_path)
        pages = {}
        for i, page in enumerate(doc):
            pages[i + 1] = page.get_text()
        doc.close()
        print(f"[TextExtractor] [OK] Extracted {len(pages)} pages")
        return pages
