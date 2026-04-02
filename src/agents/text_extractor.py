import fitz  # PyMuPDF

class TextExtractorAgent:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def extract(self):
        doc = fitz.open(self.pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
