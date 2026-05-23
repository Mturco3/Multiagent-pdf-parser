import os
import fitz  # PyMuPDF

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache")


class Transcriber:
    """Transcriber is responsible for extracting text from PDF slides and saving them as text files in a cache directory.
    Each slide's text is saved in a separate file named 'slide_XXX.txt' where XXX is the slide number with leading zeros.
    """
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def run(self) -> str:
        pdf_name = os.path.splitext(os.path.basename(self.pdf_path))[0]
        out_dir = os.path.join(CACHE_DIR, pdf_name, "transcriptions")
        os.makedirs(out_dir, exist_ok=True)

        print("=" * 60)
        print("PDF Transcriber")
        print(f"Input : {self.pdf_path}")
        print(f"Cache : {out_dir}")
        print("=" * 60)

        doc = fitz.open(self.pdf_path)
        total = len(doc)

        for i, page in enumerate(doc):
            slide_num = i + 1
            text = page.get_text().strip()
            out_path = os.path.join(out_dir, f"slide_{slide_num:03d}.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"[{slide_num}/{total}] -> {os.path.basename(out_path)}")

        doc.close()
        print("=" * 60)
        print(f"[OK] {total} slides written to cache/{pdf_name}/transcriptions/")
        print("=" * 60)
        return out_dir
