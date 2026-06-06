import os
import fitz

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache")


class Transcriber:
    """Extracts text from each PDF page and saves it as a separate text file."""

    def __init__(self, pdf_path):
        """Initialize with the path to the PDF file."""
        self.pdf_path = pdf_path

    def run(self) -> str:
        """Extract all slides and write them to the cache directory."""
        if not os.path.isfile(self.pdf_path):
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")

        pdf_name = os.path.splitext(os.path.basename(self.pdf_path))[0]
        output_dir = os.path.join(CACHE_DIR, pdf_name, "transcriptions")
        os.makedirs(output_dir, exist_ok=True)

        print("=" * 60)
        print("PDF Transcriber")
        print(f"Input: {self.pdf_path}")
        print(f"Cache: {output_dir}")
        print("=" * 60)

        pdf_document = fitz.open(self.pdf_path)
        total = len(pdf_document)
        expected_filenames = {f"slide_{slide_number:03d}.txt" for slide_number in range(1, total + 1)}

        for filename in os.listdir(output_dir):
            if filename.startswith("slide_") and filename.endswith(".txt") and filename not in expected_filenames:
                os.remove(os.path.join(output_dir, filename))

        for page_index, page in enumerate(pdf_document):
            slide_number = page_index + 1
            slide_text = page.get_text().strip()
            slide_path = os.path.join(output_dir, f"slide_{slide_number:03d}.txt")
            with open(slide_path, "w", encoding="utf-8") as file_handle:
                file_handle.write(slide_text)
            print(f"[{slide_number}/{total}] -> {os.path.basename(slide_path)}")

        pdf_document.close()
        print("=" * 60)
        print(f"[OK] {total} slides written to cache/{pdf_name}/transcriptions/")
        print("=" * 60)
        return output_dir
