"""Extract structured slide content from PDF lecture decks.

The module defines extracted page data classes, file hashing, and
``Transcriber`` helpers for layout-aware text extraction and boilerplate removal.
"""

import hashlib
import json
import math
import os
import re
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from .utilities.normalizer import clean_line, is_bullet_line, repair_text

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache")
BOILERPLATE_PAGE_RATIO = 0.7
PAGE_NUMBER_PATTERN = re.compile(r"^(?:page\s+|slide\s+)?\d+(?:\s*/\s*\d+)?$", re.IGNORECASE)


@dataclass(frozen=True)
class ExtractedLine:
    """A line of PDF text with the layout information needed for cleanup."""

    text: str
    block_number: int
    bbox: tuple[float, float, float, float]
    font_size: float
    is_bold: bool
    page_height: float

    def position_key(self) -> tuple[str, int, int]:
        """Return a stable signature for repeated text at the same page position."""
        x_bucket = round(self.bbox[0] / 20)
        y_bucket = round(self.bbox[1] / 20)
        return self.text.casefold(), x_bucket, y_bucket


@dataclass(frozen=True)
class ExtractedPage:
    """Structured text and image metadata extracted from one PDF page."""

    slide_number: int
    lines: list[ExtractedLine]
    image_count: int


def get_file_hash(filepath: str) -> str:
    """Return the SHA-256 hash of a file without loading it fully into memory."""
    digest = hashlib.sha256()
    with open(filepath, "rb") as file_handle:
        while file_chunk := file_handle.read(1024 * 1024):
            digest.update(file_chunk)
    return digest.hexdigest()


class Transcriber:
    """Extract structured page text and remove repeated deck boilerplate."""

    def __init__(self, pdf_path: str, cache_dir: str | None = None) -> None:
        """Initialize the transcriber for a PDF and optional isolated cache directory."""
        self.pdf_path = pdf_path
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        self.cache_dir = cache_dir or os.path.join(CACHE_DIR, pdf_name)

    def extract_page(self, page: Any, slide_number: int) -> ExtractedPage:
        """Extract text lines, font cues, positions, and image count from one page."""
        page_content = page.get_text("dict", sort=True)
        extracted_lines: list[ExtractedLine] = []
        image_count = 0

        for block_number, block in enumerate(page_content.get("blocks", [])):
            if block.get("type") == 1:
                image_count += 1
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                line_text = clean_line(repair_text("".join(span.get("text", "") for span in spans)))
                if not line_text:
                    continue
                font_size = max((float(span.get("size", 0.0)) for span in spans), default=0.0)
                is_bold = any(int(span.get("flags", 0)) & 16 for span in spans)
                bbox = tuple(float(coordinate) for coordinate in line.get("bbox", (0.0, 0.0, 0.0, 0.0)))
                extracted_line = ExtractedLine(
                    line_text,
                    block_number,
                    bbox,
                    font_size,
                    is_bold,
                    float(page.rect.height)
                )
                extracted_lines.append(extracted_line)

        return ExtractedPage(slide_number, extracted_lines, image_count)

    def find_boilerplate_keys(self, pages: list[ExtractedPage]) -> set[tuple[str, int, int]]:
        """Identify text repeated at the same position across most deck pages."""
        if len(pages) < 2:
            return set()
        occurrence_counts: Counter[tuple[str, int, int]] = Counter()
        for page in pages:
            occurrence_counts.update({line.position_key() for line in page.lines})

        required_occurrences = max(2, math.ceil(len(pages) * BOILERPLATE_PAGE_RATIO))
        return {key for key, count in occurrence_counts.items() if count >= required_occurrences}

    def is_page_number(self, line: ExtractedLine) -> bool:
        """Return whether a line is page numbering rather than lecture content."""
        if not PAGE_NUMBER_PATTERN.fullmatch(line.text):
            return False
        return "/" in line.text or line.bbox[1] >= line.page_height * 0.8

    def choose_title(self, lines: list[ExtractedLine]) -> ExtractedLine | None:
        """Choose the strongest title candidate using font, weight, and page position."""
        candidates: list[ExtractedLine] = []
        for line in lines:
            if line.bbox[1] >= line.page_height * 0.65:
                continue
            if is_bullet_line(line.text):
                continue
            if len(line.text) < 3 or len(line.text) > 160:
                continue
            candidates.append(line)

        if not candidates:
            return None

        def title_score(line: ExtractedLine) -> tuple[float, int, float]:
            """Rank a possible title without relying on extraction order."""
            bold_score = 1 if line.is_bold else 0
            return line.font_size, bold_score, -line.bbox[1]

        return max(candidates, key=title_score)

    def get_content_lines(
        self,
        page: ExtractedPage,
        boilerplate_keys: set[tuple[str, int, int]]
    ) -> list[ExtractedLine]:
        """Remove repeated deck text and page numbers from one page."""
        content_lines: list[ExtractedLine] = []
        for line in page.lines:
            if line.position_key() in boilerplate_keys:
                continue
            if self.is_page_number(line):
                continue
            content_lines.append(line)
        return content_lines

    def group_body_blocks(
        self,
        content_lines: list[ExtractedLine],
        title_line: ExtractedLine | None
    ) -> list[str]:
        """Group non-title lines by their original PDF text block."""
        block_texts: list[str] = []
        current_block_number: int | None = None
        current_block_lines: list[str] = []

        for line in content_lines:
            if line is title_line:
                continue
            if current_block_number is not None and line.block_number != current_block_number:
                block_texts.append("\n".join(current_block_lines))
                current_block_lines = []
            current_block_number = line.block_number
            current_block_lines.append(line.text)

        if current_block_lines:
            block_texts.append("\n".join(current_block_lines))
        return block_texts

    def build_clean_text(
        self,
        page: ExtractedPage,
        boilerplate_keys: set[tuple[str, int, int]]
    ) -> tuple[str, str | None]:
        """Build readable slide text while keeping title and block boundaries."""
        content_lines = self.get_content_lines(page, boilerplate_keys)
        title_line = self.choose_title(content_lines)
        block_texts = self.group_body_blocks(content_lines, title_line)

        title = title_line.text if title_line is not None else None
        text_parts = [title] if title else []
        text_parts.extend(block for block in block_texts if block)
        return "\n\n".join(text_parts).strip(), title

    def write_page_artifacts(
        self,
        output_dir: str,
        page: ExtractedPage,
        text: str,
        title: str | None
    ) -> None:
        """Write human-readable text and structured JSON artifacts atomically."""
        filename_stem = f"slide_{page.slide_number:03d}"
        text_path = os.path.join(output_dir, f"{filename_stem}.txt")
        json_path = os.path.join(output_dir, f"{filename_stem}.json")
        artifact_id = uuid.uuid4().hex
        text_temp_path = f"{text_path}.{artifact_id}.tmp"
        json_temp_path = f"{json_path}.{artifact_id}.tmp"

        with open(text_temp_path, "w", encoding="utf-8") as file_handle:
            file_handle.write(text)
        os.replace(text_temp_path, text_path)

        payload = {
            "slide_number": page.slide_number,
            "title": title,
            "image_count": page.image_count,
            "lines": [asdict(line) for line in page.lines]
        }
        with open(json_temp_path, "w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, indent=2, ensure_ascii=False)
        os.replace(json_temp_path, json_path)

    def extract_all_pages(self, pdf_document: Any) -> list[ExtractedPage]:
        """Extract every PDF page and assign one-based slide numbers."""
        return [
            self.extract_page(page, page_index + 1)
            for page_index, page in enumerate(pdf_document)
        ]

    def remove_stale_artifacts(
        self,
        output_dir: str,
        expected_filenames: set[str]
    ) -> None:
        """Remove slide artifacts left by an older version of the same PDF."""
        for filename in os.listdir(output_dir):
            if filename.startswith("slide_") and filename not in expected_filenames:
                os.remove(os.path.join(output_dir, filename))

    def run(self) -> str:
        """Extract, clean, and persist every slide in the PDF."""
        import fitz

        if not os.path.isfile(self.pdf_path):
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")

        output_dir = os.path.join(self.cache_dir, "transcriptions")
        os.makedirs(output_dir, exist_ok=True)

        print("=" * 60)
        print("PDF Transcriber")
        print(f"Input: {self.pdf_path}")
        print(f"Cache: {output_dir}")
        print("=" * 60)

        pdf_document = fitz.open(self.pdf_path)
        try:
            pages = self.extract_all_pages(pdf_document)
        finally:
            pdf_document.close()

        boilerplate_keys = self.find_boilerplate_keys(pages)
        expected_filenames: set[str] = set()
        for page in pages:
            text, title = self.build_clean_text(page, boilerplate_keys)
            self.write_page_artifacts(output_dir, page, text, title)
            expected_filenames.add(f"slide_{page.slide_number:03d}.txt")
            expected_filenames.add(f"slide_{page.slide_number:03d}.json")
            print(f"[{page.slide_number}/{len(pages)}] -> slide_{page.slide_number:03d}.txt")

        self.remove_stale_artifacts(output_dir, expected_filenames)

        print("=" * 60)
        print(f"[OK] {len(pages)} slides written to {output_dir}")
        print("=" * 60)
        return output_dir
