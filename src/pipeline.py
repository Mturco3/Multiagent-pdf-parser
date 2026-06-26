import json
import os
import time
import hashlib
import shutil

from .checker import LLMChecker
from .math_formatter import MathFormatter
from .models import MathReplacementResponse, QualityReport, SlideReview, SlideRewrite, SlideType, TitleAnalysis
from .quality_checker import QualityChecker
from .rewriter import LLMRewriter
from .title_editor import TitleEditor
from .transcriber import CACHE_DIR, Transcriber
from .utilities.model_config import WINDOW_SECONDS, get_model_summary
from .utilities.prompts import (
    CHECKER_REVIEWER_PROMPT,
    CHECKER_SYSTEM_PROMPT,
    MATH_FORMATTER_PROMPT,
    QUALITY_CHECKER_PROMPT,
    REWRITE_REVIEWER_PROMPT,
    REWRITER_SYSTEM_PROMPT,
    TITLE_IDENTIFIER_PROMPT,
)

REWRITE_CACHE_MODES = {
    "rewrite_review_v2",
    "deterministic_passthrough_v2",
    "introduction_heading_v2",
}
CACHE_SOURCE_VERSION = "pipeline_cache_sources_v1"


class Pipeline:
    """Orchestrates the full flow from PDF input to final markdown output."""

    def __init__(self, pdf_path: str, clear_cache: bool = False):
        """Initialize the pipeline for a single input PDF."""
        self.pdf_path = pdf_path
        self.pdf_name = os.path.splitext(os.path.basename(os.path.normpath(pdf_path)))[0]
        self.cache_dir = os.path.join(CACHE_DIR, self.pdf_name)
        self.clear_cache = clear_cache

    def validate_input_pdf(self):
        """Fail before cache mutation when the configured input is not a readable PDF file."""
        if not self.pdf_name:
            raise ValueError("Input PDF path must include a file name.")

        if not os.path.isfile(self.pdf_path):
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")

    def ensure_safe_cache_target(self):
        """Make sure destructive cache operations stay inside this project's cache root."""
        cache_root = os.path.abspath(CACHE_DIR)
        target_dir = os.path.abspath(self.cache_dir)

        if target_dir == cache_root or os.path.commonpath([cache_root, target_dir]) != cache_root:
            raise ValueError(f"Refusing to clear unsafe cache path: {target_dir}")

    def reset_pdf_cache(self):
        """Remove only this PDF's cache directory before a fresh run."""
        self.ensure_safe_cache_target()
        if not os.path.exists(self.cache_dir):
            return

        shutil.rmtree(self.cache_dir)
        print("=" * 60)
        print(f"[RESET] Cleared cache/{self.pdf_name}/")
        print("=" * 60)

    def get_text_hash(self, text: str) -> str:
        """Return a stable hash for a document-sized text input."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get_cache_source(self, *parts) -> str:
        """Build a stable source string for cache provenance checks."""
        source_parts: list[str] = []
        for part in parts:
            if isinstance(part, str):
                source_parts.append(part)
            else:
                source_parts.append(json.dumps(part, sort_keys=True, ensure_ascii=False))
        return "\n--- cache-source-part ---\n".join(source_parts)

    def get_review_cache_source(self, slide_text: str) -> str:
        """Return the source fingerprint input for checker/reviewer review artifacts."""
        return self.get_cache_source(
            CACHE_SOURCE_VERSION,
            "review",
            CHECKER_SYSTEM_PROMPT,
            CHECKER_REVIEWER_PROMPT,
            SlideReview.model_json_schema(),
            slide_text,
        )

    def get_rewrite_cache_source(self, slide_text: str, review: SlideReview) -> str:
        """Return the source fingerprint input for rewritten slide artifacts."""
        return self.get_cache_source(
            CACHE_SOURCE_VERSION,
            "rewrite",
            REWRITER_SYSTEM_PROMPT,
            REWRITE_REVIEWER_PROMPT,
            SlideRewrite.model_json_schema(),
            slide_text,
            review.model_dump(mode="json"),
        )

    def get_math_cache_source(self, slide: SlideRewrite) -> str:
        """Return the source fingerprint input for math-formatted slide artifacts."""
        return self.get_cache_source(
            CACHE_SOURCE_VERSION,
            "math",
            MATH_FORMATTER_PROMPT,
            MathReplacementResponse.model_json_schema(),
            SlideRewrite.model_json_schema(),
            slide.model_dump(mode="json"),
        )

    def get_title_cache_source(self, document: str) -> str:
        """Return the source fingerprint input for title-analysis artifacts."""
        return self.get_cache_source(
            CACHE_SOURCE_VERSION,
            "title",
            TITLE_IDENTIFIER_PROMPT,
            TitleAnalysis.model_json_schema(),
            document,
        )

    def get_quality_cache_source(self, document: str) -> str:
        """Return the source fingerprint input for quality-report artifacts."""
        return self.get_cache_source(
            CACHE_SOURCE_VERSION,
            "quality",
            QUALITY_CHECKER_PROMPT,
            QualityReport.model_json_schema(),
            document,
        )

    def save_json(self, name: str, data, source_text: str | None = None):
        """Save a JSON-serializable object in the PDF cache directory."""
        json_path = os.path.join(self.cache_dir, f"{name}.json")
        payload = data
        if source_text is not None:
            payload = {"source_hash": self.get_text_hash(source_text), "data": data}

        with open(json_path, "w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, indent=2, ensure_ascii=False)
        print(f"[saved] {name}.json")

    def load_json(self, name: str, source_text: str | None = None) -> dict | list | None:
        """Load a cached JSON object when it exists and still matches its source."""
        json_path = os.path.join(self.cache_dir, f"{name}.json")
        if not os.path.exists(json_path):
            return None

        with open(json_path, encoding="utf-8") as file_handle:
            payload = json.load(file_handle)

        if source_text is None:
            return payload

        if not isinstance(payload, dict):
            return None

        if payload.get("source_hash") != self.get_text_hash(source_text):
            return None

        return payload.get("data")

    def clear_json_files(self, directory: str, suffix: str = ".json"):
        """Remove cached JSON files so stale artifacts cannot be reused."""
        if not os.path.exists(directory):
            return

        for filename in os.listdir(directory):
            if not filename.endswith(suffix):
                continue

            filepath = os.path.join(directory, filename)
            os.remove(filepath)

    def save_review_json(self, reviews_dir: str, review: SlideReview, source_text: str):
        """Persist one slide review immediately so the checker can resume after failures."""
        review_filename = f"slide_{review.slide_number:03d}_review.json"
        review_path = os.path.join(reviews_dir, review_filename)
        payload = {
            "source_hash": self.get_text_hash(source_text),
            "data": review.model_dump(mode="json"),
        }
        with open(review_path, "w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, indent=2, ensure_ascii=False)

    def load_review_json(self, reviews_dir: str, slide_number: int, source_text: str) -> SlideReview | None:
        """Load one cached review when it matches the current schema."""
        review_filename = f"slide_{slide_number:03d}_review.json"
        review_path = os.path.join(reviews_dir, review_filename)
        if not os.path.exists(review_path):
            return None

        with open(review_path, encoding="utf-8") as file_handle:
            payload = json.load(file_handle)

        if not isinstance(payload, dict) or payload.get("source_hash") != self.get_text_hash(source_text):
            os.remove(review_path)
            return None

        review_payload = payload.get("data")
        if not isinstance(review_payload, dict):
            os.remove(review_path)
            return None

        if "reviewer_approved" not in review_payload or "checker_attempts" not in review_payload:
            os.remove(review_path)
            return None

        return SlideReview(**review_payload)

    def save_slide_jsons(self, directory: str, slides: list[SlideRewrite]):
        """Save per-slide rewrite artifacts after clearing stale files."""
        dir_path = os.path.join(self.cache_dir, directory)
        os.makedirs(dir_path, exist_ok=True)
        self.clear_json_files(dir_path)

        for slide in slides:
            filename = f"slide_{slide.slide_number:03d}.json"
            filepath = os.path.join(dir_path, filename)
            with open(filepath, "w", encoding="utf-8") as file_handle:
                json.dump(slide.model_dump(), file_handle, indent=2, ensure_ascii=False)

        print(f"[saved] {len(slides)} slides to {directory}/")

    def save_slide_json(self, directory: str, slide: SlideRewrite, source_text: str | None = None):
        """Persist one slide artifact immediately so long stages can resume safely."""
        dir_path = os.path.join(self.cache_dir, directory)
        os.makedirs(dir_path, exist_ok=True)
        filename = f"slide_{slide.slide_number:03d}.json"
        filepath = os.path.join(dir_path, filename)
        payload = slide.model_dump(mode="json")
        if source_text is not None:
            payload = {
                "source_hash": self.get_text_hash(source_text),
                "data": payload,
            }
        with open(filepath, "w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, indent=2, ensure_ascii=False)

    def load_slide_json(self, directory: str, slide_number: int, source_text: str | None = None) -> SlideRewrite | None:
        """Load one cached slide artifact when it matches the current schema."""
        dir_path = os.path.join(self.cache_dir, directory)
        filepath = os.path.join(dir_path, f"slide_{slide_number:03d}.json")
        if not os.path.exists(filepath):
            return None

        with open(filepath, encoding="utf-8") as file_handle:
            payload = json.load(file_handle)

        if source_text is not None:
            if not isinstance(payload, dict) or payload.get("source_hash") != self.get_text_hash(source_text):
                os.remove(filepath)
                return None
            payload = payload.get("data")

        if not isinstance(payload, dict) or payload.get("rewrite_mode") not in REWRITE_CACHE_MODES:
            os.remove(filepath)
            return None

        return SlideRewrite(**payload)

    def load_slide_jsons(
        self,
        directory: str,
        expected_slide_numbers: list[int],
        source_by_slide: dict[int, str] | None = None,
    ) -> list[SlideRewrite] | None:
        """Load cached slide artifacts only when the file set matches exactly."""
        dir_path = os.path.join(self.cache_dir, directory)
        if not os.path.exists(dir_path):
            return None

        expected_filenames: list[str] = []
        for slide_number in expected_slide_numbers:
            expected_filenames.append(f"slide_{slide_number:03d}.json")

        cached_filenames = os.listdir(dir_path)
        cached_filenames = [name for name in cached_filenames if name.startswith("slide_") and name.endswith(".json")]
        cached_filenames.sort()
        if cached_filenames != expected_filenames:
            return None

        slides: list[SlideRewrite] = []
        for filename in expected_filenames:
            slide_number = int(filename.removeprefix("slide_").removesuffix(".json"))
            source_text = source_by_slide.get(slide_number) if source_by_slide else None
            slide = self.load_slide_json(directory, slide_number, source_text)
            if slide is None:
                return None
            slides.append(slide)

        return slides

    def get_output_slide_numbers(self, reviews: list[SlideReview]) -> list[int]:
        """Return the slide numbers that should appear in the final document."""
        slide_numbers: list[int] = []

        for review in reviews:
            if self.is_outline_review(review):
                continue

            if review.slide_type in (SlideType.COURSE_INFO, SlideType.IMAGE_DESCRIPTION):
                continue

            if review.slide_type == SlideType.INTRODUCTION and not review.title:
                continue

            slide_numbers.append(review.slide_number)

        return slide_numbers

    def is_outline_review(self, review: SlideReview) -> bool:
        """Return whether a slide review is an agenda/outline slide that should not become notes."""
        if not review.title:
            return False

        normalized_title = review.title.strip().lower()
        normalized_title = normalized_title.rstrip(":")
        outline_titles = {
            "agenda",
            "contents",
            "outline",
            "overview",
            "roadmap",
            "table of contents",
            "today",
            "today's agenda",
            "today's outline",
        }
        return normalized_title in outline_titles

    def get_slide_number(self, filename: str) -> int:
        """Extract the slide number from a cached transcription filename."""
        number_text = filename.removeprefix("slide_")
        number_text = number_text.removesuffix(".txt")
        return int(number_text)

    def assemble(self, slides: list[SlideRewrite]) -> str:
        """Concatenate slide rewrites into a single markdown document."""
        parts: list[str] = []
        previous_title: str | None = None

        for slide in slides:
            section_parts: list[str] = []

            if slide.is_continuation:
                if slide.text:
                    section_parts.append(slide.text)
            else:
                # Only emit a heading when the slide has body text to follow it
                if slide.title and slide.title != previous_title and slide.text:
                    section_parts.append(f"## {slide.title}")
                    previous_title = slide.title

                if slide.text:
                    section_parts.append(slide.text)

            if section_parts:
                parts.append("\n".join(section_parts))

        return "\n\n".join(parts)

    def print_model_configuration(self):
        """Show the active stage-to-model mapping before the pipeline starts."""
        print("=" * 60)
        print("Model Configuration")
        for stage, model_name, rpm, rpd in get_model_summary():
            print(f"{stage:18} {model_name}  rpm={rpm}  rpd={rpd}")
        print("=" * 60)

    def run(self):
        """Execute the full pipeline and save the final markdown file."""
        self.validate_input_pdf()

        if self.clear_cache:
            self.reset_pdf_cache()

        os.makedirs(self.cache_dir, exist_ok=True)
        self.print_model_configuration()

        transcriber = Transcriber(self.pdf_path)
        transcriptions_dir = transcriber.run()

        reviews_dir = os.path.join(self.cache_dir, "reviews")
        os.makedirs(reviews_dir, exist_ok=True)

        slide_filenames = os.listdir(transcriptions_dir)
        slide_filenames = [name for name in slide_filenames if name.startswith("slide_") and name.endswith(".txt")]
        slide_filenames.sort()

        if not slide_filenames:
            print("No slides found in cache.")
            return

        slides: list[tuple[int, str]] = []
        for filename in slide_filenames:
            slide_number = self.get_slide_number(filename)
            filepath = os.path.join(transcriptions_dir, filename)
            with open(filepath, encoding="utf-8") as file_handle:
                slides.append((slide_number, file_handle.read()))

        total = len(slides)

        # Checker produces per-slide review metadata.
        reviews = self._run_checker(slides, reviews_dir, total)
        output_slide_numbers = self.get_output_slide_numbers(reviews)

        # Rewriter produces the slide text that will be assembled later.
        rewrites = self._run_rewriter(slides, reviews, output_slide_numbers)

        # Math formatting runs on the rewritten slide outputs only.
        math_slides = self._run_math_formatter(rewrites, output_slide_numbers)

        print("\n" + "=" * 60)
        print("Assembling document")
        print("=" * 60)
        document = self.assemble(math_slides)

        # Title editing updates markdown headings after assembly.
        document = self._run_title_editor(document)

        # Quality checking makes targeted fixes to the full document.
        document = self._run_quality_checker(document)

        output_path = os.path.join(self.cache_dir, f"{self.pdf_name}.md")
        with open(output_path, "w", encoding="utf-8") as file_handle:
            file_handle.write(document)

        print("\n" + "=" * 60)
        print(f"[DONE] Final document saved to cache/{self.pdf_name}/{self.pdf_name}.md")
        print("=" * 60)

    def _run_checker(self, slides: list[tuple[int, str]], reviews_dir: str, total: int) -> list[SlideReview]:
        """Run the checker step, reusing and extending any valid per-slide cached reviews."""
        expected_review_filenames = {f"slide_{slide_number:03d}_review.json" for slide_number, _ in slides}
        for filename in os.listdir(reviews_dir):
            if not filename.endswith("_review.json"):
                continue
            if filename not in expected_review_filenames:
                os.remove(os.path.join(reviews_dir, filename))

        print("=" * 60)
        print("LLM Checker")
        print(f"Slides: {total}")
        print(f"Output: cache/{self.pdf_name}/reviews/")
        print("=" * 60)

        checker = LLMChecker()
        reviews: list[SlideReview] = []
        for index, (slide_number, text) in enumerate(slides, start=1):
            print(f"[{slide_number}/{total}]", end=" ", flush=True)
            review_source = self.get_review_cache_source(text)
            cached_review = self.load_review_json(reviews_dir, slide_number, review_source)
            if cached_review is not None:
                print(f"cached - {cached_review.slide_type.value}")
                reviews.append(cached_review)
                continue

            review = checker.check_one(slide_number, text)
            self.save_review_json(reviews_dir, review, review_source)
            reviews.append(review)

        print("\n" + "=" * 60)
        print("Writing review files...")
        for review in reviews:
            review_filename = f"slide_{review.slide_number:03d}_review.json"

            title_display = f'"{review.title}"' if review.title else "no title"
            actions_display = f"{len(review.actions)} action(s)" if review.actions else "no actions"
            print(f"-> {review_filename} [{review.slide_type.value}] {title_display} {actions_display}")

        print("=" * 60)
        print(f"[DONE] {total} reviews saved in cache/{self.pdf_name}/reviews/")
        print("=" * 60)
        return reviews

    def _run_rewriter(self, slides: list[tuple[int, str]], reviews: list[SlideReview], output_slide_numbers: list[int]) -> list[SlideRewrite]:
        """Run the rewriter step or load exact cached rewrites."""
        review_by_number = {review.slide_number: review for review in reviews}
        source_by_number = {
            slide_number: self.get_rewrite_cache_source(text, review_by_number[slide_number])
            for slide_number, text in slides
            if slide_number in output_slide_numbers
        }

        cached = self.load_slide_jsons("rewrites", output_slide_numbers, source_by_number)
        if cached is not None:
            print("=" * 60)
            print("Loading cached rewrites")
            print("=" * 60)
            return cached

        print(f"\nWaiting {WINDOW_SECONDS}s before rewriting step...")
        time.sleep(WINDOW_SECONDS)

        print("\n" + "=" * 60)
        print("LLM Rewriter")
        print(f"Slides: {len(slides)}")
        print("=" * 60)

        rewriter = LLMRewriter()
        rewrites_dir = os.path.join(self.cache_dir, "rewrites")
        os.makedirs(rewrites_dir, exist_ok=True)

        valid_rewrite_filenames = {f"slide_{slide_number:03d}.json" for slide_number in output_slide_numbers}
        for filename in os.listdir(rewrites_dir):
            if filename not in valid_rewrite_filenames:
                os.remove(os.path.join(rewrites_dir, filename))

        rewrites: list[SlideRewrite] = []
        previous_paragraph: str | None = None
        current_section_title: str | None = None
        total = len(slides)

        for slide_number, text in slides:
            review = review_by_number[slide_number]
            print(f"[{slide_number}/{total}]", end=" ", flush=True)

            cached_slide = self.load_slide_json("rewrites", slide_number, source_by_number.get(slide_number))
            if cached_slide is not None:
                print("cached rewrite")
                rewrites.append(cached_slide)
                if cached_slide.title and not cached_slide.is_continuation:
                    current_section_title = cached_slide.title
                previous_paragraph = cached_slide.text or None
                continue

            transition_paragraph = previous_paragraph
            starts_new_section = bool(review.title and review.title != current_section_title and not review.is_continuation)
            if starts_new_section:
                transition_paragraph = None

            should_call_llm = (
                review.slide_type not in (SlideType.COURSE_INFO, SlideType.IMAGE_DESCRIPTION, SlideType.INTRODUCTION)
                and review.reviewer_approved
                and bool(review.actions)
            )

            rewrite = rewriter.rewrite_one(text, review, transition_paragraph if should_call_llm else None)

            if rewrite is None:
                continue

            self.save_slide_json("rewrites", rewrite, source_by_number.get(slide_number))
            rewrites.append(rewrite)

            if rewrite.title and not rewrite.is_continuation:
                current_section_title = rewrite.title

            previous_paragraph = rewrite.text or None

        return rewrites

    def _run_math_formatter(self, slides: list[SlideRewrite], output_slide_numbers: list[int]) -> list[SlideRewrite]:
        """Run the math formatter step or load exact cached results."""
        source_by_number = {
            slide.slide_number: self.get_math_cache_source(slide)
            for slide in slides
        }

        cached = self.load_slide_jsons("math", output_slide_numbers, source_by_number)
        if cached is not None:
            print("=" * 60)
            print("Loading cached math formatted slides")
            print("=" * 60)
            return cached

        print(f"\nWaiting {WINDOW_SECONDS}s before math formatting step...")
        time.sleep(WINDOW_SECONDS)

        print("\n" + "=" * 60)
        print("Math Formatter")
        print(f"Slides: {len(slides)}")
        print("=" * 60)

        formatter = MathFormatter()
        math_dir = os.path.join(self.cache_dir, "math_replacements")
        os.makedirs(math_dir, exist_ok=True)
        math_cache_dir = os.path.join(self.cache_dir, "math")
        os.makedirs(math_cache_dir, exist_ok=True)

        valid_math_filenames = {f"slide_{slide.slide_number:03d}.json" for slide in slides}
        for filename in os.listdir(math_cache_dir):
            if filename not in valid_math_filenames:
                os.remove(os.path.join(math_cache_dir, filename))

        valid_replacement_filenames = {f"slide_{slide.slide_number:03d}_replacements.json" for slide in slides}
        for filename in os.listdir(math_dir):
            if filename not in valid_replacement_filenames:
                os.remove(os.path.join(math_dir, filename))

        updated_slides: list[SlideRewrite] = []
        for slide in slides:
            cached_slide = self.load_slide_json("math", slide.slide_number, source_by_number[slide.slide_number])
            if cached_slide is not None:
                print(f"[slide {slide.slide_number:03d}] cached math")
                updated_slides.append(cached_slide)
                continue

            updated_slide, response = formatter.format_slide(slide)
            self.save_slide_json("math", updated_slide, source_by_number[slide.slide_number])

            filename = f"slide_{slide.slide_number:03d}_replacements.json"
            filepath = os.path.join(math_dir, filename)
            if response.replacements:
                with open(filepath, "w", encoding="utf-8") as file_handle:
                    json.dump(response.model_dump(), file_handle, indent=2, ensure_ascii=False)
            elif os.path.exists(filepath):
                os.remove(filepath)

            updated_slides.append(updated_slide)

        return updated_slides

    def _run_title_editor(self, document: str) -> str:
        """Run title analysis and apply the resulting heading edits."""
        print("\n" + "=" * 60)
        print("Title Editor")
        print("=" * 60)

        title_editor = TitleEditor()
        title_source = self.get_title_cache_source(document)
        cached_analysis = self.load_json("title_analysis", title_source)
        if cached_analysis:
            print("Loading cached title analysis")
            analysis = TitleAnalysis(**cached_analysis)
        else:
            print(f"Waiting {WINDOW_SECONDS}s before title editing step...")
            time.sleep(WINDOW_SECONDS)
            analysis = title_editor.edit(document)
            self.save_json("title_analysis", analysis.model_dump(), title_source)

        document = title_editor.apply(document, analysis)
        return document

    def _run_quality_checker(self, document: str) -> str:
        """Run the quality checker and apply any requested fixes."""
        print("\n" + "=" * 60)
        print("Quality Checker")
        print("=" * 60)

        quality_checker = QualityChecker()
        document = quality_checker.sanitize_document(document)
        quality_source = self.get_quality_cache_source(document)
        cached_report = self.load_json("quality_report", quality_source)
        if cached_report:
            print("Loading cached quality report")
            report = QualityReport(**cached_report)
        else:
            print(f"Waiting {WINDOW_SECONDS}s before quality check step...")
            time.sleep(WINDOW_SECONDS)
            try:
                report = quality_checker.check(document)
                self.save_json("quality_report", report.model_dump(), quality_source)
            except Exception as error:
                print(f"[warn] Quality checker model failed; using deterministic cleanup only ({error})")
                report = QualityReport(issues=[])

        if report.issues:
            document = quality_checker.fix(document, report)

        return quality_checker.sanitize_document(document)
