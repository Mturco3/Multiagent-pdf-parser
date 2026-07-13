import hashlib
import json
import os
import shutil
import uuid

from pydantic import ValidationError

from .checker import LLMChecker
from .math_formatter import MathFormatter
from .models import MathReplacementResponse, QualityReport, SlideReview, SlideRewrite, SlideType, TitleAnalysis
from .quality_checker import QualityChecker
from .rewriter import LLMRewriter
from .title_editor import TitleEditor
from .transcriber import CACHE_DIR, Transcriber, get_file_hash
from .utilities.model_config import CHECKER_MODEL, FALLBACK_MODEL, MATH_MODEL, QUALITY_FIXER_MODEL, QUALITY_IDENTIFIER_MODEL, REWRITER_MODEL, TITLE_MODEL, get_model_summary
from .utilities.prompts import CHECKER_SYSTEM_PROMPT, MATH_FORMATTER_PROMPT, QUALITY_CHECKER_PROMPT, QUALITY_FIXER_PROMPT, REWRITER_SYSTEM_PROMPT, TITLE_IDENTIFIER_PROMPT

REWRITE_CACHE_MODES = {"validated_rewrite_v3", "deterministic_passthrough_v3", "introduction_heading_v3"}
CACHE_SOURCE_VERSION = "pipeline_cache_sources_v3"


class Pipeline:
    """Convert one lecture PDF into source-aware Markdown notes."""

    def __init__(self, pdf_path: str, clear_cache: bool = False, cache_root: str = CACHE_DIR):
        """Initialize input, cache policy, and the provisional output location."""
        self.pdf_path = pdf_path
        self.pdf_name = os.path.splitext(os.path.basename(os.path.normpath(pdf_path)))[0]
        self.cache_root = cache_root
        self.cache_dir = os.path.join(cache_root, self.pdf_name)
        self.pdf_fingerprint: str | None = None
        self.clear_cache = clear_cache

    def validate_input_pdf(self):
        """Fail before cache mutation when the input is not a readable PDF file."""
        if not self.pdf_name:
            raise ValueError("Input PDF path must include a file name.")
        if not os.path.isfile(self.pdf_path):
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")
        if os.path.splitext(self.pdf_path)[1].casefold() != ".pdf":
            raise ValueError(f"Input file must use the .pdf extension: {self.pdf_path}")

    def configure_cache_identity(self):
        """Use PDF content rather than its filename as the cache namespace."""
        self.pdf_fingerprint = get_file_hash(self.pdf_path)
        cache_name = f"{self.pdf_name}-{self.pdf_fingerprint[:12]}"
        self.cache_dir = os.path.join(self.cache_root, cache_name)

    def ensure_safe_cache_target(self):
        """Ensure destructive cache operations remain inside the configured root."""
        cache_root = os.path.abspath(self.cache_root)
        target_dir = os.path.abspath(self.cache_dir)
        if target_dir == cache_root or os.path.commonpath([cache_root, target_dir]) != cache_root:
            raise ValueError(f"Refusing to clear unsafe cache path: {target_dir}")

    def reset_pdf_cache(self):
        """Remove only the current content-addressed PDF cache directory."""
        self.ensure_safe_cache_target()
        if not os.path.exists(self.cache_dir):
            return
        shutil.rmtree(self.cache_dir)
        print("=" * 60)
        print(f"[RESET] Cleared {self.cache_dir}")
        print("=" * 60)

    def get_text_hash(self, text: str) -> str:
        """Return a stable SHA-256 hash for cache provenance."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get_cache_source(self, *parts) -> str:
        """Build a stable source string from text and JSON-compatible values."""
        source_parts: list[str] = []
        for part in parts:
            if isinstance(part, str):
                source_parts.append(part)
            else:
                source_parts.append(json.dumps(part, sort_keys=True, ensure_ascii=False))
        return "\n--- cache-source-part ---\n".join(source_parts)

    def get_review_cache_source(self, slide_text: str) -> str:
        """Return provenance for checker artifacts."""
        return self.get_cache_source(CACHE_SOURCE_VERSION, "review", CHECKER_MODEL, FALLBACK_MODEL, CHECKER_SYSTEM_PROMPT, SlideReview.model_json_schema(), slide_text)

    def get_rewrite_cache_source(self, slide_text: str, review: SlideReview) -> str:
        """Return provenance for context-independent rewrite artifacts."""
        return self.get_cache_source(CACHE_SOURCE_VERSION, "rewrite", REWRITER_MODEL, FALLBACK_MODEL, REWRITER_SYSTEM_PROMPT, SlideRewrite.model_json_schema(), slide_text, review.model_dump(mode="json"))

    def get_math_cache_source(self, slide: SlideRewrite) -> str:
        """Return provenance for math-formatting artifacts."""
        return self.get_cache_source(CACHE_SOURCE_VERSION, "math", MATH_MODEL, FALLBACK_MODEL, MATH_FORMATTER_PROMPT, MathReplacementResponse.model_json_schema(), SlideRewrite.model_json_schema(), slide.model_dump(mode="json"))

    def get_title_cache_source(self, document: str) -> str:
        """Return provenance for heading-analysis artifacts."""
        return self.get_cache_source(CACHE_SOURCE_VERSION, "title", TITLE_MODEL, FALLBACK_MODEL, TITLE_IDENTIFIER_PROMPT, TitleAnalysis.model_json_schema(), document)

    def get_quality_cache_source(self, source_document: str, document: str) -> str:
        """Return provenance for source-aware quality reports and fixes."""
        return self.get_cache_source(CACHE_SOURCE_VERSION, "quality", QUALITY_IDENTIFIER_MODEL, QUALITY_FIXER_MODEL, FALLBACK_MODEL, QUALITY_CHECKER_PROMPT, QUALITY_FIXER_PROMPT, QualityReport.model_json_schema(), source_document, document)

    def write_json_atomic(self, filepath: str, payload):
        """Write JSON atomically so interrupted writes cannot corrupt a valid cache."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        temp_path = f"{filepath}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        with open(temp_path, "w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, indent=2, ensure_ascii=False)
        os.replace(temp_path, filepath)

    def write_text_atomic(self, filepath: str, text: str):
        """Write text atomically so readers never observe partial documents."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        temp_path = f"{filepath}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        with open(temp_path, "w", encoding="utf-8") as file_handle:
            file_handle.write(text)
        os.replace(temp_path, filepath)

    def load_json_file(self, filepath: str):
        """Load cached JSON and discard a corrupt partial artifact."""
        if not os.path.exists(filepath):
            return None
        try:
            with open(filepath, encoding="utf-8") as file_handle:
                return json.load(file_handle)
        except (OSError, json.JSONDecodeError):
            try:
                os.remove(filepath)
            except OSError:
                pass
            return None

    def save_json(self, name: str, payload, source_text: str | None = None):
        """Save a document-level JSON artifact with optional provenance."""
        filepath = os.path.join(self.cache_dir, f"{name}.json")
        stored_payload = payload
        if source_text is not None:
            stored_payload = {"source_hash": self.get_text_hash(source_text), "data": payload}
        self.write_json_atomic(filepath, stored_payload)
        print(f"[saved] {name}.json")

    def load_json(self, name: str, source_text: str | None = None):
        """Load a document artifact only when its provenance matches."""
        filepath = os.path.join(self.cache_dir, f"{name}.json")
        payload = self.load_json_file(filepath)
        if payload is None or source_text is None:
            return payload
        if not isinstance(payload, dict) or payload.get("source_hash") != self.get_text_hash(source_text):
            return None
        return payload.get("data")

    def save_review_json(self, reviews_dir: str, review: SlideReview, source_text: str):
        """Persist one validated slide review with source provenance."""
        filepath = os.path.join(reviews_dir, f"slide_{review.slide_number:03d}_review.json")
        payload = {"source_hash": self.get_text_hash(source_text), "data": review.model_dump(mode="json")}
        self.write_json_atomic(filepath, payload)

    def load_review_json(self, reviews_dir: str, slide_number: int, source_text: str) -> SlideReview | None:
        """Load one review when provenance and schema remain valid."""
        filepath = os.path.join(reviews_dir, f"slide_{slide_number:03d}_review.json")
        payload = self.load_json_file(filepath)
        if not isinstance(payload, dict) or payload.get("source_hash") != self.get_text_hash(source_text):
            return None
        review_payload = payload.get("data")
        if not isinstance(review_payload, dict):
            return None
        try:
            return SlideReview(**review_payload)
        except ValidationError:
            return None

    def save_slide_json(self, directory: str, slide: SlideRewrite, source_text: str):
        """Persist one rewrite or math slide with source provenance."""
        dir_path = os.path.join(self.cache_dir, directory)
        filepath = os.path.join(dir_path, f"slide_{slide.slide_number:03d}.json")
        payload = {"source_hash": self.get_text_hash(source_text), "data": slide.model_dump(mode="json")}
        self.write_json_atomic(filepath, payload)

    def load_slide_json(self, directory: str, slide_number: int, source_text: str) -> SlideRewrite | None:
        """Load one slide artifact when provenance, mode, and schema remain valid."""
        filepath = os.path.join(self.cache_dir, directory, f"slide_{slide_number:03d}.json")
        payload = self.load_json_file(filepath)
        if not isinstance(payload, dict) or payload.get("source_hash") != self.get_text_hash(source_text):
            return None
        slide_payload = payload.get("data")
        if not isinstance(slide_payload, dict) or slide_payload.get("rewrite_mode") not in REWRITE_CACHE_MODES:
            return None
        try:
            return SlideRewrite(**slide_payload)
        except ValidationError:
            return None

    def load_slide_jsons(self, directory: str, expected_slide_numbers: list[int], source_by_slide: dict[int, str]) -> list[SlideRewrite] | None:
        """Load a complete stage only when its exact file set and provenance match."""
        dir_path = os.path.join(self.cache_dir, directory)
        if not os.path.exists(dir_path):
            return None
        expected_filenames = {f"slide_{slide_number:03d}.json" for slide_number in expected_slide_numbers}
        cached_filenames = {name for name in os.listdir(dir_path) if name.startswith("slide_") and name.endswith(".json")}
        if cached_filenames != expected_filenames:
            return None
        slides: list[SlideRewrite] = []
        for slide_number in expected_slide_numbers:
            slide = self.load_slide_json(directory, slide_number, source_by_slide[slide_number])
            if slide is None:
                return None
            slides.append(slide)
        return slides

    def remove_unexpected_files(self, directory: str, valid_filenames: set[str], suffix: str):
        """Remove stale stage files while leaving unrelated artifacts untouched."""
        if not os.path.exists(directory):
            return
        for filename in os.listdir(directory):
            if filename.endswith(suffix) and filename not in valid_filenames:
                os.remove(os.path.join(directory, filename))

    def is_outline_review(self, review: SlideReview) -> bool:
        """Return whether a review represents a deck agenda rather than notes."""
        if not review.title:
            return False
        normalized_title = review.title.strip().lower().rstrip(":")
        outline_titles = {"agenda", "contents", "outline", "overview", "roadmap", "table of contents", "today", "today's agenda", "today's outline"}
        return normalized_title in outline_titles

    def get_output_slide_numbers(self, reviews: list[SlideReview]) -> list[int]:
        """Return slides that should contribute a heading, text, or figure caption."""
        slide_numbers: list[int] = []
        for review in reviews:
            if self.is_outline_review(review):
                continue
            if review.slide_type == SlideType.COURSE_INFO:
                continue
            if review.slide_type == SlideType.INTRODUCTION and not review.title:
                continue
            slide_numbers.append(review.slide_number)
        return slide_numbers

    def get_slide_number(self, filename: str) -> int:
        """Extract a numeric slide identifier from a transcription filename."""
        return int(filename.removeprefix("slide_").removesuffix(".txt"))

    def assemble(self, slides: list[SlideRewrite]) -> str:
        """Assemble slide rewrites while retaining meaningful heading-only sections."""
        parts: list[str] = []
        previous_title: str | None = None
        for slide in slides:
            section_parts: list[str] = []
            if slide.is_continuation:
                if slide.text:
                    section_parts.append(slide.text)
            else:
                if slide.title and slide.title != previous_title:
                    section_parts.append(f"## {slide.title}")
                    previous_title = slide.title
                if slide.text:
                    section_parts.append(slide.text)
            if section_parts:
                parts.append("\n".join(section_parts))
        return "\n\n".join(parts)

    def print_model_configuration(self):
        """Print the active model and local quota mapping."""
        print("=" * 60)
        print("Model Configuration")
        for stage, model_name, rpm, rpd in get_model_summary():
            print(f"{stage:18} {model_name}  rpm={rpm}  rpd={rpd}")
        print("=" * 60)

    def run(self) -> str:
        """Execute all stages, save the final Markdown document, and return it."""
        self.validate_input_pdf()
        self.configure_cache_identity()
        if self.clear_cache:
            self.reset_pdf_cache()
        os.makedirs(self.cache_dir, exist_ok=True)
        self.print_model_configuration()

        transcriber = Transcriber(self.pdf_path, self.cache_dir)
        transcriptions_dir = transcriber.run()
        slide_filenames = [name for name in os.listdir(transcriptions_dir) if name.startswith("slide_") and name.endswith(".txt")]
        slide_filenames.sort(key=self.get_slide_number)
        if not slide_filenames:
            raise ValueError("The PDF did not produce any readable slides.")

        slides: list[tuple[int, str]] = []
        for filename in slide_filenames:
            slide_number = self.get_slide_number(filename)
            filepath = os.path.join(transcriptions_dir, filename)
            with open(filepath, encoding="utf-8") as file_handle:
                slides.append((slide_number, file_handle.read()))

        reviews_dir = os.path.join(self.cache_dir, "reviews")
        os.makedirs(reviews_dir, exist_ok=True)
        reviews = self.run_checker_stage(slides, reviews_dir)
        output_slide_numbers = self.get_output_slide_numbers(reviews)
        rewrites = self.run_rewriter_stage(slides, reviews, output_slide_numbers)
        math_slides = self.run_math_stage(rewrites)

        print("\n" + "=" * 60)
        print("Assembling document")
        print("=" * 60)
        document = self.assemble(math_slides)
        document = self.run_title_stage(document)
        source_document = "\n\n".join(f"[Slide {slide_number}]\n{text}" for slide_number, text in slides if slide_number in output_slide_numbers)
        document = self.run_quality_stage(source_document, document)

        output_path = os.path.join(self.cache_dir, f"{self.pdf_name}.md")
        self.write_text_atomic(output_path, document)
        print("\n" + "=" * 60)
        print(f"[DONE] Final document saved to {output_path}")
        print("=" * 60)
        return document

    def run_checker_stage(self, slides: list[tuple[int, str]], reviews_dir: str) -> list[SlideReview]:
        """Run or resume deterministic checker validation for every slide."""
        expected_filenames = {f"slide_{slide_number:03d}_review.json" for slide_number, text in slides}
        self.remove_unexpected_files(reviews_dir, expected_filenames, "_review.json")
        print("=" * 60)
        print("LLM Checker")
        print(f"Slides: {len(slides)}")
        print("=" * 60)

        checker = LLMChecker()
        reviews: list[SlideReview] = []
        for position, (slide_number, text) in enumerate(slides, start=1):
            print(f"[{position}/{len(slides)}]", end=" ", flush=True)
            source = self.get_review_cache_source(text)
            cached_review = self.load_review_json(reviews_dir, slide_number, source)
            if cached_review is not None:
                print(f"cached - {cached_review.slide_type.value}")
                reviews.append(cached_review)
                continue
            review = checker.check_one(slide_number, text)
            self.save_review_json(reviews_dir, review, source)
            reviews.append(review)
        return reviews

    def run_rewriter_stage(self, slides: list[tuple[int, str]], reviews: list[SlideReview], output_slide_numbers: list[int]) -> list[SlideRewrite]:
        """Run or resume context-independent rewrites for included slides."""
        review_by_number = {review.slide_number: review for review in reviews}
        source_by_number = {slide_number: self.get_rewrite_cache_source(text, review_by_number[slide_number]) for slide_number, text in slides if slide_number in output_slide_numbers}
        cached = self.load_slide_jsons("rewrites", output_slide_numbers, source_by_number)
        if cached is not None:
            print("Loading cached rewrites")
            return cached

        rewrites_dir = os.path.join(self.cache_dir, "rewrites")
        os.makedirs(rewrites_dir, exist_ok=True)
        valid_filenames = {f"slide_{slide_number:03d}.json" for slide_number in output_slide_numbers}
        self.remove_unexpected_files(rewrites_dir, valid_filenames, ".json")
        rewriter = LLMRewriter()
        rewrites: list[SlideRewrite] = []
        output_slides = [(slide_number, text) for slide_number, text in slides if slide_number in output_slide_numbers]

        print("\n" + "=" * 60)
        print("LLM Rewriter")
        print(f"Slides: {len(output_slides)}")
        print("=" * 60)
        for position, (slide_number, text) in enumerate(output_slides, start=1):
            print(f"[{position}/{len(output_slides)}]", end=" ", flush=True)
            cached_slide = self.load_slide_json("rewrites", slide_number, source_by_number[slide_number])
            if cached_slide is not None:
                print("cached rewrite")
                rewrites.append(cached_slide)
                continue
            rewrite = rewriter.rewrite_one(text, review_by_number[slide_number])
            if rewrite is None:
                continue
            self.save_slide_json("rewrites", rewrite, source_by_number[slide_number])
            rewrites.append(rewrite)
        return rewrites

    def run_math_stage(self, slides: list[SlideRewrite]) -> list[SlideRewrite]:
        """Run or resume math conversion only for rewritten output slides."""
        slide_numbers = [slide.slide_number for slide in slides]
        source_by_number = {slide.slide_number: self.get_math_cache_source(slide) for slide in slides}
        cached = self.load_slide_jsons("math", slide_numbers, source_by_number)
        if cached is not None:
            print("Loading cached math-formatted slides")
            return cached

        formatter = MathFormatter()
        math_cache_dir = os.path.join(self.cache_dir, "math")
        replacements_dir = os.path.join(self.cache_dir, "math_replacements")
        os.makedirs(math_cache_dir, exist_ok=True)
        os.makedirs(replacements_dir, exist_ok=True)
        valid_math_filenames = {f"slide_{slide_number:03d}.json" for slide_number in slide_numbers}
        valid_replacement_filenames = {f"slide_{slide_number:03d}_replacements.json" for slide_number in slide_numbers}
        self.remove_unexpected_files(math_cache_dir, valid_math_filenames, ".json")
        self.remove_unexpected_files(replacements_dir, valid_replacement_filenames, ".json")

        print("\n" + "=" * 60)
        print("Math Formatter")
        print(f"Slides: {len(slides)}")
        print("=" * 60)
        updated_slides: list[SlideRewrite] = []
        for slide in slides:
            source = source_by_number[slide.slide_number]
            cached_slide = self.load_slide_json("math", slide.slide_number, source)
            if cached_slide is not None:
                print(f"[slide {slide.slide_number:03d}] cached math")
                updated_slides.append(cached_slide)
                continue
            updated_slide, response = formatter.format_slide(slide)
            self.save_slide_json("math", updated_slide, source)
            replacement_path = os.path.join(replacements_dir, f"slide_{slide.slide_number:03d}_replacements.json")
            if response.replacements:
                self.write_json_atomic(replacement_path, response.model_dump(mode="json"))
            elif os.path.exists(replacement_path):
                os.remove(replacement_path)
            updated_slides.append(updated_slide)
        return updated_slides

    def run_title_stage(self, document: str) -> str:
        """Run or resume stable-index heading analysis."""
        print("\n" + "=" * 60)
        print("Title Editor")
        print("=" * 60)
        title_editor = TitleEditor()
        source = self.get_title_cache_source(document)
        cached_analysis = self.load_json("title_analysis", source)
        if cached_analysis is not None:
            print("Loading cached title analysis")
            analysis = TitleAnalysis(**cached_analysis)
        else:
            analysis = title_editor.identify(document)
            self.save_json("title_analysis", analysis.model_dump(mode="json"), source)
        return title_editor.apply(document, analysis)

    def run_quality_stage(self, source_document: str, document: str) -> str:
        """Run or resume source-aware quality review and its final fixed output."""
        print("\n" + "=" * 60)
        print("Quality Checker")
        print("=" * 60)
        quality_checker = QualityChecker()
        sanitized_document = quality_checker.sanitize_document(document)
        source = self.get_quality_cache_source(source_document, sanitized_document)
        cached_output = self.load_json("quality_output", source)
        if isinstance(cached_output, dict) and isinstance(cached_output.get("document"), str):
            print("Loading cached quality output")
            return cached_output["document"]

        cached_report = self.load_json("quality_report", source)
        if cached_report is not None:
            report = QualityReport(**cached_report)
        else:
            report = quality_checker.check(source_document, sanitized_document)
            self.save_json("quality_report", report.model_dump(mode="json"), source)

        fixed_document = quality_checker.fix(source_document, sanitized_document, report) if report.issues else sanitized_document
        fixed_document = quality_checker.sanitize_document(fixed_document)
        self.save_json("quality_output", {"document": fixed_document}, source)
        return fixed_document
