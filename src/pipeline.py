import json
import os
import time

from .transcriber import Transcriber, CACHE_DIR
from .checker import LLMChecker, WINDOW_SECONDS
from .rewriter import LLMRewriter
from .math_formatter import MathFormatter
from .title_editor import TitleEditor
from .quality_checker import QualityChecker
from .models import SlideReview, SlideRewrite, TitleAnalysis, QualityReport


class Pipeline:
    """Orchestrates the full flow: transcribe, check, rewrite, and save."""

    def __init__(self, pdf_path: str):
        """Initialize with the path to the input PDF."""
        self.pdf_path = pdf_path
        self.pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        self.cache_dir = os.path.join(CACHE_DIR, self.pdf_name)

    def save_json(self, name: str, data):
        """Save a JSON-serializable object to the cache directory."""
        json_path = os.path.join(self.cache_dir, f"{name}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[saved] {name}.json")

    def load_json(self, name: str) -> dict | list | None:
        """Load a cached JSON object, or None if missing."""
        json_path = os.path.join(self.cache_dir, f"{name}.json")
        if os.path.exists(json_path):
            with open(json_path, encoding="utf-8") as f:
                return json.load(f)
        return None

    def save_slide_jsons(self, directory: str, slides: list[SlideRewrite]):
        """Save per-slide JSON artifacts to a subdirectory."""
        dir_path = os.path.join(self.cache_dir, directory)
        os.makedirs(dir_path, exist_ok=True)
        for slide in slides:
            filename = f"slide_{slide.slide_number:03d}.json"
            filepath = os.path.join(dir_path, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(slide.model_dump(), f, indent=2, ensure_ascii=False)
        print(f"[saved] {len(slides)} slides to {directory}/")

    def load_slide_jsons(self, directory: str, expected_count: int) -> list[SlideRewrite] | None:
        """Load per-slide JSON artifacts from a subdirectory, or None if incomplete."""
        dir_path = os.path.join(self.cache_dir, directory)
        if not os.path.exists(dir_path):
            return None
        files = sorted(f for f in os.listdir(dir_path) if f.endswith(".json"))
        if len(files) < expected_count:
            return None
        slides = []
        for filename in files:
            filepath = os.path.join(dir_path, filename)
            with open(filepath, encoding="utf-8") as f:
                slides.append(SlideRewrite(**json.load(f)))
        return slides

    def assemble(self, slides: list[SlideRewrite]) -> str:
        """Concatenate per-slide rewrites into a single markdown document."""
        parts: list[str] = []
        previous_title: str | None = None

        for slide in slides:
            section_parts: list[str] = []

            if slide.is_continuation:
                section_parts.append(slide.text)
            else:
                if slide.title and slide.title != previous_title:
                    section_parts.append(f"## {slide.title}\n")
                    previous_title = slide.title
                section_parts.append(slide.text)

            parts.append("\n".join(section_parts))

        return "\n\n".join(parts)

    def run(self):
        """Execute the full pipeline from PDF to final markdown document."""
        os.makedirs(self.cache_dir, exist_ok=True)

        # Transcribe PDF into per-slide text files
        transcriber = Transcriber(self.pdf_path)
        transcriptions_dir = transcriber.run()

        reviews_dir = os.path.join(self.cache_dir, "reviews")
        os.makedirs(reviews_dir, exist_ok=True)

        slide_filenames = sorted(f for f in os.listdir(transcriptions_dir) if f.startswith("slide_") and f.endswith(".txt"))

        if not slide_filenames:
            print("No slides found in cache.")
            return

        # Load each slide's text content
        slides = []
        for index, filename in enumerate(slide_filenames, start=1):
            filepath = os.path.join(transcriptions_dir, filename)
            with open(filepath, encoding="utf-8") as f:
                slides.append((index, f.read()))

        total = len(slides)

        # Step 1: Checker — per-slide JSON reviews
        reviews = self._run_checker(slides, reviews_dir, total)

        # Count content slides for cache validation
        content_reviews = [r for r in reviews if r.slide_type.value not in ("course_info", "image_description")]
        content_count = len(content_reviews)

        # Step 2: Rewriter — per-slide JSON rewrites
        rewrites = self._run_rewriter(slides, reviews, content_count)

        # Step 3: Math formatter — per-slide JSON with LaTeX replacements
        math_slides = self._run_math_formatter(rewrites, content_count)

        # Step 4: Assemble into markdown
        print("\n" + "=" * 60)
        print("Assembling document")
        print("=" * 60)
        document = self.assemble(math_slides)

        # Step 5: Title editor — JSON analysis, programmatic application
        document = self._run_title_editor(document)

        # Step 6: Quality checker — JSON analysis, targeted fixes
        document = self._run_quality_checker(document)

        # Save the single final markdown document
        output_path = os.path.join(self.cache_dir, f"{self.pdf_name}.md")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(document)

        print("\n" + "=" * 60)
        print(f"[DONE] Final document saved to cache/{self.pdf_name}/{self.pdf_name}.md")
        print("=" * 60)

    def _run_checker(self, slides: list[tuple[int, str]], reviews_dir: str, total: int) -> list[SlideReview]:
        """Run the checker step, loading from cache if available."""
        review_files = sorted(f for f in os.listdir(reviews_dir) if f.endswith("_review.json"))
        if len(review_files) == total:
            print("=" * 60)
            print("Loading cached reviews")
            print("=" * 60)
            reviews = []
            for review_file in review_files:
                review_path = os.path.join(reviews_dir, review_file)
                with open(review_path, encoding="utf-8") as f:
                    reviews.append(SlideReview(**json.load(f)))
            return reviews

        print("=" * 60)
        print("LLM Checker")
        print(f"Slides : {total}")
        print(f"Output : cache/{self.pdf_name}/reviews/")
        print("=" * 60)

        checker = LLMChecker()
        reviews = checker.check_all(slides)

        # Save each review as a JSON file
        print("\n" + "=" * 60)
        print("Writing review files...")
        for review in reviews:
            review_filename = f"slide_{review.slide_number:03d}_review.json"
            review_path = os.path.join(reviews_dir, review_filename)
            with open(review_path, "w", encoding="utf-8") as f:
                json.dump(review.model_dump(), f, indent=2, ensure_ascii=False)
            title_display = f'"{review.title}"' if review.title else "no title"
            actions_display = f"{len(review.actions)} action(s)" if review.actions else "no actions"
            print(f"-> {review_filename} [{review.slide_type.value}] {title_display} {actions_display}")

        print("=" * 60)
        print(f"[DONE] {total} reviews saved in cache/{self.pdf_name}/reviews/")
        print("=" * 60)

        return reviews

    def _run_rewriter(self, slides: list[tuple[int, str]], reviews: list[SlideReview], content_count: int) -> list[SlideRewrite]:
        """Run the rewriter step, loading from cache if available."""
        cached = self.load_slide_jsons("rewrites", content_count)
        if cached:
            print("=" * 60)
            print("Loading cached rewrites")
            print("=" * 60)
            return cached

        # Wait after checker step
        print(f"\nWaiting {WINDOW_SECONDS}s before rewriting step...")
        time.sleep(WINDOW_SECONDS)

        print("\n" + "=" * 60)
        print("LLM Rewriter")
        print(f"Slides : {len(slides)}")
        print("=" * 60)

        rewriter = LLMRewriter()
        rewrites = rewriter.rewrite_all(slides, reviews)
        self.save_slide_jsons("rewrites", rewrites)
        return rewrites

    def _run_math_formatter(self, slides: list[SlideRewrite], content_count: int) -> list[SlideRewrite]:
        """Run math formatting per slide, loading from cache if available."""
        cached = self.load_slide_jsons("math", content_count)
        if cached:
            print("=" * 60)
            print("Loading cached math formatted slides")
            print("=" * 60)
            return cached

        print(f"\nWaiting {WINDOW_SECONDS}s before math formatting step...")
        time.sleep(WINDOW_SECONDS)

        print("\n" + "=" * 60)
        print("Math Formatter")
        print(f"Slides : {len(slides)}")
        print("=" * 60)

        formatter = MathFormatter()
        updated_slides, responses = formatter.format_all(slides)

        # Save per-slide math JSON artifacts
        self.save_slide_jsons("math", updated_slides)

        # Save the replacement mappings separately for reference
        math_dir = os.path.join(self.cache_dir, "math_replacements")
        os.makedirs(math_dir, exist_ok=True)
        for slide, response in zip(updated_slides, responses):
            if response.replacements:
                filename = f"slide_{slide.slide_number:03d}_replacements.json"
                filepath = os.path.join(math_dir, filename)
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(response.model_dump(), f, indent=2, ensure_ascii=False)

        return updated_slides

    def _run_title_editor(self, document: str) -> str:
        """Run title editing: identify heading changes as JSON, apply programmatically."""
        print(f"\nWaiting {WINDOW_SECONDS}s before title editing step...")
        time.sleep(WINDOW_SECONDS)

        print("\n" + "=" * 60)
        print("Title Editor")
        print("=" * 60)

        title_editor = TitleEditor()

        # Step 1: identify changes (check cache)
        cached_analysis = self.load_json("title_analysis")
        if cached_analysis:
            print("Loading cached title analysis")
            analysis = TitleAnalysis(**cached_analysis)
        else:
            analysis = title_editor.edit(document)
            self.save_json("title_analysis", analysis.model_dump())

        # Step 2: apply changes programmatically
        document = title_editor.apply(document, analysis)
        return document

    def _run_quality_checker(self, document: str) -> str:
        """Run quality check: identify issues as JSON, fix targeted fragments."""
        print(f"\nWaiting {WINDOW_SECONDS}s before quality check step...")
        time.sleep(WINDOW_SECONDS)

        print("\n" + "=" * 60)
        print("Quality Checker")
        print("=" * 60)

        quality_checker = QualityChecker()

        # Step 1: identify issues (check cache)
        cached_report = self.load_json("quality_report")
        if cached_report:
            print("Loading cached quality report")
            report = QualityReport(**cached_report)
        else:
            report = quality_checker.check(document)
            self.save_json("quality_report", report.model_dump())

        # Step 2: fix identified issues
        if report.issues:
            document = quality_checker.fix(document, report)

        return document
