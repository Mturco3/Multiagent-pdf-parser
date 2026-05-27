import json
import os
import time

from .transcriber import Transcriber, CACHE_DIR
from .checker import LLMChecker, WINDOW_SECONDS
from .rewriter import LLMRewriter
from .math_formatter import MathFormatter
from .title_editor import TitleEditor
from .quality_checker import QualityChecker
from .models import SlideReview


class Pipeline:
    """Orchestrates the full flow: transcribe, check, rewrite, and save."""

    def __init__(self, pdf_path: str):
        """Initialize with the path to the input PDF."""
        self.pdf_path = pdf_path
        self.pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        self.cache_dir = os.path.join(CACHE_DIR, self.pdf_name)

    def save_step(self, step_name: str, document: str):
        """Save intermediate document after a pipeline step."""
        step_path = os.path.join(self.cache_dir, f"{step_name}.md")
        with open(step_path, "w", encoding="utf-8") as f:
            f.write(document)
        print(f"[saved] {step_name}.md")

    def load_step(self, step_name: str) -> str | None:
        """Load a previously saved intermediate document, or None if missing."""
        step_path = os.path.join(self.cache_dir, f"{step_name}.md")
        if os.path.exists(step_path):
            with open(step_path, encoding="utf-8") as f:
                return f.read()
        return None

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

        # Try to resume from cached rewriter output
        document = self.load_step("rewritten")
        if document:
            print("=" * 60)
            print("Resuming from cached rewriter output")
            print("=" * 60)
        else:
            # Check if reviews are already cached
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
            else:
                print("=" * 60)
                print("LLM Checker")
                print(f"Slides : {total}")
                print(f"Output : cache/{self.pdf_name}/reviews/")
                print("=" * 60)

                # Run LLM checker on all slides
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

                # Wait between steps
                print(f"\nWaiting {WINDOW_SECONDS}s before rewriting step...")
                time.sleep(WINDOW_SECONDS)

            # Rewrite slides and concatenate into final document
            print("\n" + "=" * 60)
            print("LLM Rewriter")
            print(f"Slides : {total}")
            print("=" * 60)

            rewriter = LLMRewriter()
            document = rewriter.rewrite_all(slides, reviews)
            self.save_step("rewritten", document)

        # Wait before math formatting step
        print(f"\nWaiting {WINDOW_SECONDS}s before math formatting step...")
        time.sleep(WINDOW_SECONDS)

        # Format math expressions as LaTeX
        print("\n" + "=" * 60)
        print("Math Formatter")
        print("=" * 60)
        math_formatter = MathFormatter()
        document = math_formatter.format_document(document)
        self.save_step("math_formatted", document)

        # Wait before title editing step
        print(f"\nWaiting {WINDOW_SECONDS}s before title editing step...")
        time.sleep(WINDOW_SECONDS)

        # Edit title hierarchy
        print("\n" + "=" * 60)
        print("Title Editor")
        print("=" * 60)
        title_editor = TitleEditor()
        document = title_editor.edit(document)

        # Wait before quality check step
        print(f"\nWaiting {WINDOW_SECONDS}s before quality check step...")
        time.sleep(WINDOW_SECONDS)

        # Run quality check on the final document
        print("\n" + "=" * 60)
        print("Quality Checker")
        print("=" * 60)
        quality_checker = QualityChecker()
        report = quality_checker.check(document)

        # Save the final markdown document
        output_path = os.path.join(self.cache_dir, f"{self.pdf_name}.md")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(document)

        # Save quality report
        quality_path = os.path.join(self.cache_dir, "quality_report.json")
        with open(quality_path, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(), f, indent=2, ensure_ascii=False)

        print("\n" + "=" * 60)
        print(f"[DONE] Final document saved to cache/{self.pdf_name}/{self.pdf_name}.md")
        print(f"[DONE] Quality report saved to cache/{self.pdf_name}/quality_report.json")
        print("=" * 60)
