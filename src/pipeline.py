import json
import os

from .transcriber import Transcriber, CACHE_DIR
from .checker import LLMChecker


class Pipeline:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path

    def run(self):
        # Step 1: transcribe PDF to per-slide .txt files
        transcriber = Transcriber(self.pdf_path)
        transcriptions_dir = transcriber.run()

        pdf_name = os.path.splitext(os.path.basename(self.pdf_path))[0]
        reviews_dir = os.path.join(CACHE_DIR, pdf_name, "reviews")
        os.makedirs(reviews_dir, exist_ok=True)

        slide_files = sorted(
            f for f in os.listdir(transcriptions_dir)
            if f.startswith("slide_") and f.endswith(".txt")
        )

        if not slide_files:
            print("No slides found in cache.")
            return

        # Load slide texts
        slides = []
        for i, filename in enumerate(slide_files, start=1):
            txt_path = os.path.join(transcriptions_dir, filename)
            with open(txt_path, encoding="utf-8") as f:
                slides.append((i, f.read()))

        total = len(slides)
        print("=" * 60)
        print("LLM Checker")
        print(f"Model  : gemma-4-31b-it")
        print(f"Slides : {total}")
        print(f"Output : cache/{pdf_name}/reviews/")
        print("=" * 60)

        # Step 2: LLM check all slides (async batched)
        checker = LLMChecker()
        reviews = checker.check_all(slides)

        # Step 3: write review JSONs
        print("\n" + "=" * 60)
        print("Writing review files...")
        for review in reviews:
            filename = f"slide_{review.slide_number:03d}_review.json"
            review_path = os.path.join(reviews_dir, filename)
            with open(review_path, "w", encoding="utf-8") as f:
                json.dump(review.model_dump(), f, indent=2, ensure_ascii=False)
            title_str = f'"{review.title}"' if review.title else "no title"
            actions_str = f"{len(review.actions)} action(s)" if review.actions else "no actions"
            print(f"  -> {filename}  [{review.slide_type.value}] {title_str}  {actions_str}")

        print("=" * 60)
        print(f"[DONE] {total} reviews saved in cache/{pdf_name}/reviews/")
        print("=" * 60)
