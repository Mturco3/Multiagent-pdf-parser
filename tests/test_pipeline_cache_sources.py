import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.models import SlideRewrite, SlideType
from src.pipeline import REWRITER_SYSTEM_PROMPT, Pipeline


class PipelineCacheSourceTests(unittest.TestCase):
    """Verify content-addressed cache provenance and document assembly."""

    def setUp(self):
        """Create an isolated cache directory for each test."""
        self.temp_directory = tempfile.TemporaryDirectory()
        self.pipeline = Pipeline("lecture.pdf", cache_root=self.temp_directory.name)
        self.pipeline.cache_dir = self.temp_directory.name

    def tearDown(self):
        """Remove all temporary cache artifacts."""
        self.temp_directory.cleanup()

    def test_rewrite_cache_source_changes_when_prompt_changes(self):
        """Changing direct extraction instructions must invalidate slide artifacts."""
        baseline = self.pipeline.get_rewrite_cache_source("Slide body")
        with patch("src.pipeline.REWRITER_SYSTEM_PROMPT", REWRITER_SYSTEM_PROMPT + "\nAdditional rule."):
            changed = self.pipeline.get_rewrite_cache_source("Slide body")
        self.assertNotEqual(baseline, changed)

    def test_rewrite_cache_source_changes_when_model_changes(self):
        """Changing the note extraction model must invalidate slide artifacts."""
        baseline = self.pipeline.get_rewrite_cache_source("Slide body")
        with patch("src.pipeline.REWRITER_MODEL", "google:different-model"):
            changed = self.pipeline.get_rewrite_cache_source("Slide body")
        self.assertNotEqual(baseline, changed)

    def test_slide_cache_rejects_stale_source_hash(self):
        """A per-slide artifact must not survive a source change."""
        slide = SlideRewrite(slide_number=1, slide_type=SlideType.CONTENT, title="Topic", is_continuation=False, text="Body text.", rewrite_mode="direct_extraction_v4")
        source_v1 = self.pipeline.get_cache_source("rewrite", "v1")
        source_v2 = self.pipeline.get_cache_source("rewrite", "v2")
        self.pipeline.save_slide_json("rewrites", slide, source_v1)
        cached_slide = self.pipeline.load_slide_json("rewrites", 1, source_v1)
        self.assertIsNotNone(cached_slide)
        self.assertEqual(cached_slide.text, "Body text.")
        self.assertIsNone(self.pipeline.load_slide_json("rewrites", 1, source_v2))

    def test_document_json_cache_rejects_stale_source_hash(self):
        """A document artifact must not survive a source change."""
        source_v1 = self.pipeline.get_cache_source("title", "v1")
        source_v2 = self.pipeline.get_cache_source("title", "v2")
        self.pipeline.save_json("title_analysis", {"changes": []}, source_v1)
        self.assertEqual(self.pipeline.load_json("title_analysis", source_v1), {"changes": []})
        self.assertIsNone(self.pipeline.load_json("title_analysis", source_v2))

    def test_corrupt_json_is_discarded(self):
        """An interrupted JSON write must behave like a cache miss."""
        filepath = Path(self.temp_directory.name) / "broken.json"
        filepath.write_text("{", encoding="utf-8")
        self.assertIsNone(self.pipeline.load_json_file(str(filepath)))
        self.assertFalse(filepath.exists())

    def test_heading_only_introduction_is_assembled(self):
        """A section introduction must remain visible without body text."""
        slide = SlideRewrite(slide_number=1, slide_type=SlideType.INTRODUCTION, title="Methods", is_continuation=False, text="", rewrite_mode="direct_extraction_v4")
        self.assertEqual(self.pipeline.assemble([slide]), "## Methods")

    def test_image_slide_is_included(self):
        """A figure caption slide must not be dropped from the output set."""
        slide = SlideRewrite(slide_number=4, slide_type=SlideType.IMAGE_DESCRIPTION, title="Architecture", is_continuation=False, text="Diagram")
        self.assertEqual(self.pipeline.get_output_slide_numbers([slide]), [4])

    def test_same_filename_uses_distinct_content_cache(self):
        """Different PDFs with the same basename must receive different cache namespaces."""
        first_dir = Path(self.temp_directory.name) / "first"
        second_dir = Path(self.temp_directory.name) / "second"
        first_dir.mkdir()
        second_dir.mkdir()
        first_pdf = first_dir / "lecture.pdf"
        second_pdf = second_dir / "lecture.pdf"
        first_pdf.write_bytes(b"first")
        second_pdf.write_bytes(b"second")
        first_pipeline = Pipeline(str(first_pdf), cache_root=self.temp_directory.name)
        second_pipeline = Pipeline(str(second_pdf), cache_root=self.temp_directory.name)
        first_pipeline.configure_cache_identity()
        second_pipeline.configure_cache_identity()
        self.assertNotEqual(first_pipeline.cache_dir, second_pipeline.cache_dir)


if __name__ == "__main__":
    unittest.main()
