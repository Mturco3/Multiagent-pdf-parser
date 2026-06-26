import os
import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from src.models import SlideRewrite
from src.pipeline import CHECKER_SYSTEM_PROMPT, Pipeline


class PipelineCacheSourceTests(unittest.TestCase):
    def setUp(self):
        test_tmp_root = Path.cwd() / ".test_tmp"
        test_tmp_root.mkdir(exist_ok=True)
        self.temp_dir = test_tmp_root / f"pipeline_cache_{uuid.uuid4().hex}"
        self.temp_dir.mkdir()
        self.pipeline = Pipeline("lecture.pdf")
        self.pipeline.cache_dir = str(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_review_cache_source_changes_when_prompt_changes(self):
        baseline = self.pipeline.get_review_cache_source("Slide body")

        with patch("src.pipeline.CHECKER_SYSTEM_PROMPT", CHECKER_SYSTEM_PROMPT + "\nAdditional rule."):
            changed = self.pipeline.get_review_cache_source("Slide body")

        self.assertNotEqual(baseline, changed)

    def test_slide_cache_rejects_stale_source_hash(self):
        slide = SlideRewrite(
            slide_number=1,
            slide_type="content",
            title="Topic",
            is_continuation=False,
            text="Body text.",
        )
        source_v1 = self.pipeline.get_cache_source("rewrite", "v1")
        source_v2 = self.pipeline.get_cache_source("rewrite", "v2")

        self.pipeline.save_slide_json("rewrites", slide, source_v1)

        cached_slide = self.pipeline.load_slide_json("rewrites", 1, source_v1)
        self.assertIsNotNone(cached_slide)
        self.assertEqual(cached_slide.text, "Body text.")

        stale_slide = self.pipeline.load_slide_json("rewrites", 1, source_v2)
        self.assertIsNone(stale_slide)
        self.assertFalse(os.path.exists(os.path.join(self.temp_dir, "rewrites", "slide_001.json")))

    def test_document_json_cache_rejects_stale_source_hash(self):
        source_v1 = self.pipeline.get_cache_source("title", "v1")
        source_v2 = self.pipeline.get_cache_source("title", "v2")

        self.pipeline.save_json("title_analysis", {"changes": []}, source_v1)

        self.assertEqual(self.pipeline.load_json("title_analysis", source_v1), {"changes": []})
        self.assertIsNone(self.pipeline.load_json("title_analysis", source_v2))


if __name__ == "__main__":
    unittest.main()
