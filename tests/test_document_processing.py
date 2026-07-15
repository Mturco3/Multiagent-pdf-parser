import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import fitz
from pydantic_ai.exceptions import ModelHTTPError

from src.math_formatter import MathFormatter
from src.models import HeadingAction, HeadingChange, IssueType, QualityIssue, QualityReport, SlideRewriteResponse, SlideType, TitleAnalysis
from src.pipeline import Pipeline
from src.quality_checker import QualityChecker
from src.rewriter import LLMRewriter
from src.title_editor import TitleEditor
from src.transcriber import Transcriber
from src.utilities.model_config import DEFAULT_MODEL, FALLBACK_MODEL, FAST_NOTE_MODEL, get_default_model_limits, get_default_note_model
from src.utilities.normalizer import normalize
from src.utilities.model_retry import DEFAULT_REQUEST_TIMEOUT_SECONDS, get_agent_model_settings, get_cached_agent, run_with_retry
from src.utilities.rate_limit import RequestPacer


class DocumentProcessingTests(unittest.TestCase):
    """Verify deterministic extraction and document transformations."""

    def setUp(self):
        """Create an isolated filesystem workspace."""
        self.temp_directory = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_directory.name)

    def tearDown(self):
        """Remove temporary PDFs and cache state."""
        self.temp_directory.cleanup()

    def create_test_pdf(self) -> Path:
        """Create a small deck with repeated headers and distinct titles."""
        pdf_path = self.temp_path / "deck.pdf"
        document = fitz.open()
        for slide_number in range(1, 4):
            page = document.new_page()
            page.insert_text((40, 30), "Repeated Course Header", fontsize=9)
            page.insert_text((90, 100), f"Topic {slide_number}", fontsize=24)
            page.insert_text((90, 150), f"Body content {slide_number}", fontsize=12)
            page.insert_text((500, 800), f"{slide_number}/3", fontsize=8)
        document.save(pdf_path)
        document.close()
        return pdf_path

    def test_transcriber_removes_positional_boilerplate(self):
        """Repeated headers and page numbers must not enter slide text."""
        pdf_path = self.create_test_pdf()
        cache_dir = self.temp_path / "cache"
        output_dir = Transcriber(str(pdf_path), str(cache_dir)).run()
        first_slide = Path(output_dir, "slide_001.txt").read_text(encoding="utf-8")
        self.assertNotIn("Repeated Course Header", first_slide)
        self.assertNotIn("1/3", first_slide)
        self.assertTrue(first_slide.startswith("Topic 1"))
        self.assertIn("Body content 1", first_slide)

    def test_pipeline_runs_end_to_end_with_mocked_model_boundaries(self):
        """All deterministic stages must compose into a final cached document."""
        pdf_path = self.create_test_pdf()
        cache_root = self.temp_path / "pipeline_cache"
        progress_events = []

        extracted_slides = [
            SlideRewriteResponse(slide_type=SlideType.CONTENT, title=f"Topic {slide_number}", is_continuation=False, text=f"Body content {slide_number}")
            for slide_number in range(1, 4)
        ]

        with patch("src.pipeline.LLMRewriter.run_rewriter_request", side_effect=extracted_slides):
            with patch("src.pipeline.TitleEditor.identify", return_value=TitleAnalysis(changes=[])):
                with patch("src.pipeline.QualityChecker.check", return_value=QualityReport(issues=[])):
                    document = Pipeline(str(pdf_path), cache_root=str(cache_root), progress_callback=lambda *event: progress_events.append(event)).run()

        self.assertIn("## Topic 1", document)
        self.assertIn("Body content 3", document)
        output_files = list(cache_root.rglob("deck.md"))
        self.assertEqual(len(output_files), 1)
        stages = [event[0] for event in progress_events]
        self.assertNotIn("Reviewing slide structure", stages)
        self.assertIn("Extracting slide notes", stages)
        self.assertIn("Formatting mathematics", stages)
        self.assertIn("Checking final quality", stages)
        self.assertEqual(stages[-1], "Complete")

    def test_control_character_bullet_is_normalized(self):
        """The sample deck bullet marker must become Markdown."""
        self.assertEqual(normalize("\u000f first\n\u000f second"), "- first\n- second")

    def test_quality_sanitizer_preserves_real_lists(self):
        """Final deterministic cleanup must not flatten Markdown lists."""
        document = "# Steps\n\n- Install\n- Configure\n- Run"
        sanitized = QualityChecker().sanitize_document(document)
        self.assertIn("- Install\n- Configure\n- Run", sanitized)

    def test_repetition_fix_removes_last_duplicate(self):
        """A repetition fix must retain the first occurrence and remove the duplicate."""
        document = "Repeated sentence.\n\nMiddle.\n\nRepeated sentence."
        issue = QualityIssue(issue_type=IssueType.REPETITION, problematic_text="Repeated sentence.", explanation="Duplicate")
        fixed = QualityChecker().fix(document, document, QualityReport(issues=[issue]))
        self.assertTrue(fixed.startswith("Repeated sentence."))
        self.assertEqual(fixed.count("Repeated sentence."), 1)

    def test_title_editor_changes_only_indexed_heading(self):
        """Heading edits must not replace heading-like body substrings."""
        document = "Mention ## Topic here.\n\n## Topic\n\nBody"
        change = HeadingChange(heading_index=1, original_heading="## Topic", action=HeadingAction.KEEP, new_level=3, new_text=None)
        updated = TitleEditor().apply(document, TitleAnalysis(changes=[change]))
        self.assertIn("Mention ## Topic here.", updated)
        self.assertIn("### Topic", updated)

    def test_rewriter_rejects_low_source_coverage(self):
        """Coverage validation must identify a rewrite that loses source content."""
        coverage = LLMRewriter().calculate_source_coverage("alpha beta gamma delta", "alpha")
        self.assertLess(coverage, 0.65)

    def test_rewriter_extracts_notes_without_checker_review(self):
        """One direct response must provide both metadata and note text."""
        response = SlideRewriteResponse(slide_type=SlideType.CONTENT, title="Topic", is_continuation=False, text="Alpha beta gamma delta.")
        rewriter = LLMRewriter()
        with patch.object(rewriter, "run_rewriter_request", return_value=response) as request:
            slide = rewriter.rewrite_one(7, "Topic\nAlpha beta gamma delta.")
        self.assertEqual(slide.slide_number, 7)
        self.assertEqual(slide.title, "Topic")
        self.assertEqual(slide.text, "Alpha beta gamma delta.")
        self.assertEqual(slide.rewrite_mode, "direct_extraction_v4")
        request.assert_called_once()

    def test_math_prefilter_skips_plain_prose(self):
        """Plain prose should not consume a math-model request."""
        formatter = MathFormatter()
        self.assertFalse(formatter.has_math_candidate("A plain lecture sentence."))
        self.assertTrue(formatter.has_math_candidate("x = y + 2"))

    def test_rate_limiter_recovers_stale_lock(self):
        """An abandoned lock must not block future requests forever."""
        pacer = RequestPacer(60)
        pacer.lock_path = str(self.temp_path / "rate.lock")
        Path(pacer.lock_path).write_text("stale", encoding="utf-8")
        stale_time = time.time() - 300
        os.utime(pacer.lock_path, (stale_time, stale_time))
        pacer.acquire_lock()
        self.assertTrue(Path(pacer.lock_path).exists())
        pacer.release_lock()
        self.assertFalse(Path(pacer.lock_path).exists())

    def test_repeated_503_switches_to_fallback_model(self):
        """A final primary 503 must retry the request with the fallback model."""
        pacer = RequestPacer(60)

        def runner(model_name: str, prompt: str) -> str:
            """Provide the callable shape expected by retry orchestration."""
            return f"{model_name}:{prompt}"

        primary_error = ModelHTTPError(503, "primary-model", {})
        with patch("src.utilities.model_retry.run_model_attempts", side_effect=[primary_error, "fallback-output"]) as attempts:
            output = run_with_retry(pacer, "test", "primary-model", 10, 100, runner, "prompt", 60)
        self.assertEqual(output, "fallback-output")
        self.assertEqual(attempts.call_args_list[1].args[2], FALLBACK_MODEL)

    def test_model_agent_has_bounded_request_timeout(self):
        """Every provider request must have an explicit timeout."""
        with patch("src.utilities.model_retry.Agent") as agent_class:
            get_cached_agent({}, "test-model", str, "instructions")
        settings = agent_class.call_args.kwargs["model_settings"]
        self.assertEqual(settings["timeout"], DEFAULT_REQUEST_TIMEOUT_SECONDS)

    def test_groq_note_model_is_selected_only_when_configured(self):
        """The fast provider must be automatic but retain a Google-only fallback."""
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}, clear=True):
            self.assertEqual(get_default_note_model(), FAST_NOTE_MODEL)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(get_default_note_model(), DEFAULT_MODEL)

    def test_groq_note_model_uses_free_limits_and_low_reasoning(self):
        """Local pacing and reasoning effort must match the fast note workload."""
        self.assertEqual(get_default_model_limits(FAST_NOTE_MODEL), (30, 1000))
        self.assertEqual(get_agent_model_settings(FAST_NOTE_MODEL)["thinking"], "low")


if __name__ == "__main__":
    unittest.main()
