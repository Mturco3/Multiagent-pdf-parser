import io
import os
import sys
import tempfile
import contextlib

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

st.set_page_config(page_title="Lecture Notes Generator", layout="wide")
st.title("Lecture Notes Generator")
st.caption("Upload PDF lecture slides and convert them into polished Markdown notes.")


uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
clear_cache = st.checkbox("Clear cache before running", value=False)

if uploaded_file is not None:
    if st.button("Generate Notes", type="primary"):
        # Save uploaded file to a temp location
        temp_dir = tempfile.mkdtemp()
        temp_pdf = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_pdf, "wb") as f:
            f.write(uploaded_file.getbuffer())

        log_area = st.empty()
        progress_bar = st.progress(0, text="Starting pipeline...")

        from src.pipeline import Pipeline

        pipeline = Pipeline(temp_pdf, clear_cache=clear_cache)

        # Capture stdout to show pipeline logs
        log_output = io.StringIO()

        stages = [
            ("validate_input_pdf", "Validating PDF...", 0.02),
            ("reset_pdf_cache", "Clearing cache...", 0.05),
        ]

        try:
            pipeline.validate_input_pdf()
            progress_bar.progress(0.02, text="PDF validated")

            if clear_cache:
                pipeline.reset_pdf_cache()

            os.makedirs(pipeline.cache_dir, exist_ok=True)

            # Transcribe
            progress_bar.progress(0.05, text="Transcribing slides...")
            from src.transcriber import Transcriber
            transcriber = Transcriber(temp_pdf)
            with contextlib.redirect_stdout(log_output):
                transcriptions_dir = transcriber.run()

            # Load slides
            slide_filenames = sorted([
                name for name in os.listdir(transcriptions_dir)
                if name.startswith("slide_") and name.endswith(".txt")
            ])

            if not slide_filenames:
                st.error("No slides found in the PDF.")
                st.stop()

            slides = []
            for filename in slide_filenames:
                slide_number = pipeline.get_slide_number(filename)
                filepath = os.path.join(transcriptions_dir, filename)
                with open(filepath, encoding="utf-8") as fh:
                    slides.append((slide_number, fh.read()))

            total = len(slides)
            st.info(f"Found {total} slides")

            # Checker
            progress_bar.progress(0.15, text=f"Running LLM Checker on {total} slides...")
            reviews_dir = os.path.join(pipeline.cache_dir, "reviews")
            os.makedirs(reviews_dir, exist_ok=True)
            with contextlib.redirect_stdout(log_output):
                reviews = pipeline._run_checker(slides, reviews_dir, total)

            output_slide_numbers = pipeline.get_output_slide_numbers(reviews)
            content_count = len(output_slide_numbers)
            skipped_count = total - content_count
            st.info(f"Checker done: {content_count} content slides, {skipped_count} skipped")

            # Rewriter
            progress_bar.progress(0.40, text="Running LLM Rewriter...")
            with contextlib.redirect_stdout(log_output):
                rewrites = pipeline._run_rewriter(slides, reviews, output_slide_numbers)

            # Math formatter
            progress_bar.progress(0.60, text="Formatting math expressions...")
            with contextlib.redirect_stdout(log_output):
                math_slides = pipeline._run_math_formatter(rewrites, output_slide_numbers)

            # Assemble
            progress_bar.progress(0.75, text="Assembling document...")
            document = pipeline.assemble(math_slides)

            # Title editor
            progress_bar.progress(0.80, text="Editing titles...")
            with contextlib.redirect_stdout(log_output):
                document = pipeline._run_title_editor(document)

            # Quality checker
            progress_bar.progress(0.90, text="Running quality check...")
            with contextlib.redirect_stdout(log_output):
                document = pipeline._run_quality_checker(document)

            progress_bar.progress(1.0, text="Done!")

            # Display result
            st.divider()
            tab_preview, tab_raw, tab_log = st.tabs(["Preview", "Raw Markdown", "Pipeline Log"])

            with tab_preview:
                st.markdown(document)

            with tab_raw:
                st.code(document, language="markdown")

            with tab_log:
                st.code(log_output.getvalue(), language="text")

            # Download button
            st.download_button(
                label="Download Markdown",
                data=document,
                file_name=f"{os.path.splitext(uploaded_file.name)[0]}.md",
                mime="text/markdown"
            )

        except Exception as error:
            progress_bar.empty()
            st.error(f"Pipeline failed: {error}")
            with st.expander("Pipeline log"):
                st.code(log_output.getvalue(), language="text")
