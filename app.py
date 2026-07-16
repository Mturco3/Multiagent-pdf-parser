"""Provide the Streamlit interface for generating notes from uploaded PDFs.

The module manages session state and contains helpers for running the pipeline
in a temporary workspace and rendering the generated Markdown document.
"""

import os
import sys
import tempfile
from collections.abc import Callable

import streamlit as st
from dotenv import load_dotenv

from src.utilities.rate_limit import DailyQuotaExceededError

ProgressCallback = Callable[[str, int | None, int | None, str], None]

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def initialize_session_state() -> None:
    """Initialize generated output fields that must survive Streamlit reruns."""
    st.session_state.setdefault("generated_document", None)
    st.session_state.setdefault("generated_filename", None)


def generate_notes(
    uploaded_file,
    progress_callback: ProgressCallback | None = None
) -> tuple[str, str]:
    """Run one upload in an isolated temporary workspace and return its Markdown."""
    safe_filename = os.path.basename(uploaded_file.name) or "lecture.pdf"
    output_filename = f"{os.path.splitext(safe_filename)[0]}.md"

    with tempfile.TemporaryDirectory(prefix="lecture_notes_") as temp_dir:
        temp_pdf = os.path.join(temp_dir, safe_filename)
        with open(temp_pdf, "wb") as file_handle:
            file_handle.write(uploaded_file.getbuffer())

        cache_root = os.path.join(temp_dir, "cache")
        # Import after loading .env because model settings are module constants.
        from src.pipeline import Pipeline

        pipeline = Pipeline(
            temp_pdf,
            clear_cache=False,
            cache_root=cache_root,
            progress_callback=progress_callback
        )
        document = pipeline.run()

    return document, output_filename


def render_document(document: str, filename: str) -> None:
    """Render preview, raw Markdown, and a persistent download action."""
    st.success("Pipeline complete!")
    preview_tab, raw_tab = st.tabs(["Preview", "Raw Markdown"])
    with preview_tab:
        st.markdown(document)
    with raw_tab:
        st.code(document, language="markdown")
    st.download_button(
        label="Download Markdown",
        data=document,
        file_name=filename,
        mime="text/markdown"
    )


st.set_page_config(page_title="Lecture Notes Generator", layout="wide")
initialize_session_state()
st.title("Lecture Notes Generator")
st.caption(
    "Upload lecture slides to generate Markdown notes. The PDF content is sent "
    "to the configured model provider and temporary upload files are deleted "
    "after processing."
)

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file is not None and st.button("Generate Notes", type="primary"):
    progress_status = st.status("Preparing the PDF...", expanded=True)
    progress_bar = st.progress(0, text="Starting pipeline")
    progress_detail = st.empty()

    def show_progress(
        stage: str,
        current: int | None,
        total: int | None,
        detail: str
    ) -> None:
        """Render pipeline events as a live status, progress bar, and activity message."""
        progress_status.update(label=f"{stage}...", state="running", expanded=True)
        if current is not None and total:
            percent = max(0, min(100, round(current / total * 100)))
            progress_bar.progress(percent, text=f"{stage}: {current}/{total}")
        else:
            progress_bar.progress(0, text=stage)
        if detail:
            progress_detail.caption(detail)

    try:
        generated_document, generated_filename = generate_notes(
            uploaded_file,
            progress_callback=show_progress
        )
        st.session_state.generated_document = generated_document
        st.session_state.generated_filename = generated_filename
        progress_bar.progress(100, text="Pipeline complete")
        progress_detail.caption("The notes are ready below.")
        progress_status.update(
            label="Notes generated",
            state="complete",
            expanded=False
        )
    except DailyQuotaExceededError as error:
        progress_status.update(
            label="Model quota exhausted",
            state="error",
            expanded=True
        )
        st.error(f"Model quota exhausted: {error}")
    except Exception as error:
        progress_status.update(
            label="Pipeline failed",
            state="error",
            expanded=True
        )
        st.exception(error)

if st.session_state.generated_document and st.session_state.generated_filename:
    st.divider()
    render_document(st.session_state.generated_document, st.session_state.generated_filename)
