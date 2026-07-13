import os
import sys
import tempfile

import streamlit as st
from dotenv import load_dotenv

from src.utilities.rate_limit import DailyQuotaExceededError

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def initialize_session_state():
    """Initialize generated output fields that must survive Streamlit reruns."""
    if "generated_document" not in st.session_state:
        st.session_state.generated_document = None
    if "generated_filename" not in st.session_state:
        st.session_state.generated_filename = None


def generate_notes(uploaded_file) -> tuple[str, str]:
    """Run one upload in an isolated temporary workspace and return its Markdown."""
    safe_filename = os.path.basename(uploaded_file.name) or "lecture.pdf"
    output_filename = f"{os.path.splitext(safe_filename)[0]}.md"
    with tempfile.TemporaryDirectory(prefix="lecture_notes_") as temp_dir:
        temp_pdf = os.path.join(temp_dir, safe_filename)
        with open(temp_pdf, "wb") as file_handle:
            file_handle.write(uploaded_file.getbuffer())
        cache_root = os.path.join(temp_dir, "cache")
        from src.pipeline import Pipeline
        pipeline = Pipeline(temp_pdf, clear_cache=False, cache_root=cache_root)
        document = pipeline.run()
    return document, output_filename


def render_document(document: str, filename: str):
    """Render preview, raw Markdown, and a persistent download action."""
    st.success("Pipeline complete!")
    preview_tab, raw_tab = st.tabs(["Preview", "Raw Markdown"])
    with preview_tab:
        st.markdown(document)
    with raw_tab:
        st.code(document, language="markdown")
    st.download_button(label="Download Markdown", data=document, file_name=filename, mime="text/markdown")


st.set_page_config(page_title="Lecture Notes Generator", layout="wide")
initialize_session_state()
st.title("Lecture Notes Generator")
st.caption("Upload lecture slides to generate Markdown notes. The PDF is sent to the configured Google model and temporary upload files are deleted after processing.")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file is not None and st.button("Generate Notes", type="primary"):
    try:
        with st.spinner("Generating notes..."):
            generated_document, generated_filename = generate_notes(uploaded_file)
        st.session_state.generated_document = generated_document
        st.session_state.generated_filename = generated_filename
    except DailyQuotaExceededError as error:
        st.error(f"Model quota exhausted: {error}")
    except Exception as error:
        st.exception(error)

if st.session_state.generated_document and st.session_state.generated_filename:
    st.divider()
    render_document(st.session_state.generated_document, st.session_state.generated_filename)
