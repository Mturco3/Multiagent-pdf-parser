import io
import os
import sys
import tempfile
import threading
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


class StreamlitLogStream:
    """Writable stream that captures text and updates a Streamlit container live."""

    def __init__(self, container):
        """Initialize with a Streamlit container for live log display."""
        self.container = container
        self.buffer = io.StringIO()

    def write(self, text):
        """Append text to the buffer and refresh the displayed log."""
        self.buffer.write(text)
        self.container.code(self.buffer.getvalue(), language="text")

    def flush(self):
        """No-op flush to satisfy the stream interface."""
        pass

    def getvalue(self):
        """Return the full captured log text."""
        return self.buffer.getvalue()


uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
clear_cache = st.checkbox("Clear cache before running", value=False)

if uploaded_file is not None:
    if st.button("Generate Notes", type="primary"):
        # Save uploaded file to a temp location
        temp_dir = tempfile.mkdtemp()
        temp_pdf = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_pdf, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Live log area
        log_expander = st.expander("Pipeline Log", expanded=True)
        log_container = log_expander.empty()
        log_stream = StreamlitLogStream(log_container)

        result = {}

        def thread_target():
            """Execute the pipeline on a dedicated thread with stdout redirected."""
            try:
                with contextlib.redirect_stdout(log_stream):
                    from src.pipeline import Pipeline
                    pipeline = Pipeline(temp_pdf, clear_cache=clear_cache)
                    pipeline.run()
                    result["pipeline"] = pipeline
            except Exception as error:
                result["error"] = error

        with st.spinner("Running pipeline (this takes several minutes)..."):
            worker = threading.Thread(target=thread_target)
            worker.start()
            worker.join()

        if "error" in result:
            st.error(f"Pipeline failed: {result['error']}")
        elif "pipeline" in result:
            pipeline = result["pipeline"]
            output_path = os.path.join(pipeline.cache_dir, f"{pipeline.pdf_name}.md")
            with open(output_path, encoding="utf-8") as fh:
                document = fh.read()

            st.success("Pipeline complete!")
            st.divider()

            tab_preview, tab_raw = st.tabs(["Preview", "Raw Markdown"])

            with tab_preview:
                st.markdown(document)

            with tab_raw:
                st.code(document, language="markdown")

            st.download_button(
                label="Download Markdown",
                data=document,
                file_name=f"{os.path.splitext(uploaded_file.name)[0]}.md",
                mime="text/markdown"
            )
