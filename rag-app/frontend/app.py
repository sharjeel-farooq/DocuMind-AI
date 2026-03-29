# frontend/app.py

import sys
from pathlib import Path

import streamlit as st
from openai import RateLimitError

# Ensure project root is on sys.path for imports when running via Streamlit
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rag.pipeline import process_document, ask_question, reset_index

st.title("📄 RAG AI Assistant")

if st.button("Clear index (remove all ingested docs)"):
    reset_index()
    st.success("Vector index cleared. Re-upload documents to continue.")

supported_types = ["pdf", "txt", "csv", "docx", "json", "xlsx", "xls"]
uploaded_files = st.file_uploader(
    "Upload documents (multiple allowed)",
    type=supported_types,
    accept_multiple_files=True,
)

if uploaded_files:
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)

    for uploaded_file in uploaded_files:
        suffix = Path(uploaded_file.name).suffix.lower()
        suffix = suffix if suffix in [f".{t}" for t in supported_types] else ".pdf"

        base_name = Path(uploaded_file.name).stem
        temp_path = temp_dir / f"{base_name}{suffix}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        try:
            process_document(str(temp_path))
            st.success(f"Processed: {uploaded_file.name}")
        except RateLimitError:
            st.error("OpenAI rate limit or quota exceeded while embedding. Check billing/quota or switch to a local model.")
        except Exception as exc:  # eslint-disable-line broad-except
            st.error(f"Failed to process {uploaded_file.name}: {exc}")

query = st.text_input("Ask a question:")
ask_clicked = st.button("Ask", disabled=not query.strip())

if ask_clicked and query:
    try:
        answer, sources = ask_question(query)

        st.write("### Answer:")
        st.write(answer)

        st.write("### Sources:")
        for s in sources:
            source = s.get("metadata", {}).get("source", "unknown")
            snippet = s.get("text", "")[:200]
            snippet_safe = snippet.replace("$", "\\$")
            st.write(f"- {source}: {snippet_safe}")
    except RateLimitError:
        st.error("OpenAI rate limit or quota exceeded while generating an answer. Check billing/quota or switch to a local model.")
    except Exception as exc:  # eslint-disable-line broad-except
        st.error(f"Failed to answer: {exc}")