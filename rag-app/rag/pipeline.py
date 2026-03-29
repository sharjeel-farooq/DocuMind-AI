# rag/pipeline.py

import re
import unicodedata
from pathlib import Path

from rag.loader import (
    pdf_loader,
    txt_loader,
    csv_loader,
    csv_loader_rows,
    excel_loader,
    docx_loader,
    json_loader,
)
from rag.chunker import chunk_text, strip_math_symbols
from rag.embedder import embed_chunks
from rag.retriever import retrieve
from rag.generator import generate_answer, clear_answer_cache
from rag.vectorstore import add_to_db, query_db, clear_db
from utils import config
from utils.logger import get_logger


logger = get_logger("pipeline")


def _basic_clean(text: str) -> str:
    """Normalize and drop control chars while keeping extended Latin/quotes/dashes."""

    text = unicodedata.normalize("NFKC", text)
    # Keep basic printable ASCII plus Latin-1/Extended and common punctuation blocks
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\u024F\u2000-\u206F]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _salvage_chunks(chunks):
    salvaged = []
    for c in chunks:
        if not c:
            continue
        stripped = re.sub(r"[^\w .,;:?!'\-]", " ", c, flags=re.UNICODE)
        stripped = re.sub(r"\s+", " ", stripped).strip()
        if sum(ch.isalnum() for ch in stripped) >= 4:
            salvaged.append(stripped)
    return salvaged


def process_document(file):
    ext = Path(file).suffix.lower()
    chunk_size = config.get_chunk_size()

    if ext == ".pdf":
        text = pdf_loader(file)
        text = strip_math_symbols(text)
        text = _basic_clean(text)
        chunks = chunk_text(text, chunk_size=chunk_size)
    elif ext == ".txt":
        text = txt_loader(file)
        text = strip_math_symbols(text)
        text = _basic_clean(text)
        chunks = chunk_text(text, chunk_size=chunk_size)
    elif ext == ".csv":
        rows = csv_loader_rows(file)
        rows = [strip_math_symbols(r) for r in rows]
        chunks = rows
    elif ext in (".xlsx", ".xls"):
        text = excel_loader(file)
        # Excel loader already produces labeled rows; keep them as-is.
        chunks = [line.strip() for line in text.splitlines() if line.strip()]
    elif ext == ".docx":
        text = docx_loader(file)
        text = strip_math_symbols(text)
        text = _basic_clean(text)
        chunks = chunk_text(text, chunk_size=chunk_size)
    elif ext == ".json":
        text = json_loader(file)
        text = strip_math_symbols(text)
        text = _basic_clean(text)
        chunks = chunk_text(text, chunk_size=chunk_size)
    else:
        raise ValueError("Unsupported file type")

    # Drop empty or low-information chunks, but fall back to raw if all are filtered
    filtered_chunks = []
    for c in chunks:
        if not c:
            continue
        c_strip = c.strip()
        if not c_strip:
            continue
        if sum(ch.isalnum() for ch in c_strip) < 4:
            continue
        filtered_chunks.append(c_strip)

    if not filtered_chunks:
        salvaged = _salvage_chunks(chunks)
        logger.info(
            "cleaning removed all chunks; salvage=%d original=%d", len(salvaged), len(chunks)
        )
        filtered_chunks = salvaged

    if not filtered_chunks:
        raise ValueError("No usable content found after cleaning; document may be empty or unsupported.")

    embeddings = embed_chunks(filtered_chunks)

    base_id = Path(file).stem
    ids = [f"{base_id}_{i}" for i in range(len(filtered_chunks))]
    metadatas = [{"source": Path(file).name} for _ in filtered_chunks]

    add_to_db(filtered_chunks, embeddings, ids, metadatas)
    clear_answer_cache()


def ask_question(query):
    results = retrieve(query)
    answer = generate_answer(query, results)

    return answer, results


def reset_index():
    clear_db()
    clear_answer_cache()