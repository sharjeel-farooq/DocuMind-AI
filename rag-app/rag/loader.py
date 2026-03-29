"""File-type specific loaders that normalize output to plain text."""

from pathlib import Path
from typing import Type

from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    JSONLoader,
    PyPDFLoader,
    TextLoader,
)
import pandas as pd
import xlrd
from pypdf import PdfReader
from pypdf.errors import PdfReadError, PdfStreamError


def _load_as_text(loader_cls: Type, file_path: str) -> str:
    """Instantiate loader, read documents, and join page content."""
    docs = loader_cls(file_path).load()
    return "\n\n".join(doc.page_content for doc in docs)


def pdf_loader(file_path: str) -> str:
    """Load PDF with fallback for truncated streams.
    

    PyPDFLoader can raise PdfStreamError for slightly corrupted or truncated
    files. When that happens we retry with pypdf directly using strict=False,
    which is more forgiving.
    """
    try:
        return _load_as_text(PyPDFLoader, file_path)
    except (PdfStreamError, PdfReadError):
        reader = PdfReader(file_path, strict=False)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)


def txt_loader(file_path: str) -> str:
    return _load_as_text(TextLoader, file_path)


def csv_loader(file_path: str) -> str:
    return _load_as_text(CSVLoader, file_path)


def csv_loader_rows(file_path: str) -> list[str]:
    """Return one string per row for better retrieval granularity."""
    df = pd.read_csv(file_path).fillna("")
    rows = df.astype(str).agg(" | ".join, axis=1).tolist()
    rows = [r.strip() for r in rows if r and r.strip().strip("|").strip()]
    return rows


def excel_loader(file_path: str) -> str:
    """Load Excel (.xls/.xlsx) to text, handling old .xls via xlrd directly.

    Returns newline-joined rows with column labels to preserve meaning.
    """

    suffix = Path(file_path).suffix.lower()

    if suffix == ".xls":
        book = xlrd.open_workbook(file_path)
        sheet = book.sheet_by_index(0)

        rows = []
        for i in range(sheet.nrows):
            row = [str(sheet.cell_value(i, j)) for j in range(sheet.ncols)]
            rows.append(row)

        if not rows:
            return ""
        headers = rows[0]
        data_rows = rows[1:] if len(rows) > 1 else []
        df = pd.DataFrame(data_rows, columns=headers)
    else:
        df = pd.read_excel(file_path, engine="openpyxl")

    df.dropna(how="all", inplace=True)
    df = df.fillna("")

    # Build labeled rows to keep context (col: value)
    labeled_rows = []
    cols = list(df.columns)
    for _, r in df.iterrows():
        parts = [f"{col}: {r[col]}" for col in cols if str(r[col]).strip()]
        text = "; ".join(parts).strip()
        if text:
            labeled_rows.append(text)

    return "\n".join(labeled_rows)


def docx_loader(file_path: str) -> str:
    primary = _load_as_text(Docx2txtLoader, file_path).strip()
    if primary:
        return primary

    # Fallback: python-docx handles tables/text boxes that docx2txt can miss
    try:
        from docx import Document

        doc = Document(file_path)
        parts = []

        for p in doc.paragraphs:
            text = p.text.strip()
            if text:
                parts.append(text)

        # Capture table cells, which are common in lab/assignment docs
        for table in doc.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text and c.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))

        combined = "\n\n".join(parts).strip()
        if combined:
            return combined
    except Exception:
        pass

    return primary


def json_loader(file_path: str) -> str:
    return _load_as_text(JSONLoader, file_path)