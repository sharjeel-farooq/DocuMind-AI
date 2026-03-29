import re

from utils import config


def _split_by_headings(text):
    """Split by task/section-like headings to keep semantic units together."""

    heading_re = re.compile(r"^(task|section|part|question)\s*\d+[:.\-]?", re.IGNORECASE)
    lines = text.splitlines()

    chunks = []
    current = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if heading_re.match(stripped):
            if current:
                chunks.append("\n".join(current).strip())
                current = []
        current.append(stripped)

    if current:
        chunks.append("\n".join(current).strip())

    # Filter tiny fragments
    return [c for c in chunks if len(c.split()) > 5]


def strip_math_symbols(text: str) -> str:
    """Remove noisy Unicode math symbols, safely, without touching ASCII text."""

    # Fix: DO NOT use \u1D400-\u1D7FF in re.sub — it corrupts ASCII on some Python builds
    # Instead, remove characters explicitly by codepoint using a filter

    def is_math_char(c):
        cp = ord(c)
        return (
            0x2200 <= cp <= 0x22FF or   # Math operators (∑ ∫ ∂ ≠)
            0x1D400 <= cp <= 0x1D7FF or # Math alphanumerics (𝐀 𝑥) — safe via ord()
            0x2070 <= cp <= 0x208F      # Superscripts/subscripts
        )

    text = "".join(" " if is_math_char(c) else c for c in text)

    # Normalize common Unicode punctuation to ASCII
    text = text.replace("\u2022", "-")   # bullet •
    text = text.replace("\u2013", "-")   # en-dash –
    text = text.replace("\u2014", "-")   # em-dash —
    text = text.replace("\u2018", "'")   # left single quote '
    text = text.replace("\u2019", "'")   # right single quote '
    text = text.replace("\u201C", '"')   # left double quote "
    text = text.replace("\u201D", '"')   # right double quote "
    text = text.replace("\u00A0", " ")   # non-breaking space

    text = re.sub(r"\s+", " ", text).strip()
    return text


def _sentence_chunks(text, chunk_size=None):
    chunk_size = chunk_size or config.get_chunk_size()
    sentences = re.split(r'(?<=[.!?]) +', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def smart_chunk(text):
    structured = _split_by_headings(text)
    if len(structured) > 1:
        return structured
    return _sentence_chunks(text)


def chunk_text(text, chunk_size=None, overlap=100):
    # Prefer structure-aware splitting; fallback to sentences with a size cap.
    structured = _split_by_headings(text)
    if len(structured) > 1:
        return structured

    return _sentence_chunks(text, chunk_size=chunk_size)


# Temporary alias for compatibility if other modules still import `chunker`
chunker = chunk_text