# rag/generator.py

import os
import time
from functools import lru_cache
from collections import OrderedDict

import numpy as np
from groq import Groq

from rag.embedder import embed_query, embed_texts
from utils import config
from utils.logger import get_logger


@lru_cache(maxsize=1)
def _get_llm_client() -> Groq:
    """Use Groq chat completions (llama3-8b-8192 by default)."""

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Missing Groq API key (set GROQ_API_KEY)")

    return Groq(api_key=api_key)


logger = get_logger("generator")

_ANSWER_CACHE: OrderedDict = OrderedDict()
_MAX_CACHE = 100


def _cache_key(query, results):
    sources = []
    for item in results:
        meta = item.get("metadata", {}) or {}
        src = meta.get("source", "unknown")
        if src:
            sources.append(src)
    sources = tuple(sorted(set(sources)))
    return (query.strip(), sources)


def _get_cached_answer(key):
    if key in _ANSWER_CACHE:
        _ANSWER_CACHE.move_to_end(key)
        return _ANSWER_CACHE[key]
    return None


def _set_cached_answer(key, value):
    _ANSWER_CACHE[key] = value
    _ANSWER_CACHE.move_to_end(key)
    if len(_ANSWER_CACHE) > _MAX_CACHE:
        _ANSWER_CACHE.popitem(last=False)


def clear_answer_cache():
    _ANSWER_CACHE.clear()


def _build_prompt(query, top_chunks, intent):
    context = "\n\n".join(top_chunks)
    
    intent_instructions = {
        "compare": "Provide a concise comparison. Start with a short summary, then a small table of key differences, then bullet recommendations.",
        "task": "Extract the requested task and list clear steps or requirements.",
        "summarize": "Give a brief bullet-point summary of the most important facts.",
        "list": "Return a clear bullet list of the requested items.",
        "generic": "Answer concisely using bullets where helpful.",
    }

    formatting = "- Use bullets or numbered steps. If comparing, include a compact table (e.g., Feature | Doc/Model A | Doc/Model B)."

    return f"""
    You are a helpful AI assistant.

    Instructions:
    - Use ONLY the provided context.
    - {intent_instructions.get(intent, intent_instructions['generic'])}
    - {formatting}
    - If the answer is not found, say "I don't know".

    Context:
    {context}

    Question: {query}

    Answer:
    """


def clean_chunks(chunks):
    cleaned = []

    for c in chunks:
        words = c.split()
        num_count = sum(w.replace('.', '', 1).isdigit() for w in words)

        if num_count < len(words) * 0.5:
            cleaned.append(c)

    return cleaned


def filter_sources(chunks):
    clean = []

    for c in chunks:
        words = c.split()
        num_count = sum(w.replace('.', '', 1).isdigit() for w in words)

        if num_count < len(words) * 0.4:
            clean.append(c)

    return clean


def dedup_chunks(chunks):
    seen = set()
    unique = []

    for c in chunks:
        key = c.strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(c)

    return unique


def detect_intent(query):
    q = query.lower()
    if "compare" in q or " vs " in q:
        return "compare"
    if "task" in q and any(ch.isdigit() for ch in q):
        return "task"
    if "summarize" in q or "summary" in q:
        return "summarize"
    if q.startswith("list") or "list " in q:
        return "list"
    return "generic"


def _query_terms(query):
    return [w.lower() for w in query.split() if w]


def _select_best_chunk(query, chunks):
    if not chunks:
        return ""

    terms = _query_terms(query)
    digit_terms = {t for t in terms if t.isdigit()}

    def score(chunk):
        text = chunk.lower()
        words = chunk.split()
        base = sum(2 if term in text else 0 for term in terms)

        for d in digit_terms:
            if f"task {d}" in text or f"task {d}:" in text:
                base += 4

        # Prefer longer, more descriptive chunks lightly
        base += min(len(words), 40) * 0.05
        return base

    scored = sorted(((score(c), c) for c in chunks), reverse=True)
    return scored[0][1] if scored else ""


def fallback_answer(query, chunks):
    best = _select_best_chunk(query, chunks)

    if not best:
        return "I don't know"

    return f"Based on the document:\n\n{best}"


def generate_with_retry(client, model_name, prompt, max_new_tokens):
    for _ in range(3):
        try:
            return client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_new_tokens,
                temperature=0.3,
            )
        except Exception:
            time.sleep(1)

    raise Exception("LLM failed after retries")


def generate_answer(query, results, k=3, max_new_tokens=256):
    intent = detect_intent(query)
    base_top_k = config.get_top_k()

    cache_key = _cache_key(query, results)
    cached = _get_cached_answer(cache_key)
    if cached:
        return cached

    if intent == "compare":
        # Ensure we include at least one chunk per source so both docs are visible
        source_chunks = {}
        for item in results:
            text = item.get("text", "")
            meta = item.get("metadata", {}) or {}
            src = meta.get("source", "unknown")
            if text and src not in source_chunks:
                source_chunks[src] = text

        context_chunks = [f"[{src}] {text}" for src, text in source_chunks.items() if text]
        context_chunks = dedup_chunks(context_chunks)

        if not context_chunks:
            return "I don't know"

        filtered_chunks = filter_sources(context_chunks)
        cleaned_chunks = clean_chunks(filtered_chunks)
        logger.info("compare intent: sources=%d kept=%d", len(source_chunks), len(cleaned_chunks))

        if not cleaned_chunks:
            return fallback_answer(query, filtered_chunks or context_chunks)

        cleaned_chunks = cleaned_chunks[:6]
        prompt = _build_prompt(query, cleaned_chunks, intent)
        llm = _get_llm_client()
        model_name = config.get_generation_model()

        try:
            start = time.time()
            resp = generate_with_retry(llm, model_name, prompt, max_new_tokens)
            elapsed = (time.time() - start) * 1000
            logger.info("llm_response model=%s time_ms=%.1f chunks=%d", model_name, elapsed, len(cleaned_chunks))
            answer = resp.choices[0].message.content.strip()
        except Exception as exc:
            answer = f"(LLM fallback: {exc})\n{fallback_answer(query, cleaned_chunks)}"

        _set_cached_answer(cache_key, answer)
        return answer

    # Default path: rank by similarity and keep top-k chunks for the prompt
    context_chunks = [item.get("text", "") for item in results]
    context_chunks = [c for c in context_chunks if c]
    context_chunks = dedup_chunks(context_chunks)

    if not context_chunks:
        return "I don't know"

    query_embedding = embed_query(query)
    context_embeddings = embed_texts(context_chunks)
    scores = context_embeddings @ query_embedding

    top_k = base_top_k
    top_indices = np.argsort(-scores)[:top_k]
    top_chunks = [context_chunks[i] for i in top_indices if context_chunks[i]]

    if not top_chunks:
        return "I don't know"

    filtered_chunks = filter_sources(top_chunks)
    cleaned_chunks = clean_chunks(filtered_chunks)
    logger.info("rerank intent=%s top_k=%d kept=%d", intent, top_k, len(cleaned_chunks))

    if not cleaned_chunks:
        return fallback_answer(query, filtered_chunks or top_chunks)

    cleaned_chunks = cleaned_chunks[:5]

    prompt = _build_prompt(query, cleaned_chunks, intent)
    llm = _get_llm_client()
    model_name = config.get_generation_model()

    try:
        start = time.time()
        resp = generate_with_retry(llm, model_name, prompt, max_new_tokens)
        elapsed = (time.time() - start) * 1000
        logger.info("llm_response model=%s time_ms=%.1f chunks=%d", model_name, elapsed, len(cleaned_chunks))
        answer = resp.choices[0].message.content.strip()
    except Exception as exc:
        answer = f"(LLM fallback: {exc})\n{fallback_answer(query, cleaned_chunks)}"

    _set_cached_answer(cache_key, answer)
    return answer