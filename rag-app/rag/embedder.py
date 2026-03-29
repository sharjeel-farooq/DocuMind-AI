import os
from functools import lru_cache
from typing import List

import numpy as np
from huggingface_hub import InferenceClient

from utils import config


@lru_cache(maxsize=1)
def _get_client() -> InferenceClient:
    model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    token = config.get_hf_token()
    return InferenceClient(model=model, token=token)


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec) + 1e-12
    return vec / norm


def embed_texts(texts: List[str]) -> np.ndarray:
    client = _get_client()
    embeddings = []
    for text in texts:
        # Hugging Face Inference API feature-extraction returns a list of floats per input
        vec = np.array(client.feature_extraction(text), dtype=np.float32)
        embeddings.append(_normalize(vec))
    return np.vstack(embeddings)


def embed_chunks(chunks: List[str]) -> np.ndarray:
    return embed_texts(chunks)


def embed_query(query: str) -> np.ndarray:
    return embed_texts([query])[0]