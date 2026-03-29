"""Configuration helpers for environment-backed secrets."""

import os

from dotenv import load_dotenv


# Load .env if present
load_dotenv()


def get_hf_token() -> str:
    """Return the Hugging Face token from env vars or raise with guidance."""

    token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
    if token:
        return token

    raise ValueError(
        "Missing Hugging Face token. Set HUGGINGFACEHUB_API_TOKEN or HF_TOKEN "
        "(PowerShell: $env:HUGGINGFACEHUB_API_TOKEN='your_token'; CMD: setx "
        "HUGGINGFACEHUB_API_TOKEN \"your_token\" then restart the shell)."
    )


# Model and retrieval settings (config-driven)
DEFAULT_MODEL = "llama-3.1-8b-instant"
DEFAULT_TOP_K = 5
DEFAULT_CHUNK_SIZE = 300


def get_generation_model() -> str:
    return os.getenv("GENERATION_MODEL", DEFAULT_MODEL)


def get_top_k() -> int:
    try:
        return int(os.getenv("TOP_K", DEFAULT_TOP_K))
    except ValueError:
        return DEFAULT_TOP_K


def get_chunk_size() -> int:
    try:
        return int(os.getenv("CHUNK_SIZE", DEFAULT_CHUNK_SIZE))
    except ValueError:
        return DEFAULT_CHUNK_SIZE
