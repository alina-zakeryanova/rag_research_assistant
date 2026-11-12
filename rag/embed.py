"""Embed text with the Ollama model named in config. Papers and questions must share it."""

from __future__ import annotations

from rag.config import EMBEDDING_MODEL
from rag.ollama_client import embed


def embed_texts(texts: list[str], progress: bool = False) -> list[list[float]]:
    if not texts:
        return []

    vectors: list[list[float]] = []
    batch_size = 16
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        vectors.extend(embed(batch, EMBEDDING_MODEL))
        if progress:
            done = min(start + batch_size, len(texts))
            print(f"  embedded {done}/{len(texts)}", flush=True)
    return vectors


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
