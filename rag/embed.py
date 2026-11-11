"""Step 3. Turn text into a vector.

An embedding model reads a sentence and returns a list of numbers. Sentences
about the same idea land close together, even when they share few words.
"olive oil fraud" can match a passage about "adulteration of extra virgin olive oil".

This project asks Ollama for all-minilm vectors (384 numbers). The same model
has to embed the papers and the question. Mixing two models makes the
distances meaningless.
"""

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
