"""Step 4. Save chunks and their vectors.

The index is two files, written next to each other:

- chunks.jsonl is the text a person can open
- embeddings.npy is one row of numbers per line of that file

Searching it later is a nearest-neighbour lookup: which saved rows sit
closest to the question row? retrieve.py does that with NumPy. Chroma,
FAISS, and Pinecone do the same lookup once a library is too big to scan
directly. Thirty papers are not that big.

`python main.py ingest` deletes the files and writes them again from the
PDFs. The PDFs stay the source of truth.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rag.chunk import Chunk
from rag.config import INDEX_DIR
from rag.embed import embed_texts

CHUNKS_PATH = INDEX_DIR / "chunks.jsonl"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"


def indexed_count() -> int:
    if not CHUNKS_PATH.exists():
        return 0
    with CHUNKS_PATH.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return matrix / norms


def build_index(chunks: list[Chunk], reset: bool = True) -> int:
    if not chunks:
        raise RuntimeError("No text chunks to index. The PDFs may be scans.")
    if not reset and CHUNKS_PATH.exists():
        raise RuntimeError("An index already exists. Re-run ingest to rebuild it.")

    vectors = embed_texts([chunk.text for chunk in chunks], progress=True)
    matrix = _normalize(np.asarray(vectors, dtype=np.float32))

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, matrix)
    with CHUNKS_PATH.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(
                json.dumps(
                    {"source": chunk.source, "page": chunk.page, "text": chunk.text},
                    ensure_ascii=False,
                )
                + "\n"
            )
    return len(chunks)


def load_index() -> tuple[np.ndarray, list[dict]]:
    if not CHUNKS_PATH.exists() or not EMBEDDINGS_PATH.exists():
        raise RuntimeError("No index yet. Run: python main.py ingest")

    with CHUNKS_PATH.open(encoding="utf-8") as handle:
        chunks = [json.loads(line) for line in handle if line.strip()]
    matrix = np.load(EMBEDDINGS_PATH)
    if len(chunks) != len(matrix):
        raise RuntimeError(
            "The index files disagree. Run: python main.py ingest"
        )
    return matrix, chunks


def index_path() -> Path:
    return INDEX_DIR
