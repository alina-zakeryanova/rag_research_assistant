"""Save and load the passage index on disk."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

from rag.chunk import Chunk
from rag.config import INDEX_DIR
from rag.embed import embed_texts

CHUNKS_PATH = INDEX_DIR / "chunks.jsonl"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"

_lock = threading.Lock()
_cache: tuple[np.ndarray, list[dict]] | None = None
_cache_stamp: tuple[int, int] | None = None


def indexed_count() -> int:
    if not CHUNKS_PATH.exists():
        return 0
    with CHUNKS_PATH.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return matrix / norms


def clear_cache() -> None:
    global _cache, _cache_stamp
    with _lock:
        _cache = None
        _cache_stamp = None


def build_index(chunks: list[Chunk], reset: bool = True) -> int:
    clear_cache()
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
    global _cache, _cache_stamp
    if not CHUNKS_PATH.exists() or not EMBEDDINGS_PATH.exists():
        raise RuntimeError("No index yet. Run: python main.py ingest")

    stamp = (CHUNKS_PATH.stat().st_mtime_ns, EMBEDDINGS_PATH.stat().st_mtime_ns)
    with _lock:
        if _cache is not None and _cache_stamp == stamp:
            return _cache
        with CHUNKS_PATH.open(encoding="utf-8") as handle:
            chunks = [json.loads(line) for line in handle if line.strip()]
        matrix = np.load(EMBEDDINGS_PATH)
        if len(chunks) != len(matrix):
            raise RuntimeError("The index files disagree. Run: python main.py ingest")
        _cache = (matrix, chunks)
        _cache_stamp = stamp
        return _cache


def index_path() -> Path:
    return INDEX_DIR
