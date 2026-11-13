"""Retrieve passages for a question.

The default is BM25 plus semantic search, fused, then a short model rerank.
Other strategies live in rag.pipeline.
"""

from __future__ import annotations

from rag.config import TOP_K
from rag.hits import Hit
from rag.pipeline import retrieve_mode

__all__ = ["Hit", "retrieve"]


def retrieve(question: str, top_k: int = TOP_K, per_paper: int = 2) -> list[Hit]:
    hits, _note = retrieve_mode(question, "rerank", top_k=top_k, per_paper=per_paper)
    return hits
