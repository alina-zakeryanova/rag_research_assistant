"""Semantic search, BM25, and reciprocal rank fusion."""

from __future__ import annotations

import numpy as np

from rag.bm25 import score_texts
from rag.config import PAGE_ONE_PENALTY, POOL, RRF_K
from rag.embed import embed_query
from rag.filters import is_front_matter, is_reference
from rag.hits import Hit


def searchable(chunk: dict) -> str:
    context = chunk.get("context") or ""
    if context:
        return f"{context}\n{chunk['text']}"
    return chunk["text"]


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise RuntimeError("The question embedding was empty.")
    return vector / norm


def _order(scores: np.ndarray, banned: set[int]) -> list[int]:
    return [int(index) for index in np.argsort(-scores) if int(index) not in banned]


def rrf(rankings: list[list[int]], size: int, k: int = RRF_K) -> np.ndarray:
    fused = np.zeros(size, dtype=np.float32)
    for ranking in rankings:
        for rank, index in enumerate(ranking):
            fused[index] += 1.0 / (k + rank + 1)
    return fused


def _penalize(scores: np.ndarray, chunks: list[dict]) -> np.ndarray:
    adjusted = scores.copy()
    for index, chunk in enumerate(chunks):
        page = int(chunk["page"])
        text = chunk["text"]
        if page == 1:
            adjusted[index] -= PAGE_ONE_PENALTY * 0.02
        if is_front_matter(text, page):
            adjusted[index] -= 0.004
    return adjusted


def _pick(order: list[int], scores: np.ndarray, chunks: list[dict], limit: int, per_paper: int) -> list[int]:
    counts: dict[str, int] = {}
    chosen: list[int] = []
    for index in order:
        source = chunks[index]["source"]
        if counts.get(source, 0) >= per_paper:
            continue
        counts[source] = counts.get(source, 0) + 1
        chosen.append(index)
        if len(chosen) >= limit:
            break
    return chosen


def _hits(indices: list[int], scores: np.ndarray, chunks: list[dict]) -> list[Hit]:
    return [
        Hit(
            source=chunks[index]["source"],
            page=int(chunks[index]["page"]),
            text=chunks[index]["text"],
            score=float(scores[index]),
        )
        for index in indices
    ]


def semantic_ranking(matrix: np.ndarray, question: str, banned: set[int]) -> list[int]:
    vector = _unit(np.asarray(embed_query(question), dtype=np.float32))
    return _order(matrix @ vector, banned)


def bm25_ranking(texts: list[str], question: str, banned: set[int]) -> list[int]:
    return _order(score_texts(texts, question), banned)


def hybrid_hits(
    question: str,
    matrix: np.ndarray,
    chunks: list[dict],
    top_k: int = 5,
    per_paper: int = 2,
    rerank: bool = False,
    queries: list[str] | None = None,
    hyde: str | None = None,
    semantic_only: bool = False,
    bm25_only: bool = False,
) -> list[Hit]:
    question = question.strip()
    if not question:
        raise ValueError("Question is empty")
    texts = [searchable(chunk) for chunk in chunks]
    banned = {index for index, chunk in enumerate(chunks) if is_reference(chunk["text"])}
    probes = queries or [question]
    rankings: list[list[int]] = []
    for probe in probes:
        if not bm25_only:
            rankings.append(semantic_ranking(matrix, probe, banned))
        if not semantic_only:
            rankings.append(bm25_ranking(texts, probe, banned))
    if hyde and not bm25_only:
        rankings.append(semantic_ranking(matrix, hyde, banned))
    fused = _penalize(rrf(rankings, len(chunks)), chunks)
    for index in banned:
        fused[index] = -1.0
    order = [int(index) for index in np.argsort(-fused) if int(index) not in banned]
    pool = _pick(order, fused, chunks, max(top_k, POOL if rerank else top_k), per_paper if not rerank else per_paper + 1)
    if rerank and len(pool) > top_k:
        from rag.rerank import rerank_indices

        ranked = rerank_indices(question, [chunks[index]["text"] for index in pool])
        order = [pool[index] for index in ranked]
        pool = _pick(order, fused, chunks, top_k, per_paper)
    else:
        pool = pool[:top_k]
    return _hits(pool, fused, chunks)
