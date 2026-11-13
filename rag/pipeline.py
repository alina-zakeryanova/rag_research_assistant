"""Run a named retrieval strategy against the current index."""

from __future__ import annotations

from rag.config import MAX_PER_PAPER, RERANK, TOP_K
from rag.hits import Hit
from rag.index import load_index
from rag.queries import decompose, hyde_passage, multi_queries, rewrite, route, step_back, sufficient
from rag.search import hybrid_hits

MODES = (
    "semantic",
    "bm25",
    "rrf",
    "rerank",
    "multi",
    "fusion",
    "decompose",
    "stepback",
    "hyde",
    "crag",
    "adaptive",
    "raptor",
    "route",
)


def retrieve_mode(
    question: str,
    mode: str = "rerank",
    top_k: int = TOP_K,
    per_paper: int = MAX_PER_PAPER,
) -> tuple[list[Hit], str]:
    mode = mode.lower()
    if mode not in MODES:
        known = ", ".join(MODES)
        raise ValueError(f"Unknown mode {mode!r}. Choose from: {known}")

    if mode == "raptor":
        from rag.raptor import raptor_hits

        return raptor_hits(question, top_k=top_k), "RAPTOR: searched two cluster summaries, then their passages"

    if mode == "route":
        kind = route(question)
        if kind == "outside":
            return [], "Routed outside the library, so nothing was retrieved."
        mode = "fusion" if kind == "broad" else "rerank"
        hits, note = retrieve_mode(question, mode, top_k=top_k, per_paper=per_paper)
        return hits, f"Routed to {kind}. {note}"

    if mode == "adaptive":
        kind = route(question)
        if kind == "outside":
            return [], "Adaptive RAG: the question is outside the library."
        chosen = "fusion" if kind == "broad" else "crag"
        hits, note = retrieve_mode(question, chosen, top_k=top_k, per_paper=per_paper)
        return hits, f"Adaptive RAG chose {chosen}. {note}"

    matrix, chunks = load_index()
    common = {
        "matrix": matrix,
        "chunks": chunks,
        "top_k": top_k,
        "per_paper": per_paper,
    }

    if mode == "semantic":
        hits = hybrid_hits(question, semantic_only=True, rerank=False, **common)
        return hits, "Semantic search only"
    if mode == "bm25":
        hits = hybrid_hits(question, bm25_only=True, rerank=False, **common)
        return hits, "BM25 only"
    if mode == "rrf":
        hits = hybrid_hits(question, rerank=False, **common)
        return hits, "BM25 + semantic, fused with reciprocal rank fusion"
    if mode == "rerank":
        hits = hybrid_hits(question, rerank=RERANK, **common)
        return hits, "Reciprocal rank fusion, then a model rerank"
    if mode in {"multi", "fusion"}:
        queries = multi_queries(question)
        hits = hybrid_hits(question, queries=queries, rerank=False, **common)
        return hits, "RAG fusion: " + " | ".join(queries)
    if mode == "decompose":
        queries = decompose(question)
        hits = hybrid_hits(question, queries=queries, rerank=False, **common)
        return hits, "Decomposition: " + " | ".join(queries)
    if mode == "stepback":
        queries = step_back(question)
        hits = hybrid_hits(question, queries=queries, rerank=False, **common)
        return hits, "Step-back: " + " | ".join(queries)
    if mode == "hyde":
        passage = hyde_passage(question)
        hits = hybrid_hits(question, hyde=passage, rerank=False, **common)
        return hits, "HyDE passage embedded, then fused with BM25 of the original question"
    if mode == "crag":
        hits = hybrid_hits(question, rerank=False, **common)
        if hits and not sufficient(question, [hit.text for hit in hits]):
            better = rewrite(question)
            hits = hybrid_hits(better, rerank=False, **common)
            return hits, f"CRAG rejected the first set and searched again for: {better}"
        return hits, "CRAG kept the first retrieved set"
    raise ValueError(mode)
