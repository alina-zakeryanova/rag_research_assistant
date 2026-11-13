"""One level of RAPTOR: cluster passages, summarize each cluster, search the summaries."""

from __future__ import annotations

import numpy as np

from rag.config import OLLAMA_MODEL
from rag.embed import embed_texts
from rag.hits import Hit
from rag.index import load_index
from rag.ollama_client import chat

_labels: np.ndarray | None = None
_notes: list[str] | None = None
_note_matrix: np.ndarray | None = None


def _kmeans(matrix: np.ndarray, clusters: int = 8, rounds: int = 12) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    count = len(matrix)
    clusters = min(clusters, count)
    centers = matrix[rng.choice(count, clusters, replace=False)].copy()
    labels = np.zeros(count, dtype=np.int32)
    for _ in range(rounds):
        labels = (matrix @ centers.T).argmax(axis=1)
        for index in range(clusters):
            members = matrix[labels == index]
            if len(members) == 0:
                continue
            center = members.mean(axis=0)
            norm = float(np.linalg.norm(center)) or 1.0
            centers[index] = center / norm
    return labels, centers


def _ensure() -> tuple[np.ndarray, list[str], np.ndarray, list[dict]]:
    global _labels, _notes, _note_matrix
    matrix, chunks = load_index()
    if _labels is not None and _notes is not None and _note_matrix is not None:
        return _labels, _notes, _note_matrix, chunks
    labels, centers = _kmeans(matrix, clusters=8)
    notes: list[str] = []
    for index in range(len(centers)):
        members = np.where(labels == index)[0]
        if len(members) == 0:
            notes.append("")
            continue
        member_vectors = matrix[members]
        nearest = members[(member_vectors @ centers[index]).argsort()[-3:]]
        samples = "\n\n".join(" ".join(chunks[int(row)]["text"].split())[:500] for row in nearest)
        try:
            note = chat(
                OLLAMA_MODEL,
                "Summarize these passages from scientific papers in two sentences.",
                samples,
            )
        except RuntimeError:
            note = samples[:400]
        notes.append(note.strip())
    vectors = embed_texts([note or "empty" for note in notes])
    note_matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(note_matrix, axis=1, keepdims=True)
    note_matrix = note_matrix / np.maximum(norms, 1e-12)
    _labels, _notes, _note_matrix = labels, notes, note_matrix
    return labels, notes, note_matrix, chunks


def raptor_hits(question: str, top_k: int = 5) -> list[Hit]:
    """Search cluster summaries, then return passages from the nearest clusters."""
    from rag.embed import embed_query

    labels, _notes, note_matrix, chunks = _ensure()
    vector = np.asarray(embed_query(question), dtype=np.float32)
    norm = float(np.linalg.norm(vector)) or 1.0
    vector = vector / norm
    best_clusters = (note_matrix @ vector).argsort()[-2:]
    chosen = [index for index, label in enumerate(labels) if label in set(int(item) for item in best_clusters)]
    if not chosen:
        return []
    matrix, _ = load_index()
    scores = matrix[chosen] @ vector
    order = scores.argsort()[::-1]
    hits: list[Hit] = []
    seen: dict[str, int] = {}
    for position in order:
        row = chosen[int(position)]
        chunk = chunks[row]
        source = chunk["source"]
        if seen.get(source, 0) >= 2:
            continue
        seen[source] = seen.get(source, 0) + 1
        hits.append(
            Hit(source=source, page=int(chunk["page"]), text=chunk["text"], score=float(scores[position]))
        )
        if len(hits) >= top_k:
            break
    return hits
