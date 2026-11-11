"""Step 5. Given a question, fetch the passages most likely to help.

This is the "R" in RAG. Nothing is generated here. The result is a short
list of chunks, each with the file and page it came from.

Both the question and the saved passages are converted to unit-length
vectors, so their dot product is cosine similarity. 1.0 means the same
direction, 0 means unrelated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rag.config import TOP_K
from rag.embed import embed_query
from rag.index import load_index


@dataclass
class Hit:
    source: str
    page: int
    text: str
    score: float


def retrieve(question: str, top_k: int = TOP_K) -> list[Hit]:
    question = question.strip()
    if not question:
        raise ValueError("Question is empty")

    matrix, chunks = load_index()
    vector = np.asarray(embed_query(question), dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise RuntimeError("The question embedding was empty.")
    vector = vector / norm

    scores = matrix @ vector
    k = min(top_k, len(chunks))
    # argpartition finds the top k without sorting the whole library.
    chosen = np.argpartition(scores, -k)[-k:]
    chosen = chosen[np.argsort(scores[chosen])[::-1]]

    hits: list[Hit] = []
    for row in chosen:
        chunk = chunks[int(row)]
        hits.append(
            Hit(
                source=chunk["source"],
                page=int(chunk["page"]),
                text=chunk["text"],
                score=float(scores[row]),
            )
        )
    return hits
