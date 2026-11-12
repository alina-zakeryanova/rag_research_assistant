"""Find passages for a question.

Vector similarity finds related wording. A word overlap score keeps names
such as Raman, HPLC, and PLS-DA from being washed out. Page 1 is penalized,
and each paper can contribute only a few passages.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from rag.config import LEXICAL_WEIGHT, MAX_PER_PAPER, PAGE_ONE_PENALTY, TOP_K
from rag.embed import embed_query
from rag.index import load_index

_STOP = frozenset(
    """
    a an the and or of to for in on with from by at as is are was were be been
    this that these those it its into about how what which who whom whose why
    when where use used using do does did can could should would will just
    than then also not no nor but if so such into over under between among
    paper papers study studies method methods
    """.split()
)
_WORD = re.compile(r"[a-z0-9][a-z0-9\-]{2,}")


@dataclass
class Hit:
    source: str
    page: int
    text: str
    score: float


def query_terms(text: str) -> list[str]:
    terms = []
    seen = set()
    for word in _WORD.findall(text.lower()):
        if word in _STOP or word in seen:
            continue
        seen.add(word)
        terms.append(word)
    return terms


def _lexical(terms: list[str], text: str) -> float:
    if not terms:
        return 0.0
    lowered = text.lower()
    words = set(_WORD.findall(lowered))
    found = sum(1 for term in terms if term in words or term in lowered)
    return found / len(terms)


def _references(text: str) -> bool:
    years = len(re.findall(r"\b(?:19|20)\d{2}\b", text))
    # PDFs often extract initials as "P ." rather than "P."
    initials = len(re.findall(r"\b[A-Z]\s*\.", text))
    if years >= 5 or (years >= 2 and initials >= 6):
        return True
    lowered = text.lower()
    if lowered.count("crossref") >= 2:
        return True
    return lowered.count("doi") >= 2 or lowered.count("et al") >= 3


def _front_matter(text: str, page: int) -> bool:
    if page != 1:
        return False
    head = text[:500].lower()
    markers = ("e-mail", "email", "received", "accepted", "copyright", "corresponding")
    return "@" in head or any(marker in head for marker in markers)


def retrieve(
    question: str,
    top_k: int = TOP_K,
    per_paper: int = MAX_PER_PAPER,
) -> list[Hit]:
    question = question.strip()
    if not question:
        raise ValueError("Question is empty")

    matrix, chunks = load_index()
    vector = np.asarray(embed_query(question), dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise RuntimeError("The question embedding was empty.")
    vector = vector / norm

    cosine = matrix @ vector
    terms = query_terms(question)
    scores = np.empty(len(chunks), dtype=np.float32)
    for index, chunk in enumerate(chunks):
        page = int(chunk["page"])
        text = chunk["text"]
        mixed = (1.0 - LEXICAL_WEIGHT) * float(cosine[index]) + LEXICAL_WEIGHT * _lexical(terms, text)
        if page == 1:
            mixed -= PAGE_ONE_PENALTY
        if _front_matter(text, page):
            mixed -= 0.08
        if _references(text):
            mixed -= 0.35
        scores[index] = mixed

    order = np.argsort(-scores)

    def pick(skip_references: bool) -> list[Hit]:
        counts: dict[str, int] = {}
        chosen: list[Hit] = []
        for row in order:
            chunk = chunks[int(row)]
            text = chunk["text"]
            if skip_references and _references(text):
                continue
            source = chunk["source"]
            if counts.get(source, 0) >= per_paper:
                continue
            counts[source] = counts.get(source, 0) + 1
            chosen.append(
                Hit(
                    source=source,
                    page=int(chunk["page"]),
                    text=text,
                    score=float(scores[row]),
                )
            )
            if len(chosen) >= top_k:
                break
        return chosen

    hits = pick(skip_references=True)
    if len(hits) < top_k:
        hits = pick(skip_references=False)
    return hits
