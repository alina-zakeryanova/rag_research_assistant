"""Extra queries: multi-query, decomposition, step-back, HyDE, routing, CRAG."""

from __future__ import annotations

import re

from rag.config import OLLAMA_MODEL
from rag.ollama_client import chat

_LINES = """Reply with search queries only, one per line, no numbering and no preamble."""


def _clean_lines(raw: str, limit: int) -> list[str]:
    queries: list[str] = []
    for line in raw.splitlines():
        line = re.sub(r"^[\d\.\)\-\*\s]+", "", line).strip().strip('"')
        if len(line) < 8:
            continue
        if line.lower().startswith(("here", "query", "sure", "the following")):
            continue
        queries.append(line[:300])
        if len(queries) >= limit:
            break
    return queries


def _ask(system: str, user: str) -> str:
    return chat(OLLAMA_MODEL, system, user).strip()


def multi_queries(question: str, count: int = 3) -> list[str]:
    raw = _ask(_LINES, f"Write {count} different search queries for this question.\n\n{question}")
    queries = _clean_lines(raw, count)
    if question not in queries:
        queries.insert(0, question)
    return queries[: count + 1]


def decompose(question: str) -> list[str]:
    raw = _ask(_LINES, f"Split this into 2 or 3 simpler search queries.\n\n{question}")
    queries = _clean_lines(raw, 3)
    return queries or [question]


def step_back(question: str) -> list[str]:
    broader = _ask(
        "Reply with one broader research question, a single line.",
        f"The specific question is:\n{question}",
    )
    broader = broader.splitlines()[0].strip().strip('"')
    if len(broader) < 8:
        return [question]
    return [question, broader[:300]]


def hyde_passage(question: str) -> str:
    passage = _ask(
        "Write a short paragraph that a scientific paper might contain as the answer. "
        "Do not say that it is hypothetical. Stay under 90 words.",
        question,
    )
    return passage[:900]


def route(question: str) -> str:
    raw = _ask(
        "Classify the question. Reply with one word: lookup, broad, or outside.\n"
        "lookup = a specific fact in the library.\n"
        "broad = needs several papers or a comparison.\n"
        "outside = not a question about these documents.",
        question,
    )
    word = raw.lower().split()[0].strip(".,") if raw else "lookup"
    if word not in {"lookup", "broad", "outside"}:
        return "lookup"
    return word


def sufficient(question: str, passages: list[str]) -> bool:
    blocks = [f"[{index}] {' '.join(text.split())[:350]}" for index, text in enumerate(passages)]
    raw = _ask(
        "Reply yes or no. Yes means at least one passage contains evidence for the question.",
        f"Question: {question}\n\n" + "\n".join(blocks),
    )
    return raw.lower().startswith("y")


def rewrite(question: str) -> str:
    raw = _ask(
        "Rewrite the question as a better search query for scientific papers. One line.",
        question,
    )
    line = raw.splitlines()[0].strip().strip('"') if raw else ""
    return line[:300] if len(line) >= 8 else question
