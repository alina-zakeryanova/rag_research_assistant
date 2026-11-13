"""Ask the local model to reorder a short list of passages."""

from __future__ import annotations

import re

from rag.config import OLLAMA_MODEL
from rag.ollama_client import chat

_PROMPT = """Rank these passages for the question.
Reply with the passage numbers, best first, separated by commas.
Do not explain. Use only numbers from the list.

Question: {question}

{passages}"""


def rerank_indices(question: str, passages: list[str]) -> list[int]:
    if len(passages) < 2:
        return list(range(len(passages)))
    blocks = []
    for index, text in enumerate(passages):
        snippet = " ".join(text.split())[:450]
        blocks.append(f"[{index}] {snippet}")
    try:
        raw = chat(OLLAMA_MODEL, "You reorder evidence. Reply with numbers only.", _PROMPT.format(question=question, passages="\n\n".join(blocks)))
    except RuntimeError:
        return list(range(len(passages)))
    order: list[int] = []
    for match in re.findall(r"\d+", raw):
        index = int(match)
        if 0 <= index < len(passages) and index not in order:
            order.append(index)
    for index in range(len(passages)):
        if index not in order:
            order.append(index)
    return order
