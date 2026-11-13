"""Questions, follow-ups, summaries, and comparisons over the library."""

from __future__ import annotations

from dataclasses import dataclass, field

from rag.answer import (
    COMPARE_PROMPT,
    SUMMARY_PROMPT,
    citation_warnings,
    generate,
)
from rag.config import OPENAI_API_KEY, TOP_K, writer_name
from rag.hits import Hit
from rag.library import display_name, resolve_source, sample_passages
from rag.ollama_client import chat, installed_models, model_ready
from rag.retrieve import retrieve

_REWRITE = """You rewrite follow-up questions into standalone search queries for a library of scientific papers.
If the follow-up already stands alone, return it unchanged.
Reply with the query only, one line, no quotes."""


@dataclass
class Exchange:
    question: str
    search_query: str
    answer: str
    note: str
    hits: list[Hit] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    kind: str = "answer"


def standalone_query(question: str, history: list[dict] | None) -> str:
    history = [turn for turn in (history or []) if turn.get("content")]
    if not history:
        return question
    if not OPENAI_API_KEY:
        names = installed_models()
        if names is None or not model_ready(names, writer_name()):
            return question

    previous = history[-4:]
    lines = []
    for turn in previous:
        role = "User" if turn.get("role") == "user" else "Assistant"
        text = " ".join(str(turn.get("content", "")).split())
        lines.append(f"{role}: {text[:500]}")
    lines.append(f"Follow-up: {question}")
    try:
        rewritten = chat(writer_name(), _REWRITE, "\n".join(lines))
    except RuntimeError:
        return question
    rewritten = rewritten.strip().strip('"').strip("'")
    rewritten = rewritten.splitlines()[0].strip() if rewritten else ""
    if len(rewritten) < 8:
        return question
    return rewritten[:300]


def _finish(question: str, search_query: str, hits: list[Hit], result, kind: str) -> Exchange:
    return Exchange(
        question=question,
        search_query=search_query,
        answer=result.text,
        note=result.note,
        hits=hits,
        warnings=citation_warnings(result.text, hits),
        kind=kind,
    )


def answer_question(
    question: str,
    history: list[dict] | None = None,
    top_k: int = TOP_K,
) -> Exchange:
    question = question.strip()
    lowered = question.lower()
    if lowered.startswith("summarize "):
        return summarize_paper(question.split(maxsplit=1)[1])
    if lowered.startswith("compare "):
        return compare_topic(question.split(maxsplit=1)[1])

    search_query = standalone_query(question, history)
    hits = retrieve(search_query, top_k=top_k)
    result = generate(question, hits)
    return _finish(question, search_query, hits, result, "answer")


def summarize_paper(query: str) -> Exchange:
    source = resolve_source(query)
    hits = sample_passages(source, limit=6)
    title = display_name(source)
    result = generate(
        f"Summarize {title} ({source}).",
        hits,
        system=SUMMARY_PROMPT,
    )
    return _finish(f"Summarize {source}", source, hits, result, "summary")


def compare_topic(topic: str) -> Exchange:
    topic = topic.strip()
    if not topic:
        raise ValueError("Say what to compare.")
    hits = retrieve(topic, top_k=6, per_paper=1)
    result = generate(topic, hits, system=COMPARE_PROMPT)
    return _finish(f"Compare {topic}", topic, hits, result, "compare")
