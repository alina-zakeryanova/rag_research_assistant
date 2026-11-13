"""The catalog of indexed papers: names, coverage samples, lookup."""

from __future__ import annotations

import re

from rag.index import load_index
from rag.hits import Hit

_LEADING_NUMBER = re.compile(r"^\d+_")


def display_name(source: str) -> str:
    stem = source[:-4] if source.lower().endswith(".pdf") else source
    stem = _LEADING_NUMBER.sub("", stem)
    return stem.replace("_", " ").replace("-", " ").strip()


def list_papers() -> list[dict]:
    _, chunks = load_index()
    by_source: dict[str, dict] = {}
    for chunk in chunks:
        source = chunk["source"]
        record = by_source.get(source)
        if record is None:
            record = {
                "source": source,
                "title": display_name(source),
                "chunks": 0,
                "pages": 0,
            }
            by_source[source] = record
        record["chunks"] += 1
        record["pages"] = max(record["pages"], int(chunk["page"]))
    return sorted(by_source.values(), key=lambda paper: paper["title"].lower())


def resolve_source(query: str) -> str:
    papers = list_papers()
    needle = query.strip().lower()
    if not needle:
        raise ValueError("Name a paper, or part of its file name.")
    matches = [
        paper
        for paper in papers
        if needle in paper["source"].lower() or needle in paper["title"].lower()
    ]
    if len(matches) == 1:
        return matches[0]["source"]
    if not matches:
        raise ValueError(f"No paper matches {query!r}.")
    names = "\n".join(f"  {paper['source']}" for paper in matches[:8])
    raise ValueError(f"Several papers match {query!r}:\n{names}")


def sample_passages(source: str, limit: int = 6) -> list[Hit]:
    """Spread a few passages across the paper so a summary sees more than the abstract."""
    _, chunks = load_index()
    owned = [chunk for chunk in chunks if chunk["source"] == source]
    if not owned:
        raise ValueError(f"{source} is not in the index.")
    if len(owned) <= limit:
        chosen = owned
    else:
        step = len(owned) / limit
        chosen = [owned[int(index * step)] for index in range(limit)]
    return [
        Hit(source=source, page=int(chunk["page"]), text=chunk["text"], score=0.0)
        for chunk in chosen
    ]
