"""Compare chunking and retrieval strategies on a fixed set of questions.

    python main.py bench

The first run embeds several versions of the library. That takes several minutes
on this Mac. Later runs reuse data/compare/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rag.chunk import Chunk
from rag.chunkers import (
    fixed_chunks,
    overlap_chunks,
    semantic_chunks,
    structure_chunks,
    with_context,
)
from rag.config import ROOT
from rag.embed import embed_texts
from rag.load import load_papers
from rag.pipeline import retrieve_mode
from rag.search import hybrid_hits

COMPARE_DIR = ROOT / "data" / "compare"

# Each question is paired with a filename fragment that should appear in the top 3.
QUESTIONS = (
    ("Which paper uses Raman spectroscopy on olive oil?", "raman"),
    ("How is hyperspectral imaging used to detect adulteration?", "hyperspectral"),
    ("What does the neural network paper say about olive oil?", "ANN_applied"),
    ("How is frying oil quality evaluated with visible near infrared?", "frying"),
    ("What does the git document say about version control?", "DevTestOps"),
)

CHUNKERS = ("fixed", "overlap", "structure", "semantic", "contextual")
RETRIEVERS = ("semantic", "bm25", "rrf", "rerank")
QUERY_MODES = ("rrf", "fusion", "decompose", "stepback", "hyde", "crag", "raptor")


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _ready(name: str) -> bool:
    folder = COMPARE_DIR / name
    return (folder / "chunks.jsonl").exists() and (folder / "embeddings.npy").exists()


def _save(name: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
    folder = COMPARE_DIR / name
    folder.mkdir(parents=True, exist_ok=True)
    matrix = _normalize(np.asarray(vectors, dtype=np.float32))
    np.save(folder / "embeddings.npy", matrix)
    with (folder / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(
                json.dumps(
                    {
                        "source": chunk.source,
                        "page": chunk.page,
                        "text": chunk.text,
                        "section": chunk.section,
                        "context": chunk.context,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"  saved {name}: {len(chunks)} passages", flush=True)


def _load(name: str) -> tuple[np.ndarray, list[dict]]:
    folder = COMPARE_DIR / name
    with (folder / "chunks.jsonl").open(encoding="utf-8") as handle:
        chunks = [json.loads(line) for line in handle if line.strip()]
    return np.load(folder / "embeddings.npy"), chunks


def build_bundles() -> None:
    from rag.config import PAPERS_DIR
    from rag.ollama_client import require_model
    from rag.config import EMBEDDING_MODEL

    missing = [name for name in CHUNKERS if not _ready(name)]
    if not missing:
        print("Comparison indexes already exist.", flush=True)
        return
    require_model(EMBEDDING_MODEL)
    print("Reading PDFs for the comparison indexes.", flush=True)
    pages, skipped = load_papers(PAPERS_DIR)
    for name in skipped:
        print(f"  skipped {name}", flush=True)

    makers = {
        "fixed": lambda: fixed_chunks(pages),
        "overlap": lambda: overlap_chunks(pages),
        "structure": lambda: structure_chunks(pages),
        "semantic": lambda: semantic_chunks(pages),
        "contextual": lambda: with_context(structure_chunks(pages)),
    }
    for name in missing:
        print(f"Building {name} chunks.", flush=True)
        chunks = makers[name]()
        print(f"  embedding {len(chunks)} {name} passages", flush=True)
        vectors = embed_texts([chunk.embed_text for chunk in chunks], progress=True)
        _save(name, chunks, vectors)


def _hit(sources: list[str], needle: str) -> str:
    found = any(needle.lower() in source.lower() for source in sources[:3])
    return "yes" if found else "no"


def _sources(hits) -> list[str]:
    return [hit.source for hit in hits]


def _table(title: str, columns: list[str], rows: list[tuple[str, list[str]]]) -> list[str]:
    lines = ["", title, "question".ljust(28) + "".join(column.rjust(12) for column in columns)]
    totals = [0] * len(columns)
    for label, cells in rows:
        lines.append(label.ljust(28) + "".join(cell.rjust(12) for cell in cells))
        for index, cell in enumerate(cells):
            totals[index] += cell == "yes"
    count = len(rows) or 1
    summary = "hits".ljust(28) + "".join(f"{score}/{len(rows)}".rjust(12) for score in totals)
    lines.append(summary)
    lines.append("A yes means the expected paper was in the top 3.")
    return lines


def run() -> None:
    build_bundles()
    lines = [
        "Retrieval comparison",
        "yes = the expected paper appeared in the top 3 passages.",
        "ColBERT is not in this run: it needs a token-level model and PyTorch, which is a poor fit beside Ollama on a 16GB M1 Air.",
        "Contextual here prefixes each passage with the paper name and section before embedding.",
        "A language-model sentence for every passage would be thousands of extra calls on this machine.",
    ]

    print("Comparing chunkers with semantic search.", flush=True)
    chunk_rows = []
    for question, needle in QUESTIONS:
        cells = []
        for name in CHUNKERS:
            matrix, chunks = _load(name)
            hits = hybrid_hits(question, matrix, chunks, top_k=5, per_paper=2, semantic_only=True, rerank=False)
            cells.append(_hit(_sources(hits), needle))
        chunk_rows.append((needle[:28], cells))
        print(f"  chunkers done: {needle}", flush=True)
    lines.extend(_table("Chunking (semantic search)", list(CHUNKERS), chunk_rows))

    print("Comparing lexical and semantic search on the sentence-overlap index.", flush=True)
    matrix, chunks = _load("overlap")
    retrieval_rows = []
    for question, needle in QUESTIONS:
        cells = []
        for mode in RETRIEVERS:
            hits = hybrid_hits(
                question,
                matrix,
                chunks,
                top_k=5,
                per_paper=2,
                semantic_only=mode == "semantic",
                bm25_only=mode == "bm25",
                rerank=mode == "rerank",
            )
            cells.append(_hit(_sources(hits), needle))
        retrieval_rows.append((needle[:28], cells))
        print(f"  retrievers done: {needle}", flush=True)
    lines.extend(_table("Search on overlap chunks", list(RETRIEVERS), retrieval_rows))

    print("Comparing query strategies. These call the language model.", flush=True)
    query_rows = []
    for question, needle in QUESTIONS[:3]:
        cells = []
        for mode in QUERY_MODES:
            hits, _note = retrieve_mode(question, mode, top_k=5)
            cells.append(_hit(_sources(hits), needle))
        query_rows.append((needle[:28], cells))
        print(f"  queries done: {needle}", flush=True)
    lines.extend(_table("Query strategies on the main index", list(QUERY_MODES), query_rows))

    report = "\n".join(lines) + "\n"
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)
    (COMPARE_DIR / "report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {COMPARE_DIR / 'report.txt'}")
