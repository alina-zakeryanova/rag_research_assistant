"""A short walkthrough of the mechanics, using one of your papers.

No language model. It loads a single PDF, cuts one page into chunks, and
compares a question vector with a related passage and an unrelated one.

    python main.py lesson
"""

from __future__ import annotations

import math
import textwrap

from rag.chunk import chunk_text
from rag.config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL, PAPERS_DIR
from rag.embed import embed_texts
from rag.load import load_pdf
from rag.ollama_client import require_model


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _pick_pdf():
    for path in sorted(PAPERS_DIR.glob("*.pdf")):
        try:
            pages = load_pdf(path)
        except Exception:
            continue
        # Skip the first page when a later one is long enough. Page 1 is
        # usually the title, authors, and journal header.
        body = [page for page in pages if page.number > 1 and len(page.text) > 1500]
        if not body:
            body = [page for page in pages if len(page.text) > 1500]
        if body:
            return path, body[0]
    return None, None


def _show(label: str, text: str) -> None:
    print()
    print(label)
    print(textwrap.fill(text, width=88, initial_indent="  ", subsequent_indent="  "))


def run() -> None:
    path, page = _pick_pdf()
    if path is None or page is None:
        raise SystemExit(f"No readable PDF text in {PAPERS_DIR}")

    chunks = chunk_text(page.text, CHUNK_SIZE, CHUNK_OVERLAP)
    if len(chunks) < 2:
        chunks = chunk_text(page.text, 500, 100)

    question = "How is adulteration of oil detected?"
    unrelated = "The git commit history stores each version of a source file."

    print("Lesson: what a RAG does before any language model is involved")
    print()
    print("1. The PDF becomes text.")
    print(f"   File: {path.name}")
    print(f"   Using page {page.number}, which has {len(page.text)} characters.")
    _show("Start of that page:", page.text[:400] + "...")

    print()
    print("2. The page is cut into chunks the search can look up.")
    print(f"   This page became {len(chunks)} chunks of about {CHUNK_SIZE} characters,")
    print(f"   with {CHUNK_OVERLAP} characters of overlap so a cut sentence is kept.")
    _show("First chunk:", chunks[0][:500] + ("..." if len(chunks[0]) > 500 else ""))

    print()
    print(f"3. Text becomes {EMBEDDING_MODEL} vectors (lists of numbers).")
    print("   Asking Ollama for three embeddings. The first call loads the model.")
    require_model(EMBEDDING_MODEL)
    vectors = embed_texts([question, chunks[0], unrelated])
    q_vec, chunk_vec, other_vec = vectors
    preview = ", ".join(f"{value:.3f}" for value in q_vec[:8])
    print(f"   The question is {len(q_vec)} numbers. The first 8 are: [{preview}]")
    print()
    print("   Cosine similarity, higher means closer in meaning:")
    print(f"   question vs first chunk of the paper : {cosine(q_vec, chunk_vec):.2f}")
    print(f"   question vs an unrelated sentence    : {cosine(q_vec, other_vec):.2f}")
    print()
    print("   The index does this comparison against every chunk of every paper.")
    print("   The language model only receives the chunks that score highest.")
    print()
    print("4. Try the full loop, retrieve then generate:")
    print('   python main.py "What methods detect olive oil adulteration?"')


def main() -> None:
    run()


if __name__ == "__main__":
    main()
