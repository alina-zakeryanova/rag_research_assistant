"""Build the index: PDFs -> pages -> chunks -> vectors -> data/index.

Run from the project folder:

    python main.py ingest
"""

from __future__ import annotations

import time

from rag.chunk import chunk_pages
from rag.config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL, PAPERS_DIR
from rag.index import build_index
from rag.load import load_papers
from rag.ollama_client import require_model


def build() -> None:
    require_model(EMBEDDING_MODEL)
    print(f"Reading PDFs in {PAPERS_DIR.name}/")
    print(f"Chunk size {CHUNK_SIZE} characters, overlap {CHUNK_OVERLAP}.")
    started = time.time()

    pages, skipped = load_papers(PAPERS_DIR)
    files = sorted({page.source for page in pages})
    print(f"Extracted text from {len(files)} files, {len(pages)} pages.")
    for name in skipped:
        print(f"  skipped {name}")

    chunks = chunk_pages(pages)
    print(f"Cut into {len(chunks)} chunks. Embedding them with {EMBEDDING_MODEL}...")

    count = build_index(chunks, reset=True)
    elapsed = time.time() - started
    print(f"Saved {count} chunks in {elapsed:.0f}s.")
    print("Open the library with: python main.py serve")


def main() -> None:
    build()


if __name__ == "__main__":
    main()
