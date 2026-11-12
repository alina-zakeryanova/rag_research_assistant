"""Cut pages into overlapping passages. A passage stays on one page."""

from __future__ import annotations

from dataclasses import dataclass

from rag.config import CHUNK_OVERLAP, CHUNK_SIZE
from rag.load import Page


@dataclass
class Chunk:
    id: str
    source: str
    page: int
    text: str


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split one string into pieces of about `size` characters."""
    if size <= overlap:
        raise ValueError("CHUNK_SIZE must be larger than CHUNK_OVERLAP")

    text = " ".join(text.split())
    if not text:
        return []
    if len(text) <= size:
        return [text]

    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            # Prefer to stop at the end of a sentence, inside the second half
            # of the window, so chunks stay readable.
            cut = text.rfind(". ", start, end)
            if cut > start + size // 2:
                end = cut + 1
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break
        next_start = end - overlap
        if next_start <= start:
            next_start = end
        start = next_start
    return pieces


def chunk_pages(
    pages: list[Page],
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        for index, text in enumerate(chunk_text(page.text, size, overlap)):
            if len(text) < 80:
                continue
            chunks.append(
                Chunk(
                    id=f"{page.source}::p{page.number}::c{index}",
                    source=page.source,
                    page=page.number,
                    text=text,
                )
            )
    return chunks
