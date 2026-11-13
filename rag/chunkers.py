"""Four ways to cut a paper, plus a context line for contextual retrieval."""

from __future__ import annotations

import re

from rag.chunk import Chunk
from rag.load import Page

_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_HEADING = re.compile(
    r"(?:^|[.!?]\s+)("
    r"abstract|introduction|related work|background|"
    r"materials and methods|material and methods|methodology|"
    r"methods|experimental setup|experimental|"
    r"results and discussion|results|discussion|"
    r"conclusion|conclusions|references"
    r")\b",
    re.IGNORECASE,
)
_LEADING = re.compile(r"^\d+_")


def _title(source: str) -> str:
    stem = source[:-4] if source.lower().endswith(".pdf") else source
    stem = _LEADING.sub("", stem)
    return stem.replace("_", " ").replace("-", " ").strip()


def sentences(text: str) -> list[str]:
    parts = [part.strip() for part in _SENTENCE.split(text) if part.strip()]
    return [part for part in parts if len(part) > 40]


def _emit(chunks: list[Chunk], source: str, page: int, text: str, section: str, index: int) -> None:
    text = " ".join(text.split())
    if len(text) < 80:
        return
    chunks.append(
        Chunk(
            id=f"{source}::p{page}::c{index}",
            source=source,
            page=page,
            text=text,
            section=section,
        )
    )


def fixed_chunks(pages: list[Page], size: int = 1000) -> list[Chunk]:
    """Equal character windows. The next window starts where the last one ended."""
    chunks: list[Chunk] = []
    for page in pages:
        start = 0
        index = 0
        text = page.text
        while start < len(text):
            _emit(chunks, page.source, page.number, text[start : start + size], "", index)
            index += 1
            start += size
    return chunks


def overlap_chunks(pages: list[Page], window: int = 5, overlap: int = 1) -> list[Chunk]:
    """Windows of `window` sentences, stepping so `overlap` sentences are shared."""
    step = max(window - overlap, 1)
    chunks: list[Chunk] = []
    for page in pages:
        parts = sentences(page.text)
        index = 0
        for start in range(0, len(parts), step):
            group = parts[start : start + window]
            if len(group) < 2 and chunks and chunks[-1].source == page.source:
                break
            _emit(chunks, page.source, page.number, " ".join(group), "", index)
            index += 1
            if start + window >= len(parts):
                break
    return chunks


def structure_chunks(pages: list[Page]) -> list[Chunk]:
    """Split where a paper section starts, then cap long sections at five sentences."""
    chunks: list[Chunk] = []
    for page in pages:
        text = page.text
        matches = list(_HEADING.finditer(text))
        spans: list[tuple[str, str]] = []
        if not matches:
            spans.append(("", text))
        else:
            if matches[0].start() > 80:
                spans.append(("", text[: matches[0].start()]))
            for position, match in enumerate(matches):
                section = match.group(1).lower()
                body_start = match.end()
                body_end = matches[position + 1].start() if position + 1 < len(matches) else len(text)
                spans.append((section, text[body_start:body_end]))
        index = 0
        for section, body in spans:
            parts = sentences(body) or ([" ".join(body.split())] if body.strip() else [])
            for start in range(0, len(parts), 5):
                _emit(chunks, page.source, page.number, " ".join(parts[start : start + 5]), section, index)
                index += 1
    return chunks


def semantic_chunks(pages: list[Page], threshold: float = 0.45, max_sentences: int = 5) -> list[Chunk]:
    """Start a new passage when the next sentence drifts away from the previous one."""
    from rag.embed import embed_texts

    records: list[tuple[str, int, str]] = []
    for page in pages:
        for sentence in sentences(page.text):
            records.append((page.source, page.number, sentence))
    if not records:
        return []
    vectors = embed_texts([item[2] for item in records], progress=True)
    chunks: list[Chunk] = []
    bucket: list[str] = []
    source = records[0][0]
    page = records[0][1]
    index = 0

    def flush() -> None:
        nonlocal bucket, index
        if bucket:
            _emit(chunks, source, page, " ".join(bucket), "", index)
            index += 1
            bucket = []

    for position, (src, number, sentence) in enumerate(records):
        if bucket and (src != source or number != page):
            flush()
            source, page = src, number
        if bucket:
            previous = vectors[position - 1]
            current = vectors[position]
            previous_norm = sum(value * value for value in previous) ** 0.5 or 1.0
            current_norm = sum(value * value for value in current) ** 0.5 or 1.0
            similarity = sum(a * b for a, b in zip(previous, current)) / (previous_norm * current_norm)
            if similarity < threshold or len(bucket) >= max_sentences:
                flush()
                source, page = src, number
        bucket.append(sentence)
    flush()
    return chunks


def with_context(chunks: list[Chunk]) -> list[Chunk]:
    """Contextual retrieval without an extra model call per passage.

    Each passage is embedded together with the paper name and the section.
    A full language-model sentence for every passage is a separate, much slower pass.
    """
    prepared: list[Chunk] = []
    for chunk in chunks:
        section = chunk.section or "body"
        prepared.append(
            Chunk(
                id=chunk.id,
                source=chunk.source,
                page=chunk.page,
                text=chunk.text,
                section=section,
                context=f"Paper: {_title(chunk.source)}. Section: {section}.",
            )
        )
    return prepared
