"""Step 1. Read each PDF as text, one page at a time.

A PDF is a drawing of a page. RAG needs the characters. This file extracts
those characters and remembers which page they came from, so later answers
can cite a file and a page.

Image-only scans have no characters to extract. Those files are reported
and skipped. OCR is a later project, not this one.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

# pypdf warns when a PDF uses a custom font. The page text still extracts.
logging.getLogger("pypdf").setLevel(logging.ERROR)


@dataclass
class Page:
    source: str
    number: int
    text: str


def clean_text(text: str) -> str:
    """Join wrapped lines into a single searchable string."""
    text = text.replace("\x00", " ")
    text = text.replace("\u00ad", "")
    # Words split as "adul-\nteration" across a line break.
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_pdf(path: Path) -> list[Page]:
    reader = PdfReader(str(path))
    if reader.is_encrypted:
        reader.decrypt("")

    pages: list[Page] = []
    for number, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        text = clean_text(raw)
        if len(text) < 40:
            continue
        pages.append(Page(source=path.name, number=number, text=text))
    return pages


def load_papers(folder: Path) -> tuple[list[Page], list[str]]:
    """Return (pages, skipped file names)."""
    if not folder.is_dir():
        raise FileNotFoundError(f"Papers folder not found: {folder}")

    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF files in {folder}")

    pages: list[Page] = []
    skipped: list[str] = []
    for index, pdf in enumerate(pdfs, start=1):
        try:
            extracted = load_pdf(pdf)
        except Exception as exc:
            skipped.append(f"{pdf.name} ({exc.__class__.__name__})")
            print(f"  [{index}/{len(pdfs)}] skipped {pdf.name}", flush=True)
            continue
        if not extracted:
            skipped.append(f"{pdf.name} (no extractable text)")
            print(f"  [{index}/{len(pdfs)}] skipped {pdf.name}", flush=True)
            continue
        pages.extend(extracted)
        print(f"  [{index}/{len(pdfs)}] {pdf.name}: {len(extracted)} pages", flush=True)
    return pages, skipped
