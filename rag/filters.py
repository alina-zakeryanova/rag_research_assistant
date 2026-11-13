"""Drop title-page boilerplate and bibliography chunks from search."""

from __future__ import annotations

import re


def is_reference(text: str) -> bool:
    years = len(re.findall(r"\b(?:19|20)\d{2}\b", text))
    initials = len(re.findall(r"\b[A-Z]\s*\.", text))
    if years >= 5 or (years >= 2 and initials >= 6):
        return True
    lowered = text.lower()
    if lowered.count("crossref") >= 2:
        return True
    return lowered.count("doi") >= 2 or lowered.count("et al") >= 3


def is_front_matter(text: str, page: int) -> bool:
    if page != 1:
        return False
    head = text[:500].lower()
    markers = ("e-mail", "email", "received", "accepted", "copyright", "corresponding")
    return "@" in head or any(marker in head for marker in markers)
