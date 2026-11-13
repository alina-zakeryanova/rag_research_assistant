"""A retrieved passage."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Hit:
    source: str
    page: int
    text: str
    score: float
