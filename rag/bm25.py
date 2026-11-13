"""Okapi BM25 over the passage texts. No extra package."""

from __future__ import annotations

import math
import re

import numpy as np

_WORD = re.compile(r"[a-z0-9][a-z0-9\-]{2,}")
_STOP = frozenset("a an the and or of to for in on with from by at as is are was were be been it its this that".split())

_models: dict[int, "BM25"] = {}


def tokenize(text: str) -> list[str]:
    return [word for word in _WORD.findall(text.lower()) if word not in _STOP]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.tokens = [tokenize(doc) for doc in docs]
        self.n = len(self.tokens)
        lengths = [len(tokens) for tokens in self.tokens]
        self.avg = (sum(lengths) / self.n) if self.n else 1.0
        self.lengths = lengths
        df: dict[str, int] = {}
        for tokens in self.tokens:
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1
        self.df = df

    def idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log(1.0 + (self.n - n + 0.5) / (n + 0.5))

    def score(self, query: str) -> np.ndarray:
        terms = tokenize(query)
        scores = np.zeros(self.n, dtype=np.float32)
        if not terms or self.n == 0:
            return scores
        for index, tokens in enumerate(self.tokens):
            if not tokens:
                continue
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            length_norm = 1.0 - self.b + self.b * (self.lengths[index] / self.avg)
            total = 0.0
            for term in terms:
                freq = counts.get(term, 0)
                if not freq:
                    continue
                total += self.idf(term) * (freq * (self.k1 + 1)) / (freq + self.k1 * length_norm)
            scores[index] = total
        return scores


def score_texts(texts: list[str], query: str) -> np.ndarray:
    key = id(texts)
    model = _models.get(key)
    if model is None or model.n != len(texts):
        model = BM25(texts)
        _models[key] = model
    return model.score(query)
