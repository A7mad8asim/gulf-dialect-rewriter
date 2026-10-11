"""Keep the downstream test questions out of the training data.

The end-to-end evaluation runs ask-the-data's gold set, so none of its questions (in either
language) may reach training, not even as a near-duplicate. A training pair is flagged when its
source or target matches a held-out question exactly after normalization, or shares most of its
word tokens with one.
"""

from __future__ import annotations

import json
from pathlib import Path

from .text import normalize, tokens

JACCARD_THRESHOLD = 0.6


def load_questions(path: str | Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line)["question"] for line in f if line.strip()]


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


class LeakageChecker:
    def __init__(self, held_out: list[str], threshold: float = JACCARD_THRESHOLD):
        self.threshold = threshold
        self._exact = {normalize(q) for q in held_out}
        self._token_sets = [set(tokens(q)) for q in held_out]

    def leaks(self, text: str) -> bool:
        if normalize(text) in self._exact:
            return True
        t = set(tokens(text))
        return any(_jaccard(t, h) >= self.threshold for h in self._token_sets)

    def filter_pairs(self, pairs: list[dict]) -> tuple[list[dict], list[dict]]:
        """Split pairs ({'source', 'target', ...}) into (kept, dropped)."""
        kept, dropped = [], []
        for p in pairs:
            (dropped if self.leaks(p["source"]) or self.leaks(p["target"]) else kept).append(p)
        return kept, dropped
