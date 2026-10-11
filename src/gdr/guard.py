"""The design rule, enforced in code: a rewrite may change wording, never content.

Every number and Latin-script term in the dialect question must appear in the rewrite exactly as
often as in the original, and the rewrite may not add any. A rewrite that breaks the rule, comes
back empty or is far longer than the question is discarded, and the original question is used.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .text import latin_terms, numbers_in, normalize

MAX_LENGTH_RATIO = 2.0  # a rewrite more than twice as long as the question is treated as a runaway generation


@dataclass
class GuardResult:
    text: str  # the text to pass downstream: the rewrite if accepted, otherwise the original
    accepted: bool
    reasons: list[str] = field(default_factory=list)


def _diff(original: list[str], rewrite: list[str]) -> tuple[list[str], list[str]]:
    a, b = Counter(original), Counter(rewrite)
    return sorted((a - b).elements()), sorted((b - a).elements())


def check(question: str, rewrite: str) -> GuardResult:
    reasons = []
    if not normalize(rewrite):
        reasons.append("empty rewrite")
    elif len(rewrite) > MAX_LENGTH_RATIO * max(len(question), 1):
        reasons.append("rewrite much longer than the question")

    lost, added = _diff(numbers_in(question), numbers_in(rewrite))
    if lost:
        reasons.append(f"numbers lost: {', '.join(lost)}")
    if added:
        reasons.append(f"numbers added: {', '.join(added)}")

    lost, added = _diff(latin_terms(question), latin_terms(rewrite))
    if lost:
        reasons.append(f"terms lost: {', '.join(lost)}")
    if added:
        reasons.append(f"terms added: {', '.join(added)}")

    if reasons:
        return GuardResult(text=question, accepted=False, reasons=reasons)
    return GuardResult(text=rewrite.strip(), accepted=True)
