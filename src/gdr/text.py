"""Arabic text normalization shared by the guard, the leakage check and the evaluation."""

from __future__ import annotations

import re
import unicodedata

_DIACRITICS = re.compile("[ؐ-ًؚ-ٰٟۖ-ۭـ]")  # harakat and tatweel
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_NUMBER = re.compile(r"\d+(?:[.,٫٬]\d+)*")
_LATIN = re.compile(r"[A-Za-z][A-Za-z0-9_\-]*")


def normalize(text: str) -> str:
    """Case-fold, unify digits, strip diacritics and unify common Arabic letter variants."""
    t = unicodedata.normalize("NFKC", text or "").casefold().translate(_DIGITS)
    t = _DIACRITICS.sub("", t)
    t = re.sub("[إأآٱ]", "ا", t).replace("ى", "ي").replace("ة", "ه")
    t = re.sub(r"[^\w\s.]", " ", t)
    t = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", t)  # keep decimal points only
    return " ".join(t.split())


def tokens(text: str) -> list[str]:
    return normalize(text).split()


def numbers_in(text: str) -> list[str]:
    """Numbers in a text, canonical form (Arabic digits converted, thousands separators dropped), in order."""
    t = unicodedata.normalize("NFKC", text or "").translate(_DIGITS)
    out = []
    for m in _NUMBER.finditer(t):
        n = m.group().replace("٬", "").replace("٫", ".")
        # 1,234 is a thousands separator; 1,5 is not used in this domain, so treat any comma that way
        n = n.replace(",", "")
        out.append(n.rstrip("0").rstrip(".") if "." in n else n)
    return out


def latin_terms(text: str) -> list[str]:
    """Latin-script terms (codes, drug names, units such as HbA1c or BMI), case-folded, in order."""
    return [m.group().casefold() for m in _LATIN.finditer(unicodedata.normalize("NFKC", text or ""))]
