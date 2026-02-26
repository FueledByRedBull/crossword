from __future__ import annotations

import re
import unicodedata

_WORD_RE = re.compile(r"[A-Za-z]+")
_WORD_RE_EL = re.compile(r"[Α-Ωα-ω]+")


def strip_diacritics(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def normalize_text(text: str, *, lang: str = "en") -> str:
    if lang == "el":
        return strip_diacritics(text)
    return text


def tokenize(text: str, *, lang: str = "en") -> list[str]:
    cleaned = normalize_text(text, lang=lang).lower()
    if lang == "el":
        return _WORD_RE_EL.findall(cleaned)
    return _WORD_RE.findall(cleaned)

