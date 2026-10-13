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


def enumeration(display: str) -> str:
    """Crossword letter-count notation: "Rock and roll" -> "4,3,4", "Synth-pop" -> "5-3"."""
    joined = re.sub(r"(?<=\w)['\u2019](?=\w)", "", display)  # "Bell's" counts as 5
    counted = re.sub(r"[^\W\d_]+", lambda match: str(len(match.group())), joined)
    return re.sub(r"\s+", ",", re.sub(r"[^\d\s-]", "", counted).strip())


# Sexual content and sexual abuse or exploitation, for a general audience. A page about war, slavery, genocide,
# execution, death or disease stays: those are subjects of history, art and medicine. Each entry is a regex fragment
# matched on the accent-free, lower-case words, at the start of a word (`_SENSITIVE_STEMS`) or as a whole word
# (`_SENSITIVE_WORDS`): "rape" is no "rapeseed" or "grape", "sex" no "Sextus".
#
# A keyword list is crude. It knows nothing about a page that avoids these words, and it blocks a page that merely
# holds one (a first sentence that calls rapeseed "rape"). "Rape of ..." is let through: it is the idiom for an
# abduction or a sack (the Sabine women, Europa, Belgium), so a page about a real rape titled so passes as well. The
# upgrade is the page's Wikipedia categories (Category:Child sexual abuse, Category:Pornography) via the API.
_SENSITIVE_STEMS = {
    "en": (
        r"pornograph|revenge porn|paedophil|pedophil|incest|prostitut|molest|child\w* abuse|sexually explicit"
        r"|sex(?:ual\w*)? (?:abus|assault|exploit|violen|slave|traffick|tourism|crime|offen)"
    ),
    "el": (
        r"βιασμ|πορνογραφ|πορνη|πορνε|εκπορνευ|αιμομιξ|παιδεραστ|παιδοφιλ|κτηνοβασ"
        r"|παιδικ\w* κακοποι|σεξουαλικ\w* (?:κακοποι|εκμεταλλ|βια|επιθεσ|δουλει|τουρισμ|εγκληματ)"
    ),
}
_SENSITIVE_WORDS = {
    "en": r"rape(?! of)|rapes|raped|raping|rapist|rapists|hentai|bestiality|zoophilia|necrophilia",
    "el": r"βιαστης|βιαστες",
}
_SENSITIVE_RE = {
    lang: re.compile(rf"\b(?:{_SENSITIVE_STEMS[lang]})|\b(?:{_SENSITIVE_WORDS[lang]})\b") for lang in _SENSITIVE_STEMS
}


def is_sensitive(text: str, lang: str, seed_title: str = "") -> bool:
    """Whether `text` holds a sensitive word. A seed whose own title does is the user's request: nothing is blocked."""
    if seed_title and is_sensitive(seed_title, lang):
        return False
    return bool(_SENSITIVE_RE[lang].search(" ".join(tokenize(text, lang=lang))))
