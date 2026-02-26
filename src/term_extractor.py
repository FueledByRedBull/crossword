from __future__ import annotations

from dataclasses import dataclass, field


STOPWORDS_EN = {
    "A",
    "AN",
    "AND",
    "ARE",
    "AS",
    "AT",
    "BE",
    "BUT",
    "BY",
    "FOR",
    "FROM",
    "HAS",
    "HAVE",
    "HE",
    "HER",
    "HERS",
    "HIM",
    "HIS",
    "I",
    "IN",
    "INTO",
    "IS",
    "IT",
    "ITS",
    "ME",
    "MY",
    "NO",
    "NOT",
    "OF",
    "ON",
    "OR",
    "OUR",
    "SHE",
    "SO",
    "THAN",
    "THAT",
    "THE",
    "THEIR",
    "THEM",
    "THEN",
    "THERE",
    "THESE",
    "THEY",
    "THIS",
    "THOSE",
    "TO",
    "UP",
    "US",
    "WAS",
    "WE",
    "WERE",
    "WHAT",
    "WHEN",
    "WHERE",
    "WHICH",
    "WHO",
    "WHY",
    "WILL",
    "WITH",
    "YOU",
    "YOUR",
}

STOPWORDS_EL = {
    "KAI",
    "TO",
    "THN",
    "THS",
    "TON",
    "TA",
    "TOU",
    "ME",
    "GIA",
    "APO",
    "SE",
    "STO",
    "STHN",
    "STOUS",
    "STIS",
    "AUTO",
    "AUTH",
    "AUTOI",
    "AUTES",
    "TIS",
    "ALL",
    "ENA",
    "ENAN",
    "MIA",
    "OTI",
    "AN",
    "EAN",
    "OS",
}

from .text_normalize import normalize_text, strip_diacritics, tokenize

RARE_LETTERS = {"Q", "X", "Z", "J"}


@dataclass
class TermCandidate:
    answer: str
    normalized_answer: str
    length: int
    source_method: str
    lead_bold_signal: bool
    source_titles: set[str] = field(default_factory=set)


def _token_alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    alpha = sum(1 for ch in text if ch.isalpha())
    return alpha / max(1, len(text))


def _normalize_answer(text: str, *, lang: str = "en") -> str:
    normalized = normalize_text(text, lang=lang)
    if lang == "el":
        normalized = strip_diacritics(normalized)
        normalized = normalized.translate(
            str.maketrans(
                {
                    "Α": "A",
                    "Β": "B",
                    "Γ": "G",
                    "Δ": "D",
                    "Ε": "E",
                    "Ζ": "Z",
                    "Η": "H",
                    "Θ": "TH",
                    "Ι": "I",
                    "Κ": "K",
                    "Λ": "L",
                    "Μ": "M",
                    "Ν": "N",
                    "Ξ": "X",
                    "Ο": "O",
                    "Π": "P",
                    "Ρ": "R",
                    "Σ": "S",
                    "Τ": "T",
                    "Υ": "Y",
                    "Φ": "F",
                    "Χ": "CH",
                    "Ψ": "PS",
                    "Ω": "O",
                    "α": "a",
                    "β": "b",
                    "γ": "g",
                    "δ": "d",
                    "ε": "e",
                    "ζ": "z",
                    "η": "h",
                    "θ": "th",
                    "ι": "i",
                    "κ": "k",
                    "λ": "l",
                    "μ": "m",
                    "ν": "n",
                    "ξ": "x",
                    "ο": "o",
                    "π": "p",
                    "ρ": "r",
                    "σ": "s",
                    "ς": "s",
                    "τ": "t",
                    "υ": "y",
                    "φ": "f",
                    "χ": "ch",
                    "ψ": "ps",
                    "ω": "o",
                }
            )
        )
    return "".join(ch for ch in normalized if ch.isalpha()).upper()


def shape_penalty(normalized: str) -> float:
    if not normalized or len(normalized) < 3:
        return 0.0
    middle = normalized[1:-1]
    if not middle:
        return 0.0
    rare_count = sum(1 for ch in middle if ch in RARE_LETTERS)
    return rare_count / len(middle)


def crosswordability_score(normalized: str) -> float:
    return max(0.0, 1.0 - shape_penalty(normalized))

def _clean_answer_text(text: str) -> str:
    return " ".join(text.strip(" \t\r\n\"'()[]{}.,;:").split())


def extract_terms_spacy(doc, *, source_title: str, lang: str = "en") -> list[TermCandidate]:
    terms: list[TermCandidate] = []
    seen: set[str] = set()

    for chunk in getattr(doc, "noun_chunks", []):
        raw = _clean_answer_text(chunk.text)
        if not raw:
            continue
        if raw in seen:
            continue
        seen.add(raw)
        normalized = _normalize_answer(raw, lang=lang)
        if not normalized:
            continue
        terms.append(
            TermCandidate(
                answer=raw,
                normalized_answer=normalized,
                length=len(normalized),
                source_method="spacy",
                lead_bold_signal=False,
                source_titles={source_title},
            )
        )

    for ent in getattr(doc, "ents", []):
        raw = _clean_answer_text(ent.text)
        if not raw or raw in seen:
            continue
        seen.add(raw)
        normalized = _normalize_answer(raw, lang=lang)
        if not normalized:
            continue
        terms.append(
            TermCandidate(
                answer=raw,
                normalized_answer=normalized,
                length=len(normalized),
                source_method="spacy",
                lead_bold_signal=False,
                source_titles={source_title},
            )
        )

    return terms


def extract_terms_nltk(
    text: str,
    *,
    source_title: str,
    lang: str = "en",
) -> list[TermCandidate]:
    if lang != "en" or not text:
        return []
    try:
        import nltk
    except Exception:
        return []
    try:
        tokens = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(tokens)
    except Exception:
        return []

    terms: list[TermCandidate] = []
    seen: set[str] = set()

    grammar = r"NP: {<JJ.*>*<NN.*>+}"
    try:
        parser = nltk.RegexpParser(grammar)
        tree = parser.parse(tagged)
        for subtree in tree.subtrees(filter=lambda t: t.label() == "NP"):
            raw = _clean_answer_text(" ".join(word for word, _ in subtree.leaves()))
            if not raw or raw in seen:
                continue
            seen.add(raw)
            normalized = _normalize_answer(raw, lang=lang)
            if not normalized:
                continue
            terms.append(
                TermCandidate(
                    answer=raw,
                    normalized_answer=normalized,
                    length=len(normalized),
                    source_method="nltk",
                    lead_bold_signal=False,
                    source_titles={source_title},
                )
            )
    except Exception:
        return terms

    return terms


def extract_terms_lead_bold(
    terms: list[str],
    *,
    source_title: str,
    lang: str = "en",
) -> list[TermCandidate]:
    results: list[TermCandidate] = []
    for term in terms:
        raw = _clean_answer_text(term)
        if not raw:
            continue
        normalized = _normalize_answer(raw, lang=lang)
        if not normalized:
            continue
        results.append(
            TermCandidate(
                answer=raw,
                normalized_answer=normalized,
                length=len(normalized),
                source_method="lead_bold",
                lead_bold_signal=True,
                source_titles={source_title},
            )
        )
    return results


def merge_terms(
    term_lists: list[list[TermCandidate]],
    *,
    min_len: int = 4,
    max_len: int = 12,
    min_alpha_ratio: float = 0.8,
    stopwords: set[str] | None = None,
) -> list[TermCandidate]:
    merged: dict[str, TermCandidate] = {}
    stopwords = stopwords or set()
    for terms in term_lists:
        for term in terms:
            if term.length < min_len or term.length > max_len:
                continue
            if _token_alpha_ratio(term.answer) < min_alpha_ratio:
                continue
            key = term.normalized_answer
            if key in stopwords:
                continue
            if key in merged:
                existing = merged[key]
                existing.lead_bold_signal = existing.lead_bold_signal or term.lead_bold_signal
                if existing.source_method != term.source_method:
                    existing.source_method = "hybrid"
                existing.source_titles.update(term.source_titles)
                continue
            merged[key] = term
    return list(merged.values())


def term_frequency_across_docs(
    docs: list[str],
    *,
    lang: str = "en",
) -> dict[str, int]:
    df: dict[str, int] = {}
    for doc in docs:
        tokens = set(tokenize(doc, lang=lang))
        for term in tokens:
            if lang == "el":
                normalized = _normalize_answer(term, lang=lang)
            else:
                normalized = term.upper()
            if not normalized:
                continue
            df[normalized] = df.get(normalized, 0) + 1
    return df


def get_stopwords(lang: str) -> set[str]:
    return STOPWORDS_EN if lang == "en" else STOPWORDS_EL



