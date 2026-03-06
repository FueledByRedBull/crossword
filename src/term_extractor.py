from __future__ import annotations

import re
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

LEADING_DETERMINERS_EN = {
    "A",
    "AN",
    "THE",
    "THIS",
    "THAT",
    "THESE",
    "THOSE",
}

from .text_normalize import normalize_text, strip_diacritics, tokenize

RARE_LETTERS = {"Q", "X", "Z", "J"}
MAX_NLP_TERM_TOKENS = 3


@dataclass(slots=True)
class TermCandidate:
    answer: str
    normalized_answer: str
    length: int
    source_method: str
    lead_bold_signal: bool
    source_titles: set[str] = field(default_factory=set)
    token_count: int = 1
    answer_cleanliness_score: float = 1.0
    clueability_score: float = 1.0


@dataclass(slots=True)
class MergeDiagnostics:
    dirty_phrase_reject_count: int = 0
    multiword_reject_count: int = 0
    stopword_boundary_reject_count: int = 0
    low_cleanliness_reject_count: int = 0
    promoted_by_lead_bold_count: int = 0
    promoted_by_source_diversity_count: int = 0


def _token_alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    alpha = sum(1 for ch in text if ch.isalpha())
    return alpha / max(1, len(text))


def _answer_tokens(text: str) -> list[str]:
    return [token for token in text.replace("’", "'").split() if token]


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


def answer_cleanliness_score(answer: str, *, lang: str = "en") -> float:
    if not answer:
        return 0.0
    if lang != "en":
        return 1.0

    tokens = _answer_tokens(answer)
    if not tokens:
        return 0.0

    score = 1.0
    if len(tokens) > MAX_NLP_TERM_TOKENS:
        score -= 0.35 + (0.1 * (len(tokens) - MAX_NLP_TERM_TOKENS))
    if tokens[0].upper() in STOPWORDS_EN:
        score -= 0.35
    if len(tokens) > 1 and tokens[-1].upper() in STOPWORDS_EN:
        score -= 0.25
    if re.search(r"(?:'s|’s)\b", answer.lower()):
        score -= 0.35
    if any(token.upper() in {"AND", "OR", "OF", "IN", "TO", "WITH", "FOR"} for token in tokens[1:-1]):
        score -= 0.12
    if any(len(token) == 1 for token in tokens):
        score -= 0.08
    alpha_ratio = _token_alpha_ratio(answer)
    if alpha_ratio < 0.9:
        score -= min(0.3, 0.3 * ((0.9 - alpha_ratio) / 0.9))
    return max(0.0, min(1.0, score))


def clueability_score(answer: str, normalized: str, *, lang: str = "en") -> float:
    if not normalized:
        return 0.0
    tokens = _answer_tokens(answer)
    token_score = 1.0 if len(tokens) <= 1 else 0.8 if len(tokens) == 2 else 0.55
    length = len(normalized)
    if 4 <= length <= 9:
        length_score = 1.0
    elif length <= 12:
        length_score = 0.85
    else:
        length_score = 0.65
    cleanliness = answer_cleanliness_score(answer, lang=lang)
    crossword_score = crosswordability_score(normalized)
    return max(
        0.0,
        min(
            1.0,
            (0.4 * cleanliness)
            + (0.35 * token_score)
            + (0.15 * crossword_score)
            + (0.10 * length_score),
        ),
    )


def _clean_answer_text(text: str, *, lang: str = "en") -> str:
    cleaned = " ".join(text.strip(" \t\r\n\"'()[]{}.,;:").split())
    if not cleaned or lang != "en":
        return cleaned
    parts = cleaned.split()
    while parts and parts[0].upper() in LEADING_DETERMINERS_EN:
        parts = parts[1:]
    return " ".join(parts)


def _build_term_candidate(
    text: str,
    *,
    source_title: str,
    source_method: str,
    lead_bold_signal: bool,
    lang: str = "en",
) -> TermCandidate | None:
    raw = _clean_answer_text(text, lang=lang)
    if not raw:
        return None
    normalized = _normalize_answer(raw, lang=lang)
    if not normalized:
        return None
    return TermCandidate(
        answer=raw,
        normalized_answer=normalized,
        length=len(normalized),
        source_method=source_method,
        lead_bold_signal=lead_bold_signal,
        source_titles={source_title},
        token_count=len(_answer_tokens(raw)),
        answer_cleanliness_score=answer_cleanliness_score(raw, lang=lang),
        clueability_score=clueability_score(raw, normalized, lang=lang),
    )


def _surface_rank(term: TermCandidate) -> tuple[int, float, float, int, int, int, str]:
    return (
        1 if term.lead_bold_signal else 0,
        term.answer_cleanliness_score,
        term.clueability_score,
        len(term.source_titles),
        -term.token_count,
        -len(term.answer),
        term.answer,
    )


def extract_terms_spacy(doc, *, source_title: str, lang: str = "en") -> list[TermCandidate]:
    terms: list[TermCandidate] = []
    seen: set[str] = set()

    for chunk in getattr(doc, "noun_chunks", []):
        candidate = _build_term_candidate(
            chunk.text,
            source_title=source_title,
            source_method="spacy",
            lead_bold_signal=False,
            lang=lang,
        )
        if candidate is None:
            continue
        if candidate.answer in seen:
            continue
        seen.add(candidate.answer)
        terms.append(candidate)

    for ent in getattr(doc, "ents", []):
        candidate = _build_term_candidate(
            ent.text,
            source_title=source_title,
            source_method="spacy",
            lead_bold_signal=False,
            lang=lang,
        )
        if candidate is None or candidate.answer in seen:
            continue
        seen.add(candidate.answer)
        terms.append(candidate)

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
            candidate = _build_term_candidate(
                " ".join(word for word, _ in subtree.leaves()),
                source_title=source_title,
                source_method="nltk",
                lead_bold_signal=False,
                lang=lang,
            )
            if candidate is None or candidate.answer in seen:
                continue
            seen.add(candidate.answer)
            terms.append(candidate)
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
        candidate = _build_term_candidate(
            term,
            source_title=source_title,
            source_method="lead_bold",
            lead_bold_signal=True,
            lang=lang,
        )
        if candidate is None:
            continue
        results.append(candidate)
    return results


def merge_terms(
    term_lists: list[list[TermCandidate]],
    *,
    lang: str = "en",
    min_len: int = 4,
    max_len: int = 12,
    min_alpha_ratio: float = 0.8,
    stopwords: set[str] | None = None,
    return_stats: bool = False,
) -> list[TermCandidate] | tuple[list[TermCandidate], MergeDiagnostics]:
    merged: dict[str, TermCandidate] = {}
    source_titles_by_key: dict[str, set[str]] = {}
    diagnostics = MergeDiagnostics()
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
            source_titles_by_key.setdefault(key, set()).update(term.source_titles)
            if lang == "en" and not term.lead_bold_signal:
                tokens = _answer_tokens(term.answer)
                if term.token_count > MAX_NLP_TERM_TOKENS:
                    diagnostics.multiword_reject_count += 1
                    diagnostics.dirty_phrase_reject_count += 1
                    continue
                if tokens and (
                    tokens[0].upper() in STOPWORDS_EN
                    or (len(tokens) > 1 and tokens[-1].upper() in STOPWORDS_EN)
                ):
                    diagnostics.stopword_boundary_reject_count += 1
                    diagnostics.dirty_phrase_reject_count += 1
                    continue
                cleanliness_floor = 0.8 if term.token_count > 1 else 0.6
                if term.answer_cleanliness_score < cleanliness_floor:
                    diagnostics.low_cleanliness_reject_count += 1
                    diagnostics.dirty_phrase_reject_count += 1
                    continue
            if key in merged:
                existing = merged[key]
                replace_surface = _surface_rank(term) > _surface_rank(existing)
                existing.lead_bold_signal = existing.lead_bold_signal or term.lead_bold_signal
                if existing.source_method != term.source_method:
                    existing.source_method = "hybrid"
                existing.source_titles.update(term.source_titles)
                if replace_surface:
                    existing.answer = term.answer
                    existing.token_count = term.token_count
                    existing.answer_cleanliness_score = term.answer_cleanliness_score
                    existing.clueability_score = term.clueability_score
                continue
            merged[key] = term
    for key, term in merged.items():
        term.source_titles.update(source_titles_by_key.get(key, set()))
    merged_terms = list(merged.values())
    diagnostics.promoted_by_lead_bold_count = sum(
        1 for term in merged_terms if term.lead_bold_signal
    )
    diagnostics.promoted_by_source_diversity_count = sum(
        1 for term in merged_terms if len(term.source_titles) >= 2
    )
    if return_stats:
        return merged_terms, diagnostics
    return merged_terms


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



