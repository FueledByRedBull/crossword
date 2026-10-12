from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from ..text_normalize import strip_diacritics, tokenize
from ..clue_builder import (
    BLANK,
    blank_is_part_of_a_name,
    answer_pattern,
    split_sentences,
    strip_trailing_sections,
)
from ..provenance import has_packageable_clue_provenance
from ..lexicon import lexicon_score_for_token
from ..wiki_client import WikiClient


@dataclass
class SeedStageResult:
    diagnostics_path: Path
    diagnostics: dict


@dataclass
class ScoringStageResult:
    scores_path: Path
    scores: list[dict]
    diagnostics_path: Path
    diagnostics: dict
    rel_scores: list[float]
    pairwise: list[list[float]]
    ranked_indices: list[int]
    term_scores: list[float]
    term_answer_inventories: list[set[str]]


@dataclass
class KSelectionStageResult:
    selected_path: Path
    trace_path: Path
    diagnostics_path: Path
    selected: dict
    trace: list[dict]
    diagnostics: dict


@dataclass
class TermExtractionStageResult:
    terms_path: Path
    diagnostics_path: Path
    terms: list[dict]
    diagnostics: dict


@dataclass
class VocabGateStageResult:
    diagnostics_path: Path
    diagnostics: dict


@dataclass
class ClueStageResult:
    clues_path: Path
    diagnostics_path: Path
    clues: list[dict]
    diagnostics: dict


@dataclass
class LayoutStageResult:
    grid_path: Path
    diagnostics_path: Path
    diagnostics: dict


@dataclass
class PackagingStageResult:
    puzzle_path: Path
    attribution_path: Path
    diagnostics_path: Path
    diagnostics: dict


def _safe_float(value, *, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, *, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _is_packageable_clue(clue_row: dict) -> bool:
    text = (clue_row.get("clue") or "").strip()
    return bool(text) and has_packageable_clue_provenance(clue_row)


def _clued_answer_count(clue_rows: list[dict], *, lang: str) -> int:
    """Distinct answers with a clue that can be packaged (a clue without a revid cannot)."""
    keys = {
        _answer_key(row.get("answer", ""), row.get("normalized_answer", ""), lang=lang)
        for row in clue_rows
        if _is_packageable_clue(row)
    }
    return len(keys - {""})


def _resolve_nlp_backend(
    *,
    lang: str,
    nlp_backend: str = "auto",
) -> tuple[object | None, str | None, list[str], str | None, str | None]:
    errors: list[str] = []
    if nlp_backend.lower() not in {"auto", "spacy"}:
        errors.append(f"unsupported_nlp_backend:{nlp_backend}")

    nlp = None
    backend_used = None
    spacy_version = None
    spacy_model_version = None

    try:
        import spacy
    except ImportError as exc:  # pragma: no cover
        errors.append(f"spacy_missing:{exc}")
        spacy = None

    if spacy is not None:
        model_name = "en_core_web_sm" if lang == "en" else "el_core_news_sm"
        try:
            nlp = spacy.load(model_name)
            backend_used = "spacy"
        except OSError as exc:  # pragma: no cover
            errors.append(f"spacy_model_load_failed:{model_name}:{exc}")

        spacy_version = getattr(spacy, "__version__", None)
        if nlp is not None:
            spacy_model_version = nlp.meta.get("version")

    return nlp, backend_used, errors, spacy_version, spacy_model_version


def _fetch_extract_with_revid(errors: list[str], client: WikiClient, title: str, *, full: bool) -> dict:
    """Extract and revision id from one response: the full article if `full` and available, else the intro.

    Nothing is paired across responses: a revision id is only the one the text arrived with.
    """
    for intro_only in (False, True) if full else (True,):
        try:
            payload = client.fetch_page_extract_with_revid(title, intro_only=intro_only)
        except RuntimeError as exc:
            errors.append(f"extract_with_revid_failed:{title}:{exc}")
            continue
        if payload.get("extract") and payload.get("revid"):
            return {**payload, "extract": strip_trailing_sections(payload["extract"])}
    return {"title": title, "page_id": None, "extract": "", "revid": None}


def _prefetch(errors: list[str], prefetch, titles: list[str], **kwargs) -> None:
    """Batch-fill the cache; the single-title getters still work if this fails."""
    try:
        prefetch(titles, **kwargs)
    except RuntimeError as exc:
        errors.append(f"prefetch_failed:{exc}")


def _uses_seed_word(text: str, seed_title: str, *, lang: str, lexicon: dict[str, float], derived: bool = False) -> bool:
    seed_words = [word for word in tokenize(seed_title, lang=lang) if len(word) >= 4]
    scores = [lexicon_score_for_token(word, lexicon) for word in seed_words]
    if len(set(scores)) > 1:  # the rarest word names the topic: "Rome", not "ancient", in "Ancient Rome"
        seed_words = [seed_words[scores.index(min(scores))]]
    # A shared stem, not shared letters: thermodynamic/thermodynamics, but not thermostat or Romania/Rome.
    # `derived` also takes a short stem with at most two more letters: roman/rome.
    return any(
        len(os.path.commonprefix([word, seed])) >= max(3 if derived else 4, len(seed) - 1)
        and (not derived or len(word) <= len(seed) + 2)
        for word in set(tokenize(text, lang=lang))
        for seed in seed_words
    )


def _name_mentions(name: str, text: str, *, part_of: tuple[str, ...] = (), mid_sentence: bool = False) -> int:
    """Capitalised mentions of `name` that stand alone: outside the phrases `part_of` and next to no other
    capitalised word ("Camille Pissarro" does not name "Camille").

    With `mid_sentence` a mention that opens a sentence or a line does not count: any word has a capital there.
    """
    inside = [match.span() for phrase in part_of for match in answer_pattern(phrase).finditer(text)]
    count = 0
    for match in answer_pattern(name).finditer(text):
        if not match.group()[0].isupper() or any(start <= match.start() < end for start, end in inside):
            continue
        before = split_sentences(text[max(0, match.start() - 300) : match.start()] + BLANK)[-1]
        if blank_is_part_of_a_name(before + text[match.end() : match.end() + 3]):
            continue
        count += not (mid_sentence and before.rsplit("\n", 1)[-1].strip() == BLANK)
    return count


def _answer_key(answer: str, normalized_answer: str, *, lang: str) -> str:
    raw = (normalized_answer or answer or "").strip().upper()
    if not raw:
        return ""
    if lang == "en":
        return "".join(ch for ch in raw if "A" <= ch <= "Z")
    return "".join(ch for ch in raw if ch.isalpha())


def _read_csv_rows(path: str | Path) -> list[dict]:
    import csv

    if not Path(path).exists():
        return []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _shared_words(display: str) -> set[str]:
    """Case-folded, accent-free words of 4+ letters; two answers sharing one cannot share a grid."""
    words = re.findall(r"[^\W\d_]+", strip_diacritics(display).casefold())
    return {word for word in words if len(word) >= 4}
