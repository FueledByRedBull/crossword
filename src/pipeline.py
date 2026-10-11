from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .cache import DiskCache
from .artifact_models import GridArtifact, PuzzleArtifact, SelectedCandidatesArtifact
from .diagnostics import build_seed_stage_diagnostics, write_json
from .k_selector import select_k
from .semantic import build_pairwise_matrix, build_tfidf_vectors, cosine_similarity, mmr_rank
from .term_extractor import (
    answer_cleanliness_score,
    clueability_score,
    extract_terms_lead_bold,
    extract_terms_nltk,
    extract_terms_spacy,
    extract_terms_title_tokens,
    GENERIC_GRID_JUNK_EN,
    _TITLE_DISAMBIGUATOR,
    merge_plural_duplicates,
    merge_terms,
    shape_penalty,
    crosswordability_score,
    term_frequency_across_docs,
    title_short_name,
    title_to_answer,
)
from .layout import Placement, build_layout, layout_metrics, number_placements, validate_layout
from .text_normalize import enumeration, strip_diacritics, tokenize
from .vocab_gate import VocabGateResult, evaluate_vocab_gate
from .clue_builder import (
    BLANK,
    MAX_CLUE_WORDS,
    blank_is_part_of_a_name,
    answer_pattern,
    clue_pass_mask_trim,
    clue_pass_validate,
    definition_clue,
    enforce_diversity,
    follow_up_definition,
    has_verb,
    is_generic_definition,
    lead_subject,
    lead_word_counts,
    split_sentences,
    strip_trailing_sections,
)
from .provenance import build_oldid_url, has_packageable_clue_provenance, validate_clue_provenance
from .wikidata_client import WikidataClient
from .lexicon import load_lexicon_scores, lexicon_score_for_token, lexicon_score_for_tokens
from .wiki_client import WikiClient


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


def _infer_clue_class(row: dict) -> str:
    clue_class = (row.get("clue_class") or "").strip().lower()
    if clue_class in {"source_backed", "template_fallback", "synthetic_filler"}:
        return clue_class

    source_method = (row.get("source_method") or "").strip().lower()
    if source_method == "package_filler_fallback":
        return "synthetic_filler"

    sentence_offset = _safe_int(row.get("sentence_offset"), default=-1)
    has_provenance = bool((row.get("revid") or "").strip() or (row.get("oldid_url") or "").strip())
    if sentence_offset >= 0 and has_provenance:
        return "source_backed"
    if sentence_offset == -1:
        return "template_fallback"
    if has_provenance:
        return "source_backed"
    return "template_fallback"


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


def _normalized_token_set(text: str, *, lang: str) -> set[str]:
    return {token.upper() for token in tokenize(text, lang=lang) if token}


def _title_support_score(answer: str, source_titles: set[str], *, lang: str) -> float:
    answer_tokens = _normalized_token_set(answer, lang=lang)
    if not answer_tokens or not source_titles:
        return 0.0
    best = 0.0
    for title in source_titles:
        title_tokens = _normalized_token_set(title, lang=lang)
        if not title_tokens:
            continue
        overlap = len(answer_tokens & title_tokens)
        if overlap == 0:
            continue
        score = overlap / max(1, len(answer_tokens))
        if score > best:
            best = score
        if best >= 1.0:
            break
    return min(1.0, best)


def _inventory_min_k_target(
    *,
    size: int,
    ranked_indices: list[int],
    term_answer_inventories: list[set[str]],
    inventory_min_df: int = 2,
) -> int | None:
    if size < 15 or not ranked_indices or not term_answer_inventories:
        return None

    short_targets = {4: 12, 5: 12}
    inventory_counts: dict[str, int] = {}
    for position, idx in enumerate(ranked_indices, start=1):
        for answer in term_answer_inventories[idx]:
            inventory_counts[answer] = inventory_counts.get(answer, 0) + 1
        inventory_by_length = {4: set(), 5: set()}
        for answer, count in inventory_counts.items():
            if count < inventory_min_df:
                continue
            if len(answer) in inventory_by_length:
                inventory_by_length[len(answer)].add(answer)
        if all(len(inventory_by_length[length]) >= target for length, target in short_targets.items()):
            return position
    return len(ranked_indices)


def _effective_min_k_for_size(
    *,
    size: int,
    min_k: int,
    candidate_count: int,
    ranked_indices: list[int] | None = None,
    term_answer_inventories: list[set[str]] | None = None,
    inventory_min_df: int = 2,
) -> int:
    recommended_min_k = min_k
    inventory_min_k = _inventory_min_k_target(
        size=size,
        ranked_indices=ranked_indices or [],
        term_answer_inventories=term_answer_inventories or [],
        inventory_min_df=inventory_min_df,
    )
    if inventory_min_k is not None:
        if size >= 15:
            inventory_min_k = min(inventory_min_k, 12)
        recommended_min_k = max(recommended_min_k, inventory_min_k)
    elif size >= 15:
        recommended_min_k = max(recommended_min_k, 12)
    return min(max(1, recommended_min_k), max(1, candidate_count))


def _resolve_nlp_backend(
    *,
    lang: str,
    nlp_backend: str = "auto",
) -> tuple[object | None, str | None, list[str], str | None, str | None]:
    errors: list[str] = []
    backend = nlp_backend.lower()
    if backend not in {"auto", "spacy", "nltk"}:
        errors.append(f"unsupported_nlp_backend:{nlp_backend}")
        backend = "auto"

    nlp = None
    backend_used = None
    spacy_version = None
    spacy_model_version = None

    if backend in {"auto", "spacy"}:
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
            except Exception as exc:  # pragma: no cover
                errors.append(f"spacy_model_load_failed:{model_name}:{exc}")

            spacy_version = getattr(spacy, "__version__", None)
            if nlp is not None:
                try:
                    spacy_model_version = nlp.meta.get("version")
                except Exception:  # pragma: no cover
                    pass

    if backend_used is None and backend in {"auto", "nltk"}:
        try:
            import nltk  # noqa: F401

            backend_used = "nltk"
        except Exception as exc:  # pragma: no cover
            errors.append(f"nltk_unavailable:{exc}")

    return nlp, backend_used, errors, spacy_version, spacy_model_version


def _extract_answer_inventory(
    *,
    title: str,
    extract: str,
    lead_bold_terms: list[str],
    lang: str,
    nlp: object | None,
    backend_used: str | None,
    min_len: int = 4,
    max_len: int = 12,
    min_alpha_ratio: float = 0.8,
) -> set[str]:
    from .term_extractor import get_stopwords

    term_lists: list[list] = []
    if backend_used == "spacy" and nlp is not None:
        term_lists.append(extract_terms_spacy(nlp(extract), source_title=title, lang=lang))
    elif backend_used == "nltk":
        term_lists.append(extract_terms_nltk(extract, source_title=title, lang=lang))
    term_lists.append(extract_terms_lead_bold(lead_bold_terms, source_title=title, lang=lang))
    term_lists.append(extract_terms_title_tokens(title, source_title=title, lang=lang))
    merged = merge_terms(
        term_lists,
        lang=lang,
        min_len=min_len,
        max_len=max_len,
        min_alpha_ratio=min_alpha_ratio,
        stopwords=get_stopwords(lang),
    )
    return {term.normalized_answer for term in merged if term.normalized_answer}


def _fetch_extract_with_revid(client: WikiClient, title: str, *, full: bool) -> dict:
    """Extract and revision id from one response: the full article if `full` and available, else the intro.

    Nothing is paired across responses: a revision id is only the one the text arrived with.
    """
    for intro_only in (False, True) if full else (True,):
        try:
            payload = client.fetch_page_extract_with_revid(title, intro_only=intro_only)
        except Exception:
            continue
        if payload.get("extract") and payload.get("revid"):
            return {**payload, "extract": strip_trailing_sections(payload["extract"])}
    return {"title": title, "page_id": None, "extract": "", "revid": None}


def _prefetch(errors: list[str], prefetch, titles: list[str], **kwargs) -> None:
    """Batch-fill the cache; the single-title getters still work if this fails."""
    try:
        prefetch(titles, **kwargs)
    except Exception as exc:
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


@lru_cache(maxsize=1)
def _plain_words(text: str) -> str:
    return " ".join(re.findall(r"\w+", strip_diacritics(text).casefold()))


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


def _seed_mentions(title: str, seed_text: str, *, lang: str, intro: str = "") -> int:
    """How often the seed article names the page. `intro`, the page's own, tells a proper name from a plain word."""
    phrase = _TITLE_DISAMBIGUATOR.sub("", title).strip()
    if lang != "el":
        pattern = answer_pattern(phrase)
        own = [match for sentence in split_sentences(intro) for match in pattern.finditer(sentence)]
        # A proper name is capitalised in mid-sentence and never in lower case on its own page ("Nature", the
        # journal). Only a capitalised mention names it then: "the wave nature of light" does not.
        if any(m.start() and m.group()[0].isupper() for m in own) and not any(m.group().islower() for m in own):
            return _name_mentions(phrase, seed_text)
        return len(pattern.findall(seed_text))
    # Greek inflects: each title word matches a text word that starts with its first max(4, len - 2) letters
    # and adds at most three more, so a stem is not a substring ("ύλη" is not "υλικό").
    stems = [word[: max(4, len(word) - 2)] for word in re.findall(r"\w+", strip_diacritics(phrase).casefold())]
    if not stems:
        return 0
    pattern = r"(?<!\w)" + " ".join(re.escape(stem) + r"\w{0,3}" for stem in stems) + r"(?!\w)"
    return len(re.findall(pattern, _plain_words(seed_text)))


def _answer_key(answer: str, normalized_answer: str, *, lang: str) -> str:
    raw = (normalized_answer or answer or "").strip().upper()
    if not raw:
        return ""
    if lang == "en":
        return "".join(ch for ch in raw if "A" <= ch <= "Z")
    return "".join(ch for ch in raw if ch.isalpha())


def _build_clue_row(
    *,
    answer: str,
    normalized_answer: str,
    answer_key_value: str,
    clue_text: str,
    clue_class: str,
    source_method: str,
    source_page: str,
    revid: int | None,
    sentence_offset: int | None,
    lang: str,
    clue_style: str,
) -> dict:
    clue_word_count = len(clue_text.split())
    clue_score = round(1.0 - abs(clue_word_count - 9) / max(9, clue_word_count), 4)
    return {
        "answer": answer,
        "normalized_answer": normalized_answer or answer_key_value,
        "clue": clue_text,
        "clue_score": clue_score,
        "clue_class": clue_class,
        "clue_style": clue_style,
        "source_method": source_method,
        "source_page": source_page,
        "revid": revid,
        "sentence_offset": sentence_offset,
        "oldid_url": build_oldid_url(source_page, revid, lang=lang),
    }


def _clue_candidate_rank(row: dict, seed_title: str) -> tuple[bool, bool, float, int, int, str]:
    clue_text = str(row.get("clue") or "")
    sentence_offset = _safe_int(row.get("sentence_offset"), default=9999)
    masked_length_score = -abs(len(clue_text.split()) - 9)
    return (
        not clue_text.startswith("...") and not clue_text.endswith("..."),
        row.get("source_page") == seed_title,
        _safe_float(row.get("clue_score"), default=0.0),
        masked_length_score,
        -sentence_offset,
        str(row.get("source_page") or ""),
    )


def _collect_source_backed_clue_candidates(
    *,
    answer: str,
    normalized_answer: str,
    source_titles: list[str],
    source_payloads: dict[str, dict],
    seed_title: str,
    term_source_method: str,
    lang: str,
    min_words: int,
    max_words: int,
    nlp: object | None,
    allow_fragments: bool = True,
    full_name: str | None = None,
) -> list[dict]:
    """Fill-in candidates, complete sentences first, then the seed article's before other pages'.

    A fragment is a window that is not a whole sentence and not a single clause-boundary cut; the
    term stage still counts them as support, a clue never uses one. With `full_name` the answer is a
    surname and the blank may stand for the whole name: "hired ____ as" from "hired Wayne Shorter as".
    """
    key = _answer_key(answer, normalized_answer, lang=lang)
    if not key:
        return []

    pattern = answer_pattern(answer)
    named = answer_pattern(full_name) if full_name else None
    candidates: list[dict] = []
    seen: set[tuple[str, int, str]] = set()
    for title in source_titles:
        payload = source_payloads.get(title, {})
        source_title = payload.get("title", title) or title
        extract = payload.get("extract", "")
        revid = payload.get("revid")
        for sentence_offset, candidate_sentence in enumerate(split_sentences(extract)):
            if named is not None:  # another person's name ("Alice Coltrane") stays, and still spoils the blank
                candidate_sentence = named.sub(answer, candidate_sentence)
            if not pattern.search(candidate_sentence):
                continue
            # Headings and link lists are glued to a sentence by line breaks; the clue is the line with the answer.
            candidate_sentence = max(
                (line for line in candidate_sentence.splitlines() if pattern.search(line)),
                key=len,
                default=candidate_sentence,
            )
            clue_text, clean = clue_pass_mask_trim(candidate_sentence, answer, max_words=max_words)
            if not (clean or allow_fragments):
                continue
            if not clue_pass_validate(
                clue_text,
                answer,
                min_words=min_words,
                nlp=nlp,
                original_sentence=candidate_sentence,
                require_blank=True,
                lang=lang,
            ):
                continue
            signature = (source_title, sentence_offset, clue_text)
            if signature in seen:
                continue
            seen.add(signature)
            candidates.append(
                _build_clue_row(
                    answer=answer,
                    normalized_answer=normalized_answer,
                    answer_key_value=key,
                    clue_text=clue_text,
                    clue_class="source_backed",
                    source_method=term_source_method,
                    source_page=source_title,
                    revid=revid,
                    sentence_offset=sentence_offset,
                    lang=lang,
                    clue_style="fill_in_blank",
                )
            )

    candidates.sort(key=lambda row: _clue_candidate_rank(row, seed_title), reverse=True)
    return candidates


def _definition_clue_rows(
    *,
    answer: str,
    normalized_answer: str,
    answer_key_value: str,
    title: str,
    payload: dict,
    term_source_method: str,
    lang: str,
    nlp: object | None,
    lead_counts: Counter[str],
    pool_size: int,
    seed_title: str,
) -> list[dict]:
    """Clue an article-title answer with its own article's lead sentence, subject removed.

    The second, longer cut is there for generic definitions ("An American physicist"): it may
    reach the relative clause that says who the person was. A definition that stays generic is dropped.
    When none is left, the second sentence is tried ("He was one of the most influential figures in jazz piano.").
    """
    sentences = split_sentences(payload.get("extract", ""))
    rows: list[dict] = []
    # The second sentence is the last resort of an answer whose first one gave no definition.
    for offset, cut in enumerate((definition_clue, follow_up_definition)):
        if rows or offset >= len(sentences):
            break
        texts: list[str] = []
        for max_words in (16, MAX_CLUE_WORDS):
            clue_text = cut(sentences[offset], answer=answer, max_words=max_words, nlp=nlp, lang=lang)
            if (
                clue_text
                and clue_text not in texts
                and clue_pass_validate(clue_text, answer, min_words=3, nlp=nlp, lang=lang)
                and not is_generic_definition(clue_text, lead_counts, pool_size, nlp, seed_title=seed_title)
            ):
                texts.append(clue_text)
        rows = [
            _build_clue_row(
                answer=answer,
                normalized_answer=normalized_answer,
                answer_key_value=answer_key_value,
                clue_text=clue_text,
                clue_class="source_backed",
                source_method=term_source_method,
                source_page=payload.get("title", title) or title,
                revid=payload.get("revid"),
                sentence_offset=offset,
                lang=lang,
                clue_style="definition",
            )
            for clue_text in texts
        ]
    return rows


def run_seed_ingestion_stage(
    *,
    seed_title: str,
    lang: str = "en",
    diagnostics_path: str | Path = "outputs/diagnostics.json",
    cache_dir: str | Path = "data/cache/wiki",
    include_backlinks: bool = True,
    max_links: int | None = None,
    max_backlinks: int | None = None,
    expansion: str = "one_hop_only",
    max_two_hop_parents: int | None = None,
    max_two_hop_links: int | None = None,
    max_candidates: int | None = None,
) -> SeedStageResult:
    errors: list[str] = []
    cache = DiskCache(cache_dir)
    allowed_langs = {"en", "el"}
    if lang not in allowed_langs:
        errors.append(f"unsupported_lang: {lang}")
        diagnostics = build_seed_stage_diagnostics(
            lang=lang,
            seed_requested=seed_title,
            seed_resolved=seed_title,
            seed_page_id=None,
            candidates=[],
            cache_stats=cache.stats(),
            include_backlinks=include_backlinks,
            errors=errors,
        )
        final_path = write_json(diagnostics_path, diagnostics)
        return SeedStageResult(diagnostics_path=final_path, diagnostics=diagnostics)

    api_base = f"https://{lang}.wikipedia.org/w/api.php"
    client = WikiClient(cache, api_base=api_base)

    links = []
    resolved_seed_title = seed_title
    resolved_seed_page_id: int | None = None
    try:
        links_payload = client.fetch_links(seed_title)
        links = links_payload["links"]
        if max_links is not None and len(links) > max_links:
            # Evenly spaced, not the first N: the API lists links alphabetically.
            links = [links[i * len(links) // max_links] for i in range(max_links)]
        resolved_seed_title = links_payload["seed_title"]
        resolved_seed_page_id = links_payload["seed_page_id"]
    except Exception as exc:  # pragma: no cover - runtime/network path
        errors.append(f"link_fetch_failed: {exc}")

    backlinks: set[str] = set()
    if include_backlinks and links:
        try:
            backlinks = client.fetch_backlinks(resolved_seed_title, max_backlinks=max_backlinks)
        except Exception as exc:  # pragma: no cover - runtime/network path
            errors.append(f"backlink_fetch_failed: {exc}")

    expansion_modes = {"one_hop_only", "one_hop_plus_bounded_two_hop"}
    if expansion not in expansion_modes:
        errors.append(f"unsupported_expansion:{expansion}")
        expansion = "one_hop_only"

    candidate_map: dict[str, dict] = {}
    for candidate in links:
        title = candidate["title"]
        candidate_map[title] = {
            "title": title,
            "depth": 1,
            "links_back_to_seed": title in backlinks if include_backlinks else None,
            "status": "unscored",
        }

    two_hop_parents: list[str] = []
    two_hop_added = 0
    if expansion == "one_hop_plus_bounded_two_hop" and links:
        try:
            seed_extract = client.fetch_page_extract(resolved_seed_title, intro_only=True).get(
                "extract", ""
            )
        except Exception as exc:  # pragma: no cover - runtime/network path
            seed_extract = ""
            errors.append(f"two_hop_seed_extract_failed:{exc}")

        parent_titles = [candidate["title"] for candidate in links]
        parent_extracts: list[str] = []
        if seed_extract:
            _prefetch(errors, client.prefetch_intro_extracts, parent_titles)
            extract_payloads, extract_errors = client.fetch_page_extracts_concurrent(
                parent_titles,
                intro_only=True,
            )
            for title in parent_titles:
                payload = extract_payloads.get(title, {})
                parent_extracts.append(payload.get("extract", ""))
            for title, exc_text in extract_errors.items():
                errors.append(f"two_hop_parent_extract_failed:{title}:{exc_text}")

            vectors = build_tfidf_vectors([seed_extract] + parent_extracts, lang=lang)
            seed_vec = vectors[0]
            rel_scores = [cosine_similarity(seed_vec, vec) for vec in vectors[1:]]
            ranked = sorted(
                zip(parent_titles, rel_scores), key=lambda item: item[1], reverse=True
            )
            limit = max_two_hop_parents or len(ranked)
            two_hop_parents = [title for title, _ in ranked[:limit]]

        parent_link_payloads: dict[str, dict] = {}
        parent_link_errors: dict[str, str] = {}
        if two_hop_parents:
            parent_link_payloads, parent_link_errors = client.fetch_links_concurrent(
                two_hop_parents,
                max_links=max_two_hop_links,
            )
            for title, exc_text in parent_link_errors.items():
                errors.append(f"two_hop_links_failed:{title}:{exc_text}")

        for parent_title in two_hop_parents:
            payload = parent_link_payloads.get(parent_title, {"links": []})
            for link in payload.get("links", []):
                title = link.get("title")
                if not title or title in candidate_map:
                    continue
                candidate_map[title] = {
                    "title": title,
                    "depth": 2,
                    "links_back_to_seed": title in backlinks if include_backlinks else None,
                    "status": "unscored",
                }
                two_hop_added += 1
                if max_candidates is not None and len(candidate_map) >= max_candidates:
                    break
            if max_candidates is not None and len(candidate_map) >= max_candidates:
                break

    candidates = list(candidate_map.values())
    if max_candidates is not None and len(candidates) > max_candidates:
        candidates = candidates[:max_candidates]

    diagnostics = build_seed_stage_diagnostics(
        lang=lang,
        seed_requested=seed_title,
        seed_resolved=resolved_seed_title,
        seed_page_id=resolved_seed_page_id,
        candidates=candidates,
        cache_stats=cache.stats(),
        include_backlinks=include_backlinks,
        errors=errors,
    )
    diagnostics["expansion"] = {
        "mode": expansion,
        "max_two_hop_parents": max_two_hop_parents,
        "max_two_hop_links": max_two_hop_links,
        "max_candidates": max_candidates,
        "two_hop_parents_used": len(two_hop_parents),
        "two_hop_candidates_added": two_hop_added,
    }
    diagnostics["counts"]["depth1_count"] = len([c for c in candidates if c.get("depth") == 1])
    diagnostics["counts"]["depth2_count"] = len([c for c in candidates if c.get("depth") == 2])
    final_path = write_json(diagnostics_path, diagnostics)
    return SeedStageResult(diagnostics_path=final_path, diagnostics=diagnostics)


def run_candidate_scoring_stage(
    *,
    seed_title: str,
    lang: str = "en",
    cache_dir: str | Path = "data/cache/wiki",
    diagnostics_path: str | Path = "outputs/diagnostics_scores.json",
    scores_path: str | Path = "outputs/candidate_scores.csv",
    include_backlinks: bool = True,
    max_links: int | None = None,
    max_backlinks: int | None = None,
    expansion: str = "one_hop_only",
    max_two_hop_parents: int | None = None,
    max_two_hop_links: int | None = None,
    max_candidates: int | None = None,
    extra_titles: list[str] | None = None,
    keep_threshold: float = 0.2,
    borderline_threshold: float = 0.1,
    weights: dict[str, float] | None = None,
    lexicon_path: str | Path | None = "data/lexicon/combined_wordfreq.txt",
    lexicon_weight: float = 0.08,
    max_len: int = 15,
) -> ScoringStageResult:
    weights = weights or {"a": 1.5, "b": 0.3, "c": 0.3, "d": 0.2}
    errors: list[str] = []

    seed_result = run_seed_ingestion_stage(
        seed_title=seed_title,
        lang=lang,
        diagnostics_path=diagnostics_path,
        cache_dir=cache_dir,
        include_backlinks=include_backlinks,
        max_links=max_links,
        max_backlinks=max_backlinks,
        expansion=expansion,
        max_two_hop_parents=max_two_hop_parents,
        max_two_hop_links=max_two_hop_links,
        max_candidates=max_candidates,
    )

    candidates = seed_result.diagnostics.get("candidates", [])
    if candidates:  # two-hop pages: the seed does not link them, so no back-link
        candidates = candidates + [{"title": t, "depth": 2, "links_back_to_seed": False} for t in extra_titles or []]
    if not candidates:
        diagnostics = {
            "stage": "candidate_scoring",
            "lang": lang,
            "seed_title": seed_title,
            "errors": seed_result.diagnostics.get("errors", []),
            "candidate_count": 0,
            "scores_written": False,
            "weights": weights,
        }
        final_diagnostics = write_json(diagnostics_path, diagnostics)
        return ScoringStageResult(
            scores_path=Path(scores_path),
            scores=[],
            diagnostics_path=final_diagnostics,
            diagnostics=diagnostics,
            rel_scores=[],
            pairwise=[],
            ranked_indices=[],
            term_scores=[],
            term_answer_inventories=[],
        )

    cache = DiskCache(cache_dir)
    api_base = f"https://{lang}.wikipedia.org/w/api.php"
    client = WikiClient(cache, api_base=api_base)

    candidate_titles = [candidate["title"] for candidate in candidates]
    _prefetch(errors, client.prefetch_intro_extracts, [seed_title, *candidate_titles])
    _prefetch(errors, client.prefetch_lead_wikitext, candidate_titles)

    try:
        seed_extract = client.fetch_page_extract(seed_title, intro_only=True).get("extract", "")
    except Exception as exc:  # pragma: no cover
        seed_extract = ""
        errors.append(f"seed_extract_failed: {exc}")
    seed_text = _fetch_extract_with_revid(client, seed_title, full=True)["extract"]

    nlp, backend_used, backend_errors, _, _ = _resolve_nlp_backend(lang=lang, nlp_backend="auto")
    errors.extend(backend_errors)

    lexicon = load_lexicon_scores(lexicon_path)
    texts = [seed_extract]
    titles = [seed_title]
    term_answer_inventories: list[set[str]] = []
    intro_on_topic: list[bool] = []
    seed_mentions: list[int] = []
    for candidate in candidates:
        title = candidate["title"]
        try:
            payload = client.fetch_page_extract(title, intro_only=True)
        except Exception as exc:  # pragma: no cover
            payload = {}
            errors.append(f"candidate_extract_failed:{title}:{exc}")
        extract = payload.get("extract", "")
        try:
            lead_wikitext = client.fetch_lead_wikitext(title).get("wikitext", "")
        except Exception as exc:  # pragma: no cover
            lead_wikitext = ""
            errors.append(f"lead_wikitext_failed:{title}:{exc}")
        titles.append(title)
        texts.append(extract)
        # A derived form ("Roman") counts only where the page is defined; later on it is a passing mention.
        first_sentence = (split_sentences(extract) or [""])[0]
        intro_on_topic.append(
            _uses_seed_word(extract, seed_title, lang=lang, lexicon=lexicon)
            or _uses_seed_word(first_sentence, seed_title, lang=lang, lexicon=lexicon, derived=True)
        )
        if intro_on_topic[-1]:
            seed_mentions.append(_seed_mentions(title, seed_text, lang=lang))
        else:
            # Only its mentions can admit this page, so they must be mentions of the answer itself: of the
            # page's own title, not of a redirect to it ("axiomatic" for "Axiom"), and in the same sense.
            page_title = payload.get("title") or title
            named = page_title if title_to_answer(page_title, lang=lang, max_len=max_len) else title
            seed_mentions.append(_seed_mentions(named, seed_text, lang=lang, intro=extract))
        term_answer_inventories.append(
            _extract_answer_inventory(
                title=title,
                extract=extract,
                lead_bold_terms=client.extract_lead_bold_terms(lead_wikitext),
                lang=lang,
                nlp=nlp,
                backend_used=backend_used,
            )
        )

    term_counts = [len(inventory) for inventory in term_answer_inventories]
    max_term_count = max(term_counts) if term_counts else 1
    max_term_count = max(1, max_term_count)
    term_scores = [count / max_term_count for count in term_counts]

    vectors = build_tfidf_vectors(texts, lang=lang)
    seed_vec = vectors[0]
    candidate_vecs = vectors[1:]
    rel_scores = [cosine_similarity(seed_vec, vec) for vec in candidate_vecs]
    pairwise = build_pairwise_matrix(candidate_vecs)
    depth_penalties = [1.0 if c.get("depth", 1) == 2 else 0.0 for c in candidates]
    backlink_bonus = [1.0 if c.get("links_back_to_seed") else 0.0 for c in candidates]

    lexicon_weight = max(0.0, min(1.0, lexicon_weight))
    title_lexicon_scores: list[float] = []
    for title in titles[1:]:
        title_tokens = tokenize(title, lang=lang)
        title_lexicon_scores.append(lexicon_score_for_tokens(title_tokens, lexicon))

    ranking_rel_scores = [
        rel + (lexicon_weight * lex_score)
        for rel, lex_score in zip(rel_scores, title_lexicon_scores)
    ]

    ranked = mmr_rank(
        rel_scores=ranking_rel_scores,
        pairwise=pairwise,
        depth_penalties=depth_penalties,
        backlink_bonus=backlink_bonus,
        a=weights["a"],
        b=weights["b"],
        c=weights["c"],
        d=weights["d"],
    )

    # A linked page is on topic when its intro uses a seed word or the seed article names it twice. Pages whose
    # intro does is first, then pages the seed merely names, then the rest; MMR order within each group (stable).
    group = [0 if intro else 1 if mentions >= 2 else 2 for intro, mentions in zip(intro_on_topic, seed_mentions)]
    ranked.sort(key=lambda row: group[int(row["index"])])

    ranked_indices = [int(row["index"]) for row in ranked]
    scored_rows: list[dict] = []
    for rank_idx, row in enumerate(ranked):
        idx = int(row["index"])
        rel = rel_scores[idx]
        depth = candidates[idx].get("depth", 1)
        links_back = bool(candidates[idx].get("links_back_to_seed"))
        if group[idx] == 2:
            status = "CUT"
            reason = "cut_off_topic"
        elif rel >= keep_threshold:
            status = "KEEP"
            reason = "relevance_above_keep_threshold"
        elif rel >= borderline_threshold:
            status = "BORDERLINE"
            reason = "relevance_above_borderline_threshold"
        else:
            status = "CUT"
            if depth == 2:
                reason = "cut_depth2_low_relevance"
            elif not links_back:
                reason = "cut_no_backlink_low_relevance"
            else:
                reason = "cut_low_relevance"
        scored_rows.append(
            {
                "rank": rank_idx + 1,
                "title": titles[idx + 1],
                "depth": depth,
                "links_back_to_seed": links_back,
                "rel_score": round(rel, 6),
                "lexicon_score": round(title_lexicon_scores[idx], 6),
                "red_score": round(row["red_score"], 6),
                "selection_score": round(row["selection_score"], 6),
                "intro_on_topic": intro_on_topic[idx],
                "seed_mentions": seed_mentions[idx],
                "status": status,
                "reason": reason,
            }
        )

    Path(scores_path).parent.mkdir(parents=True, exist_ok=True)
    import csv

    with Path(scores_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "title",
                "depth",
                "links_back_to_seed",
                "rel_score",
                "lexicon_score",
                "red_score",
                "selection_score",
                "intro_on_topic",
                "seed_mentions",
                "status",
                "reason",
            ]
        )
        for row in scored_rows:
            writer.writerow(
                [
                    row["rank"],
                    row["title"],
                    row["depth"],
                    row["links_back_to_seed"],
                    row["rel_score"],
                    row["lexicon_score"],
                    row["red_score"],
                    row["selection_score"],
                    row["intro_on_topic"],
                    row["seed_mentions"],
                    row["status"],
                    row["reason"],
                ]
            )

    lexicon_hits = sum(1 for score in title_lexicon_scores if score > 0.0)
    diagnostics = {
        "stage": "candidate_scoring",
        "lang": lang,
        "seed_title": seed_title,
        "candidate_count": len(candidates),
        "scores_written": True,
        "weights": weights,
        "lexicon": {
            "path": str(lexicon_path) if lexicon_path is not None else None,
            "loaded": bool(lexicon),
            "entry_count": len(lexicon),
            "weight": lexicon_weight,
            "hits": lexicon_hits,
        },
        "thresholds": {"keep": keep_threshold, "borderline": borderline_threshold},
        "errors": seed_result.diagnostics.get("errors", []) + errors,
    }
    final_diagnostics = write_json(diagnostics_path, diagnostics)
    return ScoringStageResult(
        scores_path=Path(scores_path),
        scores=scored_rows,
        diagnostics_path=final_diagnostics,
        diagnostics=diagnostics,
        rel_scores=rel_scores,
        pairwise=pairwise,
        ranked_indices=ranked_indices,
        term_scores=term_scores,
        term_answer_inventories=term_answer_inventories,
    )


def run_k_selection_stage(
    *,
    seed_title: str,
    lang: str = "en",
    cache_dir: str | Path = "data/cache/wiki",
    diagnostics_path: str | Path = "outputs/diagnostics_k.json",
    scores_path: str | Path = "outputs/candidate_scores.csv",
    trace_path: str | Path = "outputs/k_selection_trace.csv",
    selected_path: str | Path = "outputs/selected_candidates.json",
    include_backlinks: bool = True,
    max_links: int | None = None,
    max_backlinks: int | None = None,
    expansion: str = "one_hop_only",
    max_two_hop_parents: int | None = None,
    max_two_hop_links: int | None = None,
    max_candidates: int | None = None,
    keep_threshold: float = 0.2,
    borderline_threshold: float = 0.1,
    min_k: int = 5,
    max_k: int | None = None,
    epsilon: float = 0.01,
    m: int = 2,
    weights: dict[str, float] | None = None,
    size: int = 15,
    inventory_min_df: int = 2,
    lexicon_path: str | Path | None = "data/lexicon/combined_wordfreq.txt",
    lexicon_weight: float = 0.08,
    scoring: ScoringStageResult | None = None,
) -> KSelectionStageResult:
    weights = weights or {"w1": 1.0, "w2": 1.0, "w3": 1.0}
    # A caller that has already scored passes the result in; rescoring here would also drop its weights.
    scoring = scoring or run_candidate_scoring_stage(
        seed_title=seed_title,
        lang=lang,
        cache_dir=cache_dir,
        diagnostics_path=diagnostics_path,
        scores_path=scores_path,
        include_backlinks=include_backlinks,
        max_links=max_links,
        max_backlinks=max_backlinks,
        expansion=expansion,
        max_two_hop_parents=max_two_hop_parents,
        max_two_hop_links=max_two_hop_links,
        max_candidates=max_candidates,
        keep_threshold=keep_threshold,
        borderline_threshold=borderline_threshold,
        lexicon_path=lexicon_path,
        lexicon_weight=lexicon_weight,
    )

    if not scoring.scores:
        diagnostics = {
            "stage": "k_selection",
            "lang": lang,
            "seed_title": seed_title,
            "selected_k": 0,
            "errors": scoring.diagnostics.get("errors", []),
            "trace_written": False,
            "selected_written": False,
        }
        final_diagnostics = write_json(diagnostics_path, diagnostics)
        return KSelectionStageResult(
            selected_path=Path(selected_path),
            trace_path=Path(trace_path),
            diagnostics_path=final_diagnostics,
            selected={},
            trace=[],
            diagnostics=diagnostics,
        )

    rel_scores = scoring.rel_scores
    pairwise = scoring.pairwise
    ranked_indices = scoring.ranked_indices
    term_scores = scoring.term_scores
    term_answer_inventories = getattr(scoring, "term_answer_inventories", [])
    max_k_limit = max_k or len(ranked_indices)
    if max_k is None and size >= 15:
        max_k_limit = min(max_k_limit, 12)
    max_k_effective = min(max_k_limit, len(ranked_indices))
    inventory_min_k_target = _inventory_min_k_target(
        size=size,
        ranked_indices=ranked_indices[:max_k_effective],
        term_answer_inventories=term_answer_inventories,
        inventory_min_df=inventory_min_df,
    )
    effective_min_k = _effective_min_k_for_size(
        size=size,
        min_k=min_k,
        candidate_count=max_k_effective,
        ranked_indices=ranked_indices[:max_k_effective],
        term_answer_inventories=term_answer_inventories,
        inventory_min_df=inventory_min_df,
    )

    selection = select_k(
        ranked_indices=ranked_indices,
        rel_scores=rel_scores,
        pairwise=pairwise,
        weights=weights,
        min_k=effective_min_k,
        max_k=max_k_effective,
        epsilon=epsilon,
        m=m,
        term_scores=term_scores,
    )

    Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(trace_path).open("w", encoding="utf-8") as handle:
        handle.write("k,jk,delta,coherence,diversity,term_quality,stop\n")
        for row in selection.trace:
            handle.write(
                f"{row['k']},{row['jk']},{row['delta']},{row['coherence']},{row['diversity']},"
                f"{row['term_quality']},{row['stop']}\n"
            )

    ranked_scores = sorted(scoring.scores, key=lambda item: item["rank"])
    index_to_title = {
        candidate_idx: row["title"]
        for candidate_idx, row in zip(ranked_indices, ranked_scores)
    }
    selected_titles = [
        index_to_title[idx] for idx in selection.selected_indices if idx in index_to_title
    ]
    mapping_errors = list(scoring.diagnostics.get("errors", []))
    missing_indices = [idx for idx in selection.selected_indices if idx not in index_to_title]
    if missing_indices:
        mapping_errors.append(
            f"selected_title_mapping_missing_indices:{','.join(str(idx) for idx in missing_indices)}"
        )
    selected_payload_model = SelectedCandidatesArtifact(
        seed_title=seed_title,
        lang=lang,
        selected_k=selection.selected_k,
        selected_titles=selected_titles,
    )
    selected_payload = selected_payload_model.to_dict()
    final_selected = write_json(selected_path, selected_payload)

    diagnostics = {
        "stage": "k_selection",
        "lang": lang,
        "seed_title": seed_title,
        "selected_k": selection.selected_k,
        "effective_min_k": effective_min_k,
        "effective_max_k": max_k_effective,
        "inventory_min_k_target": inventory_min_k_target,
        "inventory_min_df": inventory_min_df,
        "trace_written": True,
        "selected_written": True,
        "weights": weights,
        "thresholds": {"keep": keep_threshold, "borderline": borderline_threshold},
        "errors": mapping_errors,
        "notes": [
            "pairwise_similarity_cosine",
            "term_quality_proxy_extracted_answer_count",
        ],
    }
    final_diagnostics = write_json(diagnostics_path, diagnostics)
    return KSelectionStageResult(
        selected_path=final_selected,
        trace_path=Path(trace_path),
        diagnostics_path=final_diagnostics,
        selected=selected_payload,
        trace=selection.trace,
        diagnostics=diagnostics,
    )


_MAX_ANSWER_POOL = 200

# A Greek page opens with a line that sends the reader to another article for the other meaning:
# "Για το σύστημα χρονολόγησης δείτε: Ημερολόγιο (σύστημα χρονολόγησης)".
_SEE_OTHER_ARTICLE = re.compile(r"Για [^\n]* δείτε")


# A page about a person opens with life dates or "born" in a parenthesis: "(1840 – 1926)", "(born 1946)",
# "(138–78 BC)".
_LIFE_DATES = re.compile(r"\(.*?(?:\bborn\b|\d[^()]*[\u2013\u2014][^()]*\d|\d-\d)")


def _is_person(text: str, *, lang: str) -> bool:
    return bool(_LIFE_DATES.search(lead_subject(text[:600], lang=lang)))


def _seed_short_names(
    intros: dict[str, str], seed_title: str, seed_text: str, *, lang: str, has_seed_fill_in
) -> dict[str, tuple[str, str, int, bool]]:
    """Pages the seed calls by the last word of their title: page title -> (display, normalized, mentions, person).

    A person by the surname: "Monet" for "Claude Monet" when the seed names it alone at least twice. Of several
    people with one surname, the one whose full name the seed uses more often than all the others' together
    takes it; with no such person ("Williams") nobody does. Only a Greek person is named this way, and only by
    the title's own form: the seed writes other pages in an ending of its own.

    Any other page by a word the seed uses as a proper name, capitalised in mid-sentence and alone at least twice:
    "Chicago" for "Music of Chicago", not "Rhythm" for "I Got Rhythm". The seed's sentence is its only clue, so
    `has_seed_fill_in` must find one. A word that ends two such pages is a class noun ("Empire") and nobody's
    name. A surname, another page's own title and a word of the seed title are not free.
    """
    people: dict[str, list[tuple[str, tuple[str, str]]]] = {}
    others: dict[str, list[tuple[str, tuple[str, str]]]] = {}
    for title, intro in intros.items():
        pair = title_short_name(title, lang=lang)
        if pair and title != seed_title:
            (people if _is_person(intro, lang=lang) else others).setdefault(pair[1], []).append((title, pair))
    names = {}
    for rivals in people.values():
        phrases = {title: _TITLE_DISAMBIGUATOR.sub("", title).strip() for title, _ in rivals}
        named = {title: len(answer_pattern(phrase).findall(seed_text)) for title, phrase in phrases.items()}
        for title, pair in rivals:
            if len(rivals) == 1 or 2 * named[title] > sum(named.values()):
                mentions = _name_mentions(pair[0], seed_text, part_of=(phrases[title],))
                if mentions >= 2:
                    names[title] = (*pair, mentions, True)
    if lang != "en":  # a Greek page that is no person keeps its title: the seed writes its inflected form
        return names
    taken = set(people) | {pair[1] for title in intros if (pair := title_to_answer(title, lang=lang))}
    for normalized, pages in others.items():
        title, pair = pages[0]
        if len(pages) > 1 or normalized in taken or _shared_words(pair[0]) & _shared_words(seed_title):
            continue
        phrase = _TITLE_DISAMBIGUATOR.sub("", title).strip()
        mentions = _name_mentions(pair[0], seed_text, part_of=(phrase,), mid_sentence=True)
        if mentions >= 2 and has_seed_fill_in(*pair):
            names[title] = (*pair, mentions, False)
    return names


def _build_title_answer_rows(
    *,
    seed_title: str,
    scores_path: str | Path,
    client: WikiClient,
    extracts: dict[str, str],
    lexicon: dict[str, float],
    lang: str,
    max_len: int,
    max_title_answers: int,
    errors: list[str],
    nlp: object | None = None,
) -> list[dict]:
    """Primary answer pool: titles of the most relevant linked articles, each backed by its own intro.

    Scoring has already cut the off-topic titles; the seed article joins the sources of every title it names.
    """
    import csv

    if not Path(scores_path).exists():
        return []
    with Path(scores_path).open("r", encoding="utf-8", newline="") as handle:
        all_rows = [
            (
                (row.get("title") or "").strip(),
                _safe_float(row.get("rel_score")),
                (row.get("intro_on_topic") or "").strip().lower() == "true",
                _safe_int(row.get("seed_mentions")),
                (row.get("status") or "").strip().upper() in {"KEEP", "BORDERLINE"},
            )
            for row in csv.DictReader(handle)
        ]
    scored = [item[:4] for item in all_rows if item[4]]
    on_topic_intros = {title for title, _, intro_on_topic, _, _ in all_rows if intro_on_topic}
    best_rel = max((item[1] for item in scored), default=0.0)

    _prefetch(errors, client.prefetch_intro_extracts, [seed_title, *(title for title, *_ in scored)])
    pages: list[list] = []  # link title, relevance, intro on topic, seed mentions, page title, intro
    # The seed names itself, so it earns the full bonus.
    for title, rel_score, intro_on_topic, seed_mentions in [(seed_title, best_rel, True, 10), *scored]:
        try:
            payload = client.fetch_page_extract(title, intro_only=True)
        except Exception as exc:
            errors.append(f"title_answer_extract_failed:{title}:{exc}")
            continue
        # A redirect ("Big bands") is named after the page it leads to.
        page_title = title if title == seed_title else payload.get("title") or title
        intro = (payload.get("extract") or "").strip()
        pages.append([title, rel_score, intro_on_topic, seed_mentions, page_title, intro])

    def evidence(page: list) -> tuple[bool, float]:
        return page[0] == seed_title, page[1] + 0.1 * min(page[3], 10) + (0.5 if page[2] else 0.0)

    pages.sort(key=evidence, reverse=True)
    seed_payload = _fetch_extract_with_revid(client, seed_title, full=True)

    def has_seed_fill_in(display: str, normalized: str) -> bool:
        fills = _collect_source_backed_clue_candidates(
            answer=display,
            normalized_answer=normalized,
            source_titles=[seed_title],
            source_payloads={seed_title: seed_payload},
            seed_title=seed_title,
            term_source_method="article_title",
            lang=lang,
            min_words=6,
            max_words=12,
            nlp=nlp,
            allow_fragments=False,
        )
        return any(nlp is None or has_verb(row["clue"], nlp) for row in fills)

    short_names = _seed_short_names(
        {page[4]: page[5] for page in pages},
        seed_title,
        seed_payload["extract"],
        lang=lang,
        has_seed_fill_in=has_seed_fill_in,
    )
    for page in pages:
        if page[4] in short_names:
            page[3] = short_names[page[4]][2]  # the mentions of the answer itself
    # The cap keeps the titles with the most evidence, in the order the answers are scored by (minus the penalty).
    pages.sort(key=evidence, reverse=True)

    owners = {name[1]: page for page, name in short_names.items()}  # "Monet" is the painter, not "Monet (crater)"
    rows: list[dict] = []
    seen: set[str] = set()
    seen_intros: set[str] = set()
    for title, rel_score, intro_on_topic, seed_mentions, page_title, intro in pages:
        if len(rows) >= max_title_answers:
            break
        # The seed's short name for the page, else the page's title, else the redirect's when that is no answer
        # ("Hugh Everett").
        pair = (
            short_names[page_title][:2]
            if page_title in short_names
            else title_to_answer(page_title, lang=lang, max_len=max_len)
            or title_to_answer(title, lang=lang, max_len=max_len)
        )
        # A second title for the same page is skipped, and so is another page under a person's surname.
        if pair is None or pair[1] in seen or not intro or intro in seen_intros:
            continue
        if owners.get(pair[1], page_title) != page_title:
            continue
        display, normalized = pair
        seen.add(normalized)
        seen_intros.add(intro)
        pattern = answer_pattern(display)
        # A selected article the seed merely names is no source for other answers' fill-ins, and a short name
        # of a page that is no person takes them from the seed alone.
        seed_only = page_title in short_names and not short_names[page_title][3]
        mentioning = [
            other
            for other, text in extracts.items()
            if other != page_title and other in on_topic_intros and pattern.search(text) and not seed_only
        ]
        source_titles = [page_title, *([seed_title] if seed_mentions and title != seed_title else []), *mentioning]
        lexicon_score = lexicon_score_for_token(normalized, lexicon)
        # Relevance alone is noisy: reward pages that talk about the seed or that it names often, and push
        # answers that repeat the seed word to the bottom of the title pool.
        score = 1.0 + rel_score + 0.1 * min(seed_mentions, 10)
        if intro_on_topic:
            score += 0.5
        if title != seed_title and _uses_seed_word(display, seed_title, lang=lang, lexicon=lexicon):
            score -= 1.0
        rows.append(
            {
                "answer": display,
                "normalized_answer": normalized,
                "length": len(normalized),
                "source_method": "article_title",
                "lead_bold_signal": False,
                "token_count": len(display.split()),
                "source_titles": "|".join(source_titles),
                "source_title_count": len(source_titles),
                "doc_frequency": len(mentioning),
                "theme_score": round(rel_score, 4),
                "entity_type_score": 0.0,
                "title_support_score": 1.0,
                "support_sentence_count": 0,
                "answer_cleanliness_score": round(answer_cleanliness_score(display, lang=lang), 4),
                "clueability_score": round(clueability_score(display, normalized, lang=lang), 4),
                "source_diversity_score": round(min(1.0, len(source_titles) / 3.0), 4),
                "evidence_tier": "article_title",
                "crosswordability_score": round(crosswordability_score(normalized), 4),
                "lexicon_score": round(lexicon_score, 4),
                "shape_penalty": round(shape_penalty(normalized), 4),
                "answer_score": round(score, 4),
                "intro_on_topic": intro_on_topic,
                "page_is_person": _is_person(intro, lang=lang),
            }
        )
    return rows


def run_term_extraction_stage(
    *,
    seed_title: str,
    lang: str = "en",
    cache_dir: str | Path = "data/cache/wiki",
    diagnostics_path: str | Path = "outputs/diagnostics_terms.json",
    selected_path: str | Path = "outputs/selected_candidates.json",
    terms_path: str | Path = "outputs/answer_candidates.csv",
    min_len: int = 4,
    max_len: int = 15,
    min_alpha_ratio: float = 0.8,
    min_df: int = 2,
    nlp_backend: str = "auto",
    entity_type_scoring: bool = False,
    wikidata_cache_dir: str | Path = "data/cache/wikidata",
    lexicon_path: str | Path | None = "data/lexicon/combined_wordfreq.txt",
    lexicon_weight: float = 0.15,
    scores_path: str | Path | None = None,
    max_title_answers: int = 150,
) -> TermExtractionStageResult:
    errors: list[str] = []
    cache = DiskCache(cache_dir)
    api_base = f"https://{lang}.wikipedia.org/w/api.php"
    client = WikiClient(cache, api_base=api_base)

    if Path(selected_path).exists():
        selected_payload = Path(selected_path).read_text(encoding="utf-8")
        try:
            import json

            selected = json.loads(selected_payload)
            selected_artifact = SelectedCandidatesArtifact.from_dict(selected)
            selected_titles = selected_artifact.selected_titles
        except Exception as exc:  # pragma: no cover
            errors.append(f"selected_parse_failed:{exc}")
            selected_titles = []
    else:
        selected_titles = []
        errors.append("selected_candidates_missing")

    if not selected_titles:
        diagnostics = {
            "stage": "term_extraction",
            "lang": lang,
            "seed_title": seed_title,
            "terms_written": False,
            "errors": errors,
        }
        final_diagnostics = write_json(diagnostics_path, diagnostics)
        return TermExtractionStageResult(
            terms_path=Path(terms_path),
            diagnostics_path=final_diagnostics,
            terms=[],
            diagnostics=diagnostics,
        )

    nlp, backend_used, backend_errors, spacy_version, spacy_model_version = _resolve_nlp_backend(
        lang=lang,
        nlp_backend=nlp_backend,
    )
    errors.extend(backend_errors)

    lead_terms: dict[str, list[str]] = {}
    full_payloads: dict[str, dict] = {}
    extracts: dict[str, str] = {}
    _prefetch(errors, client.prefetch_lead_wikitext, selected_titles)
    for title in selected_titles:
        try:
            intro_extract = client.fetch_page_extract(title, intro_only=True).get("extract", "")
        except Exception as exc:  # pragma: no cover
            intro_extract = ""
            errors.append(f"candidate_extract_failed:{title}:{exc}")

        full_payload = _fetch_extract_with_revid(client, title, full=True)
        if not full_payload["extract"]:  # the plain intro serves term extraction, but has no revision id
            full_payload = {"title": title, "extract": intro_extract, "revid": None}
        full_payloads[title] = full_payload
        extracts[title] = full_payload["extract"]

        try:
            lead_wikitext = client.fetch_lead_wikitext(title).get("wikitext", "")
        except Exception as exc:  # pragma: no cover
            lead_wikitext = ""
            errors.append(f"lead_wikitext_failed:{title}:{exc}")
        lead_terms[title] = client.extract_lead_bold_terms(lead_wikitext)

    term_lists = []
    if backend_used == "spacy" and nlp is not None:
        for title, extract in extracts.items():
            doc = nlp(extract)
            term_lists.append(extract_terms_spacy(doc, source_title=title, lang=lang))
    elif backend_used == "nltk":
        for title, extract in extracts.items():
            term_lists.append(extract_terms_nltk(extract, source_title=title, lang=lang))

    for title, bold_terms in lead_terms.items():
        term_lists.append(extract_terms_lead_bold(bold_terms, source_title=title, lang=lang))
        term_lists.append(extract_terms_title_tokens(title, source_title=title, lang=lang))

    from .term_extractor import get_stopwords

    stopwords = get_stopwords(lang)
    merged, merge_diagnostics = merge_terms(
        term_lists,
        lang=lang,
        min_len=min_len,
        max_len=max_len,
        min_alpha_ratio=min_alpha_ratio,
        stopwords=stopwords,
        return_stats=True,
    )

    df = term_frequency_across_docs(list(extracts.values()), lang=lang)
    eligible_terms: list[tuple] = []
    for term in merged:
        freq = df.get(term.normalized_answer, 0)
        if freq < min_df:
            continue
        eligible_terms.append((term, freq))

    max_df = max((freq for _, freq in eligible_terms), default=1)
    entity_scores: dict[str, float] = {}
    entity_hits = 0
    preferred_entity_types = {"Q115949945", "Q107715", "Q47574"}
    wikidata_client = None
    if entity_type_scoring:
        wikidata_client = WikidataClient(DiskCache(wikidata_cache_dir))
    lexicon = load_lexicon_scores(lexicon_path)
    lexicon_hits = 0

    support_sentence_count_by_key: dict[str, int] = {}
    filtered_terms = []
    unsupported_reject_count = 0
    common_reject_count = 0
    for term, freq in eligible_terms:
        # Extracted words ("players", "American", "result") only add noise; the title pool carries the
        # names, so the secondary pool keeps lead-bold terms (an article's own alternative names) only.
        if not term.lead_bold_signal or term.normalized_answer in GENERIC_GRID_JUNK_EN:
            common_reject_count += 1
            continue
        theme_score = 0.0 if max_df == 0 else freq / max_df
        shape = shape_penalty(term.normalized_answer)
        crossword_score = crosswordability_score(term.normalized_answer)
        source_title_count = len(term.source_titles)
        source_diversity_score = min(1.0, source_title_count / 3.0)
        cleanliness_score = term.answer_cleanliness_score
        clueability = term.clueability_score
        title_support_score = _title_support_score(term.answer, term.source_titles, lang=lang)
        if term.normalized_answer in support_sentence_count_by_key:
            support_sentence_count = support_sentence_count_by_key[term.normalized_answer]
        else:
            support_sentence_count = len(
                _collect_source_backed_clue_candidates(
                    answer=term.answer,
                    normalized_answer=term.normalized_answer,
                    source_titles=sorted(term.source_titles),
                    source_payloads=full_payloads,
                    seed_title=seed_title,
                    term_source_method=term.source_method,
                    lang=lang,
                    min_words=6,
                    max_words=12,
                    nlp=nlp,
                )
            )
            support_sentence_count_by_key[term.normalized_answer] = support_sentence_count
        support_sentence_score = min(1.0, support_sentence_count / 3.0)
        lexicon_score = lexicon_score_for_token(term.normalized_answer, lexicon)
        if lexicon_score > 0.0:
            lexicon_hits += 1
        entity_score = 0.0
        if wikidata_client is not None:
            cache_key = term.normalized_answer
            if cache_key in entity_scores:
                entity_score = entity_scores[cache_key]
            else:
                try:
                    entity_id = wikidata_client.search_entity(term.answer, lang=lang)
                    if entity_id:
                        instance_of = wikidata_client.fetch_instance_of(entity_id)
                        if instance_of & preferred_entity_types:
                            entity_score = 1.0
                            entity_hits += 1
                except Exception as exc:  # pragma: no cover - network path
                    errors.append(f"wikidata_lookup_failed:{term.answer}:{exc}")
                entity_scores[cache_key] = entity_score

        lead_bold_bonus = 1.0 if term.lead_bold_signal else 0.0
        support_override = bool(
            term.lead_bold_signal
            or title_support_score > 0.0
            or source_title_count >= 2
            or support_sentence_count > 0
        )
        if not support_override:
            unsupported_reject_count += 1
            continue
        if term.lead_bold_signal:
            evidence_tier = "lead"
        elif support_sentence_count > 0:
            evidence_tier = "source_backed"
        elif title_support_score > 0.0:
            evidence_tier = "title_supported"
        elif source_title_count >= 2:
            evidence_tier = "multi_source"
        else:
            evidence_tier = "weak"
        evidence_tier_score = {
            "lead": 1.0,
            "source_backed": 0.9,
            "title_supported": 0.8,
            "multi_source": 0.75,
            "weak": 0.4,
        }[evidence_tier]
        residual_weight = max(0.0, 1.0 - lexicon_weight)
        answer_score = (
            (0.14 * residual_weight * theme_score)
            + (0.12 * residual_weight * entity_score)
            + (0.08 * residual_weight * lead_bold_bonus)
            + (0.08 * residual_weight * crossword_score)
            + (0.12 * residual_weight * cleanliness_score)
            + (0.10 * residual_weight * clueability)
            + (0.10 * residual_weight * source_diversity_score)
            + (0.12 * residual_weight * title_support_score)
            + (0.16 * residual_weight * support_sentence_score)
            + (0.08 * residual_weight * evidence_tier_score)
            - (lexicon_weight * lexicon_score)
        )

        filtered_terms.append(
            {
                "answer": term.answer,
                "normalized_answer": term.normalized_answer,
                "length": term.length,
                "source_method": term.source_method,
                "lead_bold_signal": term.lead_bold_signal,
                "token_count": term.token_count,
                "source_titles": "|".join(sorted(term.source_titles)),
                "source_mentions": sorted(term.source_titles),
                "source_title_count": source_title_count,
                "doc_frequency": freq,
                "theme_score": round(theme_score, 4),
                "entity_type_score": round(entity_score, 4),
                "title_support_score": round(title_support_score, 4),
                "support_sentence_count": support_sentence_count,
                "answer_cleanliness_score": round(cleanliness_score, 4),
                "clueability_score": round(clueability, 4),
                "source_diversity_score": round(source_diversity_score, 4),
                "evidence_tier": evidence_tier,
                "crosswordability_score": round(crossword_score, 4),
                "lexicon_score": round(lexicon_score, 4),
                "shape_penalty": round(shape, 4),
                "answer_score": round(answer_score, 4),
            }
        )

    title_rows = (
        _build_title_answer_rows(
            seed_title=seed_title,
            scores_path=scores_path,
            client=client,
            extracts=extracts,
            lexicon=lexicon,
            lang=lang,
            max_len=max_len,
            max_title_answers=max_title_answers,
            errors=errors,
            nlp=nlp,
        )
        if scores_path is not None
        else []
    )
    title_keys = {row["normalized_answer"] for row in title_rows}
    filtered_terms = merge_plural_duplicates(
        title_rows + [row for row in filtered_terms if row["normalized_answer"] not in title_keys]
    )
    filtered_terms.sort(
        key=lambda row: (
            row["answer_score"],
            row["support_sentence_count"],
            row["title_support_score"],
            row["answer_cleanliness_score"],
            row["clueability_score"],
            row["source_title_count"],
            row["doc_frequency"],
            -row["token_count"],
            row["crosswordability_score"],
        ),
        reverse=True,
    )
    del filtered_terms[_MAX_ANSWER_POOL:]

    Path(terms_path).parent.mkdir(parents=True, exist_ok=True)
    import csv

    with Path(terms_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "answer",
                "normalized_answer",
                "length",
                "source_method",
                "lead_bold_signal",
                "token_count",
                "source_titles",
                "source_title_count",
                "doc_frequency",
                "theme_score",
                "entity_type_score",
                "title_support_score",
                "support_sentence_count",
                "answer_cleanliness_score",
                "clueability_score",
                "source_diversity_score",
                "evidence_tier",
                "crosswordability_score",
                "lexicon_score",
                "shape_penalty",
                "answer_score",
                "intro_on_topic",
                "page_is_person",
            ]
        )
        for row in filtered_terms:
            writer.writerow(
                [
                    row["answer"],
                    row["normalized_answer"],
                    row["length"],
                    row["source_method"],
                    row["lead_bold_signal"],
                    row["token_count"],
                    row["source_titles"],
                    row["source_title_count"],
                    row["doc_frequency"],
                    row["theme_score"],
                    row["entity_type_score"],
                    row["title_support_score"],
                    row["support_sentence_count"],
                    row["answer_cleanliness_score"],
                    row["clueability_score"],
                    row["source_diversity_score"],
                    row["evidence_tier"],
                    row["crosswordability_score"],
                    row["lexicon_score"],
                    row["shape_penalty"],
                    row["answer_score"],
                    row.get("intro_on_topic", ""),
                    row.get("page_is_person", ""),
                ]
            )

    diagnostics = {
        "stage": "term_extraction",
        "lang": lang,
        "seed_title": seed_title,
        "terms_written": True,
        "term_count": len(filtered_terms),
        "title_answer_count": sum(1 for row in filtered_terms if row["source_method"] == "article_title"),
        "min_df": min_df,
        "nlp_backend": backend_used,
        "spacy_version": spacy_version if backend_used == "spacy" else None,
        "spacy_model_version": spacy_model_version if backend_used == "spacy" else None,
        "entity_type_scoring": entity_type_scoring,
        "entity_type_hits": entity_hits if entity_type_scoring else 0,
        "dirty_phrase_reject_count": merge_diagnostics.dirty_phrase_reject_count,
        "multiword_reject_count": merge_diagnostics.multiword_reject_count,
        "stopword_boundary_reject_count": merge_diagnostics.stopword_boundary_reject_count,
        "low_cleanliness_reject_count": merge_diagnostics.low_cleanliness_reject_count,
        "common_term_reject_count": common_reject_count,
        "unsupported_term_reject_count": unsupported_reject_count,
        "promoted_by_lead_bold_count": merge_diagnostics.promoted_by_lead_bold_count,
        "promoted_by_source_diversity_count": merge_diagnostics.promoted_by_source_diversity_count,
        "full_extract_source_count": len(full_payloads),
        "lexicon": {
            "path": str(lexicon_path) if lexicon_path is not None else None,
            "loaded": bool(lexicon),
            "entry_count": len(lexicon),
            "weight": lexicon_weight,
            "hits": lexicon_hits,
        },
        "errors": errors,
    }
    final_diagnostics = write_json(diagnostics_path, diagnostics)
    return TermExtractionStageResult(
        terms_path=Path(terms_path),
        diagnostics_path=final_diagnostics,
        terms=filtered_terms,
        diagnostics=diagnostics,
    )


def run_vocab_gate_stage(
    *,
    seed_title: str,
    lang: str = "en",
    diagnostics_path: str | Path = "outputs/diagnostics_vocab_gate.json",
    terms_path: str | Path = "outputs/answer_candidates.csv",
    clues_path: str | Path | None = None,
    min_required: int = 30,
    max_allowed: int | None = None,
) -> VocabGateStageResult:
    errors: list[str] = []
    source_path = Path(terms_path if clues_path is None else clues_path)
    if not source_path.exists():
        missing = "terms_missing" if clues_path is None else "clues_missing"
        errors.append(missing)
        diagnostics = {
            "stage": "vocab_gate",
            "lang": lang,
            "seed_title": seed_title,
            "passed": False,
            "reason": missing,
            "errors": errors,
        }
        final_diagnostics = write_json(diagnostics_path, diagnostics)
        return VocabGateStageResult(diagnostics_path=final_diagnostics, diagnostics=diagnostics)

    if clues_path is None:
        term_count = max(0, len(source_path.read_text(encoding="utf-8").splitlines()) - 1)
    else:
        import csv

        with source_path.open("r", encoding="utf-8", newline="") as handle:
            term_count = _clued_answer_count(list(csv.DictReader(handle)), lang=lang)
    result = evaluate_vocab_gate(term_count, min_required=min_required, max_allowed=max_allowed)

    diagnostics = {
        "stage": "vocab_gate",
        "lang": lang,
        "seed_title": seed_title,
        "passed": result.passed,
        "reason": result.reason,
        "counted": "terms" if clues_path is None else "clued_answers",
        "term_count": result.term_count,
        "min_required": result.min_required,
        "max_allowed": result.max_allowed,
        "errors": errors,
    }
    final_diagnostics = write_json(diagnostics_path, diagnostics)
    return VocabGateStageResult(diagnostics_path=final_diagnostics, diagnostics=diagnostics)


# Widening for a thin seed: link lists of at most 6 selected articles (one request each), at most 100 new pages
# (five intro batches of 20), 6 of them selected.
_RESCUE_PARENTS = 6
_RESCUE_NEW_PAGES = 100
_RESCUE_SELECTED = 6


def _linked_titles(client: WikiClient, parents: list[str], known: set[str], errors: list[str]) -> list[str]:
    """New pages the parents link to, an even sample of each parent's links so that none dominates."""
    payloads, failures = client.fetch_links_concurrent(parents, max_links=500)
    errors.extend(f"two_hop_links_failed:{title}:{exc}" for title, exc in failures.items())
    quota = _RESCUE_NEW_PAGES // max(1, len(parents))
    found: dict[str, None] = {}
    for parent in parents:
        links = [link["title"] for link in payloads[parent]["links"]]
        fresh = [title for title in links if title not in known and title not in found]
        if len(fresh) > quota:
            fresh = [fresh[i * len(fresh) // quota] for i in range(quota)]
        found.update(dict.fromkeys(fresh))
    return list(found)


def run_rescue_ladder(
    *,
    seed_title: str,
    lang: str = "en",
    cache_dir: str | Path = "data/cache/wiki",
    selected_path: str | Path = "outputs/selected_candidates.json",
    terms_path: str | Path = "outputs/answer_candidates.csv",
    diagnostics_path: str | Path = "outputs/diagnostics_rescue.json",
    candidate_scores_path: str | Path = "outputs/candidate_scores.csv",
    terms_diagnostics_path: str | Path = "outputs/diagnostics_terms_rescue.json",
    selected_override_path: str | Path = "outputs/selected_candidates_rescue.json",
    gate_min: int = 30,
    min_len: int = 4,
    max_len: int = 15,
    min_alpha_ratio: float = 0.8,
    min_df: int = 2,
    nlp_backend: str = "auto",
    entity_type_scoring: bool = False,
    wikidata_cache_dir: str | Path = "data/cache/wikidata",
    lexicon_path: str | Path | None = "data/lexicon/combined_wordfreq.txt",
    lexicon_weight: float = 0.15,
    scoring_args: dict | None = None,
    min_words: int = 12,
) -> dict:
    errors: list[str] = []
    steps = []

    def _load_selected_titles(path: Path) -> list[str]:
        if not path.exists():
            return []
        try:
            import json

            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload.get("selected_titles", [])
        except Exception as exc:  # pragma: no cover
            errors.append(f"selected_parse_failed:{exc}")
            return []

    def _write_selected(path: Path, titles: list[str]) -> Path:
        payload = {
            "seed_title": seed_title,
            "lang": lang,
            "selected_k": len(titles),
            "selected_titles": titles,
        }
        return write_json(path, payload)

    def _run_gate(
        min_df_value: int,
        min_len_value: int,
        selected: str | Path | None = None,
    ) -> VocabGateResult:
        run_term_extraction_stage(
            seed_title=seed_title,
            lang=lang,
            cache_dir=cache_dir,
            diagnostics_path=terms_diagnostics_path,
            selected_path=selected or selected_path,
            terms_path=terms_path,
            min_len=min_len_value,
            max_len=max_len,
            min_alpha_ratio=min_alpha_ratio,
            min_df=min_df_value,
            nlp_backend=nlp_backend,
            entity_type_scoring=entity_type_scoring,
            wikidata_cache_dir=wikidata_cache_dir,
            lexicon_path=lexicon_path,
            lexicon_weight=lexicon_weight,
            scores_path=candidate_scores_path,
        )
        clues = run_clue_extraction_stage(
            seed_title=seed_title,
            lang=lang,
            cache_dir=cache_dir,
            diagnostics_path=Path(diagnostics_path).with_name("diagnostics_clues_rescue.json"),
            terms_path=terms_path,
            clues_path=Path(terms_path).with_name("clues.csv"),
            selected_path=selected or selected_path,
        )
        return evaluate_vocab_gate(_clued_answer_count(clues.clues, lang=lang), min_required=gate_min)

    selected_titles = _load_selected_titles(Path(selected_path))
    if not selected_titles:
        diagnostics = {
            "stage": "rescue_ladder",
            "lang": lang,
            "seed_title": seed_title,
            "passed": False,
            "reason": "selected_candidates_missing",
            "steps": steps,
            "errors": errors + ["selected_candidates_missing"],
        }
        final_diagnostics = write_json(diagnostics_path, diagnostics)
        return {"passed": False, "diagnostics": diagnostics, "diagnostics_path": final_diagnostics}

    gate = _run_gate(min_df_value=min_df, min_len_value=min_len)
    if gate.passed:
        diagnostics = {
            "stage": "rescue_ladder",
            "lang": lang,
            "seed_title": seed_title,
            "passed": True,
            "reason": "baseline_passed",
            "steps": steps,
            "gate": gate.__dict__,
            "errors": errors,
        }
        final_diagnostics = write_json(diagnostics_path, diagnostics)
        return {"passed": True, "diagnostics": diagnostics, "diagnostics_path": final_diagnostics}

    steps.append({"action": "loosen_min_df", "value": 1})
    gate = _run_gate(min_df_value=1, min_len_value=min_len)
    if gate.passed:
        diagnostics = {
            "stage": "rescue_ladder",
            "lang": lang,
            "seed_title": seed_title,
            "passed": True,
            "reason": "loosen_min_df",
            "steps": steps,
            "gate": gate.__dict__,
            "errors": errors,
        }
        final_diagnostics = write_json(diagnostics_path, diagnostics)
        return {"passed": True, "diagnostics": diagnostics, "diagnostics_path": final_diagnostics}

    # Promote borderline candidates (up to 3) into the selected set.
    if Path(candidate_scores_path).exists():
        import csv

        borderline = []
        with Path(candidate_scores_path).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                errors.append("candidate_scores_header_missing")
            for row in reader:
                title = (row.get("title") or "").strip()
                status = (row.get("status") or "").strip().upper()
                if not title:
                    continue
                if status == "BORDERLINE" and title not in selected_titles and title not in borderline:
                    borderline.append(title)
                if len(borderline) >= 3:
                    break
        steps.append({"action": "promote_borderline", "value": len(borderline)})
        if borderline:
            selected_titles = selected_titles + borderline
            selected_path = _write_selected(Path(selected_override_path), selected_titles)
    else:
        steps.append({"action": "promote_borderline", "value": 0, "note": "candidate_scores_missing"})

    gate = _run_gate(min_df_value=1, min_len_value=min_len)
    if gate.passed:
        diagnostics = {
            "stage": "rescue_ladder",
            "lang": lang,
            "seed_title": seed_title,
            "passed": True,
            "reason": "promote_borderline",
            "steps": steps,
            "gate": gate.__dict__,
            "errors": errors,
        }
        final_diagnostics = write_json(diagnostics_path, diagnostics)
        return {"passed": True, "diagnostics": diagnostics, "diagnostics_path": final_diagnostics}

    # Two-hop: the pages the on-topic selected articles link to are scored like the seed's own links. The best
    # join the selected articles and all of them can be answers. Without the scoring arguments nothing can be scored.
    if scoring_args is not None:
        scores = _read_csv_rows(candidate_scores_path)
        on_topic_intros = {row["title"] for row in scores if row.get("intro_on_topic") == "True"}
        parents = [title for title in selected_titles if title in on_topic_intros][:_RESCUE_PARENTS]
        client = WikiClient(DiskCache(cache_dir), api_base=f"https://{lang}.wikipedia.org/w/api.php")
        extra_titles = _linked_titles(client, parents, {row["title"] for row in scores} | {seed_title}, errors)
        steps.append({"action": "widen_from_selected", "value": len(extra_titles)})
        if extra_titles:
            widened = run_candidate_scoring_stage(
                **scoring_args,
                extra_titles=extra_titles,
                diagnostics_path=Path(diagnostics_path).with_name("diagnostics_scores_rescue.json"),
                scores_path=Path(selected_override_path).with_name("candidate_scores_rescue.csv"),
            )
            candidate_scores_path = widened.scores_path
            selected_titles = selected_titles + [
                row["title"] for row in widened.scores if row["depth"] == 2 and row["status"] != "CUT"
            ][:_RESCUE_SELECTED]
            selected_path = _write_selected(Path(selected_override_path), selected_titles)
            gate = _run_gate(min_df_value=1, min_len_value=min_len)
            if gate.passed:
                diagnostics = {
                    "stage": "rescue_ladder",
                    "lang": lang,
                    "seed_title": seed_title,
                    "passed": True,
                    "reason": "widen_from_selected",
                    "steps": steps,
                    "gate": gate.__dict__,
                    "errors": errors,
                }
                final_diagnostics = write_json(diagnostics_path, diagnostics)
                return {"passed": True, "diagnostics": diagnostics, "diagnostics_path": final_diagnostics}

    steps.append({"action": "shorten_min_len", "value": 3})
    gate = _run_gate(min_df_value=1, min_len_value=3)
    if gate.passed:
        diagnostics = {
            "stage": "rescue_ladder",
            "lang": lang,
            "seed_title": seed_title,
            "passed": True,
            "reason": "shorten_min_len",
            "steps": steps,
            "gate": gate.__dict__,
            "errors": errors,
        }
        final_diagnostics = write_json(diagnostics_path, diagnostics)
        return {"passed": True, "diagnostics": diagnostics, "diagnostics_path": final_diagnostics}

    # Too few clued answers for the gate, but enough for the smallest grid the layout accepts: the caller goes on.
    thin = gate.term_count >= min_words
    if thin:
        steps.append({"action": "accept_thin_pool", "value": gate.term_count})
    diagnostics = {
        "stage": "rescue_ladder",
        "lang": lang,
        "seed_title": seed_title,
        "passed": False,
        "reason": "accept_thin_pool" if thin else "insufficient_vocabulary",
        "steps": steps,
        "gate": gate.__dict__,
        "errors": errors,
    }
    final_diagnostics = write_json(diagnostics_path, diagnostics)
    return {"passed": False, "diagnostics": diagnostics, "diagnostics_path": final_diagnostics}


def run_clue_extraction_stage(
    *,
    seed_title: str,
    lang: str = "en",
    cache_dir: str | Path = "data/cache/wiki",
    diagnostics_path: str | Path = "outputs/diagnostics_clues.json",
    terms_path: str | Path = "outputs/answer_candidates.csv",
    clues_path: str | Path = "outputs/clues.csv",
    selected_path: str | Path = "outputs/selected_candidates.json",
    min_words: int = 6,
    max_words: int = 12,
    diversity_cap: int = 2,
) -> ClueStageResult:
    errors: list[str] = []
    if not Path(terms_path).exists():
        errors.append("terms_missing")
        diagnostics = {
            "stage": "clue_extraction",
            "lang": lang,
            "seed_title": seed_title,
            "clues_written": False,
            "errors": errors,
        }
        final_diagnostics = write_json(diagnostics_path, diagnostics)
        return ClueStageResult(
            clues_path=Path(clues_path),
            diagnostics_path=final_diagnostics,
            clues=[],
            diagnostics=diagnostics,
        )

    cache = DiskCache(cache_dir)
    api_base = f"https://{lang}.wikipedia.org/w/api.php"
    client = WikiClient(cache, api_base=api_base)
    nlp = None
    clue_spacy_version = None
    clue_spacy_model_version = None
    try:
        import spacy

        model_name = "en_core_web_sm" if lang == "en" else "el_core_news_sm"
        nlp = spacy.load(model_name)
        clue_spacy_version = getattr(spacy, "__version__", None)
        try:
            clue_spacy_model_version = nlp.meta.get("version")
        except Exception:  # pragma: no cover
            pass
    except Exception as exc:  # pragma: no cover
        errors.append(f"clue_spacy_unavailable:{exc}")

    import csv
    import json

    terms: list[dict[str, str]] = []
    with Path(terms_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            terms.append(
                {
                    "answer": row.get("answer", ""),
                    "normalized_answer": row.get("normalized_answer", ""),
                    "source_titles": row.get("source_titles", ""),
                    "source_method": row.get("source_method", ""),
                    "intro_on_topic": row.get("intro_on_topic", ""),
                    "page_is_person": row.get("page_is_person", ""),
                }
            )

    # The seed and the selected articles are cited from their full text; every other page from its intro.
    selected_titles: list[str] = []
    try:
        selected = json.loads(Path(selected_path).read_text(encoding="utf-8"))
        selected_titles = SelectedCandidatesArtifact.from_dict(selected).selected_titles
    except Exception as exc:
        errors.append(f"selected_unavailable:{exc}")
    full_titles = {seed_title, *selected_titles}
    pool_titles = {title for term in terms for title in term["source_titles"].split("|") if title}
    _prefetch(errors, client.prefetch_intro_extracts, sorted(pool_titles - full_titles), with_revid=True)

    source_cache: dict[str, dict] = {}

    def load_source_payload(title: str) -> dict:
        if title not in source_cache:
            source_cache[title] = _fetch_extract_with_revid(client, title, full=title in full_titles)
        return source_cache[title]

    leads = [
        sentences[0]
        for term in terms
        if term["source_method"] == "article_title" and term["source_titles"]
        if (sentences := split_sentences(load_source_payload(term["source_titles"].split("|")[0]).get("extract", "")))
    ]
    lead_counts = lead_word_counts(leads)

    clues: list[dict] = []
    clued_keys: set[str] = set()
    unique_term_keys: set[str] = set()
    used_texts: set[str] = set()
    used_sentences: set[tuple[str, int | None]] = set()
    fill_clues: list[dict] = []
    diversity_rejected_count = 0

    for term in terms:
        answer = term["answer"]
        normalized_answer = term.get("normalized_answer", "")
        key = _answer_key(answer, normalized_answer, lang=lang)
        if not key:
            continue
        unique_term_keys.add(key)
        if key in clued_keys:
            continue
        source_titles = [title for title in term["source_titles"].split("|") if title]
        for title in source_titles:
            load_source_payload(title)
        page = source_titles[0] if term["source_method"] == "article_title" and source_titles else ""
        own_title = _TITLE_DISAMBIGUATOR.sub("", page).strip()
        last_word = bool(page) and (title_short_name(page, lang=lang) or ("",))[0] == answer
        person = term["page_is_person"] == "True"
        # A disambiguated page, or one that sends to another article for its other meaning, whose intro is off topic
        # ("Sicily (theme)") may be another sense of the seed's word: no definition, and its own text clues nothing.
        other_sense = (
            bool(page)
            and bool(_TITLE_DISAMBIGUATOR.search(page) or _SEE_OTHER_ARTICLE.match(source_cache[page].get("extract", "")))
            and term["intro_on_topic"] == "False"
        )
        if other_sense:
            source_titles = source_titles[1:]
        # The seed's name for a page that is no person ("Chicago" for "Music of Chicago"): only the seed clues it.
        seed_only = last_word and not person
        if seed_only:
            load_source_payload(seed_title)
        candidates = _collect_source_backed_clue_candidates(
            answer=answer,
            normalized_answer=normalized_answer,
            source_titles=[seed_title] if seed_only else source_titles,
            source_payloads=source_cache,
            seed_title=seed_title,
            term_source_method=term["source_method"],
            lang=lang,
            min_words=min_words,
            max_words=max_words,
            nlp=nlp,
            allow_fragments=False,
            full_name=own_title if last_word and person else None,
        )
        if page:
            # A page defines its own name only: its title, or any name of the person it is about ("Monet", "Peter
            # Tait" for "Peter Guthrie Tait"). Another page's redirect ("Subatomic" for "Subatomic particle") is
            # another word, and takes fill-ins only.
            definitions = []
            if (answer == own_title or person) and not other_sense:
                definitions = _definition_clue_rows(
                    answer=answer,
                    normalized_answer=normalized_answer,
                    answer_key_value=key,
                    title=page,
                    payload=source_cache[page],
                    term_source_method=term["source_method"],
                    lang=lang,
                    nlp=nlp,
                    lead_counts=lead_counts,
                    pool_size=len(leads),
                    seed_title=seed_title,
                )
            # A follow-up (the second sentence) is for an answer nothing else clues: it comes after the fill-ins.
            candidates = (
                [row for row in definitions if row["sentence_offset"] == 0]
                + candidates
                + [row for row in definitions if row["sentence_offset"] == 1]
            )

        for row in candidates:
            sentence = (row["source_page"], row["sentence_offset"])
            if row["clue"].lower() in used_texts:
                continue
            if row["clue_style"] == "fill_in_blank":
                if sentence in used_sentences or (nlp is not None and not has_verb(row["clue"], nlp)):
                    continue
                kept, _ = enforce_diversity(fill_clues + [row], max_per_bucket=diversity_cap)
                if len(kept) == len(fill_clues):
                    diversity_rejected_count += 1
                    continue
                fill_clues.append(row)
            used_texts.add(row["clue"].lower())
            used_sentences.add(sentence)
            clues.append(row)
            clued_keys.add(key)
            break

    Path(clues_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(clues_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "answer",
                "normalized_answer",
                "clue",
                "clue_score",
                "clue_class",
                "clue_style",
                "source_method",
                "source_page",
                "revid",
                "sentence_offset",
                "oldid_url",
            ]
        )
        for row in clues:
            writer.writerow(
                [
                    row["answer"],
                    row.get("normalized_answer", ""),
                    row["clue"],
                    row.get("clue_score", ""),
                    row.get("clue_class", ""),
                    row.get("clue_style", ""),
                    row.get("source_method", ""),
                    row.get("source_page", ""),
                    row.get("revid", ""),
                    row.get("sentence_offset", ""),
                    row.get("oldid_url", ""),
                ]
            )

    provenance_missing = validate_clue_provenance(clues)
    clue_class_counts: dict[str, int] = {}
    for row in clues:
        clue_class = _infer_clue_class(row)
        clue_class_counts[clue_class] = clue_class_counts.get(clue_class, 0) + 1
    definition_count = sum(1 for row in clues if row["clue_style"] == "definition")
    diagnostics = {
        "stage": "clue_extraction",
        "lang": lang,
        "seed_title": seed_title,
        "clues_written": True,
        "term_count": len(unique_term_keys),
        "clue_count": len(clues),
        "definition_clue_count": definition_count,
        "fill_in_blank_clue_count": len(clues) - definition_count,
        "spacy_version": clue_spacy_version,
        "spacy_model_version": clue_spacy_model_version,
        "diversity_rejected_count": diversity_rejected_count,
        "unclued_term_count": len(unique_term_keys - clued_keys),
        "clue_class_counts": clue_class_counts,
        "source_backed_clue_count": clue_class_counts.get("source_backed", 0),
        "template_fallback_clue_count": clue_class_counts.get("template_fallback", 0),
        "provenance_missing_count": len(provenance_missing),
        "provenance_missing": provenance_missing[:10],
        "errors": errors,
    }
    final_diagnostics = write_json(diagnostics_path, diagnostics)
    return ClueStageResult(
        clues_path=Path(clues_path),
        diagnostics_path=final_diagnostics,
        clues=clues,
        diagnostics=diagnostics,
    )


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


def _layout_candidates(
    term_rows: list[dict],
    clue_rows: list[dict],
    *,
    lang: str,
    limit: int,
) -> list[tuple[str, float]]:
    """Packageable answers by descending answer_score, skipping any that repeats a word already taken."""
    clued = {
        _answer_key(row.get("answer", ""), row.get("normalized_answer", ""), lang=lang)
        for row in clue_rows
        if _is_packageable_clue(row)
    }
    ranked = sorted(term_rows, key=lambda row: -_safe_float(row.get("answer_score")))
    picked: list[tuple[str, float]] = []
    taken: set[str] = set()
    words: set[str] = set()
    for row in ranked:
        key = _answer_key(row.get("answer", ""), row.get("normalized_answer", ""), lang=lang)
        if key not in clued or key in taken:
            continue
        row_words = _shared_words(row.get("answer", ""))
        if row_words & words:
            continue
        taken.add(key)
        words |= row_words
        picked.append((key, _safe_float(row.get("answer_score"))))
        if len(picked) >= limit:
            break
    return picked


def run_layout_stage(
    *,
    seed_title: str,
    lang: str = "en",
    terms_path: str | Path = "outputs/answer_candidates.csv",
    clues_path: str | Path = "outputs/clues.csv",
    grid_path: str | Path = "outputs/grid.json",
    diagnostics_path: str | Path = "outputs/diagnostics_layout.json",
    size: int = 15,
    random_seed: int = 13,
    restarts: int = 96,
    candidates: int = 80,
    min_words: int = 12,
    gate_min: int = 30,
) -> LayoutStageResult:
    errors: list[str] = []
    term_rows = _read_csv_rows(terms_path)
    clue_rows = _read_csv_rows(clues_path)
    if _clued_answer_count(clue_rows, lang=lang) < gate_min:
        errors.append("vocab_gate_failed")

    picked = [] if errors else _layout_candidates(term_rows, clue_rows, lang=lang, limit=candidates)
    layout = build_layout(picked, size=size, random_seed=random_seed, restarts=restarts)
    problems = validate_layout(layout.grid, layout.placements)
    complete = not problems and layout.metrics["word_count"] >= min_words
    fill_status = "complete" if complete else "failed"

    grid_model = GridArtifact(
        seed_title=seed_title,
        lang=lang,
        size=size,
        grid=layout.grid,
        placements=[
            {"answer": p.answer, "row": p.row, "col": p.col, "direction": p.direction}
            for p in layout.placements
        ],
        fill_status=fill_status,
    )
    final_grid = write_json(grid_path, grid_model.to_dict())
    diagnostics = {
        "stage": "layout",
        "lang": lang,
        "seed_title": seed_title,
        "size": size,
        "random_seed": random_seed,
        "restarts": restarts,
        "min_words": min_words,
        "candidate_count": len(picked),
        "fill_status": fill_status,
        "validity_errors": problems,
        "errors": errors,
        **layout.metrics,
    }
    final_diagnostics = write_json(diagnostics_path, diagnostics)
    return LayoutStageResult(
        grid_path=final_grid,
        diagnostics_path=final_diagnostics,
        diagnostics=diagnostics,
    )


def run_packaging_stage(
    *,
    seed_title: str,
    lang: str = "en",
    selected_path: str | Path = "outputs/selected_candidates.json",
    grid_path: str | Path = "outputs/grid.json",
    clues_path: str | Path = "outputs/clues.csv",
    puzzle_path: str | Path = "outputs/puzzle.json",
    attribution_path: str | Path = "outputs/attribution.json",
    diagnostics_path: str | Path = "outputs/diagnostics_package.json",
    gate_min: int = 30,
) -> PackagingStageResult:
    import json

    errors: list[str] = []

    selected = SelectedCandidatesArtifact(seed_title=seed_title, lang=lang, selected_k=0, selected_titles=[])
    if Path(selected_path).exists():
        try:
            selected = SelectedCandidatesArtifact.from_dict(
                json.loads(Path(selected_path).read_text(encoding="utf-8"))
            )
        except Exception as exc:  # pragma: no cover
            errors.append(f"selected_parse_failed:{exc}")
    else:
        errors.append("selected_missing")

    grid = GridArtifact(seed_title=seed_title, lang=lang, size=15, grid=[], placements=[])
    if Path(grid_path).exists():
        try:
            grid = GridArtifact.from_dict(json.loads(Path(grid_path).read_text(encoding="utf-8")))
        except Exception as exc:  # pragma: no cover
            errors.append(f"grid_parse_failed:{exc}")
    else:
        errors.append("grid_missing")

    clues = _read_csv_rows(clues_path)
    if not Path(clues_path).exists():
        errors.append("clues_missing")
    clue_by_key: dict[str, dict] = {}
    for clue in clues:
        if _is_packageable_clue(clue):
            key = _answer_key(clue.get("answer", ""), clue.get("normalized_answer", ""), lang=lang)
            clue_by_key.setdefault(key, clue)

    placements = [Placement(p["answer"], p["row"], p["col"], p["direction"]) for p in grid.placements]
    numbers = number_placements(placements)
    entries: dict[str, list[dict]] = {"across": [], "down": []}
    used_clues: list[dict] = []
    unclued: list[str] = []
    for placement in sorted(placements, key=lambda p: numbers[(p.row, p.col)]):
        clue = clue_by_key.get(placement.answer)
        if clue is None:
            unclued.append(placement.answer)
            continue
        display = clue.get("answer") or placement.answer
        entries[placement.direction].append(
            {
                "number": numbers[(placement.row, placement.col)],
                "answer": placement.answer,
                "display_answer": display,
                "enumeration": enumeration(display),
                "clue": clue["clue"],
                "clue_style": clue.get("clue_style", ""),
                "clue_class": _infer_clue_class(clue),
                "row": placement.row,
                "col": placement.col,
                "length": len(placement.answer),
            }
        )
        used_clues.append(clue)

    provenance_missing = validate_clue_provenance(used_clues)
    if provenance_missing:
        errors.append("provenance_incomplete")
    if unclued:
        errors.append("clue_coverage_failed")
    if _clued_answer_count(clues, lang=lang) < gate_min:
        puzzle_status = "insufficient_vocabulary"
    elif unclued:
        puzzle_status = "insufficient_clues"
    elif grid.fill_status != "complete":
        puzzle_status = "insufficient_quality"
    else:
        puzzle_status = "ok"

    placed = [entry for rows in entries.values() for entry in rows]
    clue_texts = [entry["clue"].lower() for entry in placed]
    definition_count = sum(entry["clue_style"] == "definition" for entry in placed)
    diagnostics = {
        "stage": "packaging",
        "lang": lang,
        "seed_title": seed_title,
        "selected_k": selected.selected_k,
        "clue_count": len(clues),
        "entry_count": len(placed),
        "fill_status": grid.fill_status,
        "puzzle_status": puzzle_status,
        "definition_clue_count": definition_count,
        "fill_in_blank_clue_count": len(placed) - definition_count,
        "duplicate_clue_text_count": len(clue_texts) - len(set(clue_texts)),
        "used_clue_provenance_missing_count": len(provenance_missing),
        "errors": errors,
    }
    puzzle_model = PuzzleArtifact(
        seed_title=seed_title,
        lang=lang,
        selected_articles=selected.selected_titles,
        selected_k=selected.selected_k,
        size=grid.size,
        grid_cells=grid.grid,
        across_entries=entries["across"],
        down_entries=entries["down"],
        fill_status=grid.fill_status,
        puzzle_status=puzzle_status,
        layout=layout_metrics(grid.grid, placements),
        diagnostics=diagnostics,
        attribution=used_clues,
    )
    final_puzzle = write_json(puzzle_path, puzzle_model.to_dict())
    final_attribution = write_json(attribution_path, {"clues": used_clues})
    final_diagnostics = write_json(diagnostics_path, diagnostics)
    return PackagingStageResult(
        puzzle_path=final_puzzle,
        attribution_path=final_attribution,
        diagnostics_path=final_diagnostics,
        diagnostics=diagnostics,
    )


def run_generate_pipeline(
    *,
    seed_title: str,
    lang: str = "en",
    output_dir: str | Path = "outputs",
    cache_dir: str | Path = "data/cache/wiki",
    wikidata_cache_dir: str | Path = "data/cache/wikidata",
    include_backlinks: bool = True,
    max_links: int | None = None,
    max_backlinks: int | None = None,
    expansion: str = "one_hop_only",
    max_two_hop_parents: int | None = None,
    max_two_hop_links: int | None = None,
    max_candidates: int | None = None,
    keep_threshold: float = 0.2,
    borderline_threshold: float = 0.1,
    candidate_weights: dict[str, float] | None = None,
    k_weights: dict[str, float] | None = None,
    lexicon_path: str | Path | None = "data/lexicon/combined_wordfreq.txt",
    candidate_lexicon_weight: float = 0.08,
    term_lexicon_weight: float = 0.15,
    min_k: int = 5,
    max_k: int | None = None,
    epsilon: float = 0.01,
    m: int = 2,
    min_len: int = 4,
    max_len: int = 15,
    min_alpha_ratio: float = 0.8,
    min_df: int = 2,
    nlp_backend: str = "auto",
    entity_type_scoring: bool = False,
    gate_min: int = 30,
    rescue: bool = True,
    clue_min_words: int = 6,
    clue_max_words: int = 12,
    diversity_cap: int = 2,
    size: int = 15,
    random_seed: int = 13,
    layout_restarts: int = 96,
    layout_candidates: int = 80,
    min_words: int = 12,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scores_path = output_dir / "candidate_scores.csv"
    selected_path = output_dir / "selected_candidates.json"
    terms_path = output_dir / "answer_candidates.csv"
    clues_path = output_dir / "clues.csv"
    selected_rescue_path = output_dir / "selected_candidates_rescue.json"

    scoring_args = {
        "seed_title": seed_title,
        "lang": lang,
        "cache_dir": cache_dir,
        "include_backlinks": include_backlinks,
        "max_links": max_links,
        "max_backlinks": max_backlinks,
        "expansion": expansion,
        "max_two_hop_parents": max_two_hop_parents,
        "max_two_hop_links": max_two_hop_links,
        "max_candidates": max_candidates,
        "keep_threshold": keep_threshold,
        "borderline_threshold": borderline_threshold,
        "weights": candidate_weights,
        "lexicon_path": lexicon_path,
        "lexicon_weight": candidate_lexicon_weight,
        "max_len": max_len,
    }
    scoring = run_candidate_scoring_stage(
        **scoring_args,
        diagnostics_path=output_dir / "diagnostics_scores.json",
        scores_path=scores_path,
    )

    selection = run_k_selection_stage(
        seed_title=seed_title,
        lang=lang,
        cache_dir=cache_dir,
        diagnostics_path=output_dir / "diagnostics_k.json",
        scores_path=scores_path,
        trace_path=output_dir / "k_selection_trace.csv",
        selected_path=selected_path,
        include_backlinks=include_backlinks,
        max_links=max_links,
        max_backlinks=max_backlinks,
        expansion=expansion,
        max_two_hop_parents=max_two_hop_parents,
        max_two_hop_links=max_two_hop_links,
        max_candidates=max_candidates,
        keep_threshold=keep_threshold,
        borderline_threshold=borderline_threshold,
        min_k=min_k,
        max_k=max_k,
        epsilon=epsilon,
        m=m,
        size=size,
        inventory_min_df=min_df,
        weights=k_weights,
        lexicon_path=lexicon_path,
        lexicon_weight=candidate_lexicon_weight,
        scoring=scoring,
    )

    terms = run_term_extraction_stage(
        seed_title=seed_title,
        lang=lang,
        cache_dir=cache_dir,
        diagnostics_path=output_dir / "diagnostics_terms.json",
        selected_path=selected_path,
        terms_path=terms_path,
        min_len=min_len,
        max_len=max_len,
        min_alpha_ratio=min_alpha_ratio,
        min_df=min_df,
        nlp_backend=nlp_backend,
        entity_type_scoring=entity_type_scoring,
        wikidata_cache_dir=wikidata_cache_dir,
        lexicon_path=lexicon_path,
        lexicon_weight=term_lexicon_weight,
        scores_path=scores_path,
    )

    selected_for_package = selected_path

    def run_clues() -> ClueStageResult:
        return run_clue_extraction_stage(
            seed_title=seed_title,
            lang=lang,
            cache_dir=cache_dir,
            diagnostics_path=output_dir / "diagnostics_clues.json",
            terms_path=terms_path,
            clues_path=clues_path,
            selected_path=selected_for_package,
            min_words=clue_min_words,
            max_words=clue_max_words,
            diversity_cap=diversity_cap,
        )

    def run_gate() -> VocabGateStageResult:
        return run_vocab_gate_stage(
            seed_title=seed_title,
            lang=lang,
            diagnostics_path=output_dir / "diagnostics_vocab_gate.json",
            clues_path=clues_path,
            min_required=gate_min,
        )

    clues = run_clues()
    gate = run_gate()

    rescue_result = None
    if rescue and not gate.diagnostics.get("passed", False):
        rescue_result = run_rescue_ladder(
            seed_title=seed_title,
            lang=lang,
            cache_dir=cache_dir,
            selected_path=selected_path,
            terms_path=terms_path,
            diagnostics_path=output_dir / "diagnostics_rescue.json",
            candidate_scores_path=scores_path,
            terms_diagnostics_path=output_dir / "diagnostics_terms_rescue.json",
            selected_override_path=selected_rescue_path,
            gate_min=gate_min,
            min_len=min_len,
            max_len=max_len,
            min_alpha_ratio=min_alpha_ratio,
            min_df=min_df,
            nlp_backend=nlp_backend,
            entity_type_scoring=entity_type_scoring,
            wikidata_cache_dir=wikidata_cache_dir,
            lexicon_path=lexicon_path,
            lexicon_weight=term_lexicon_weight,
            scoring_args=scoring_args,
            min_words=min_words,
        )
        if selected_rescue_path.exists():
            selected_for_package = selected_rescue_path
        if rescue_result["diagnostics"]["reason"] == "accept_thin_pool":
            gate_min = min_words  # the gate, the layout and the package take the thin pool as it is
        clues = run_clues()
        gate = run_gate()

    layout = run_layout_stage(
        seed_title=seed_title,
        lang=lang,
        terms_path=terms_path,
        clues_path=clues_path,
        grid_path=output_dir / "grid.json",
        diagnostics_path=output_dir / "diagnostics_layout.json",
        size=size,
        random_seed=random_seed,
        restarts=layout_restarts,
        candidates=layout_candidates,
        min_words=min_words,
        gate_min=gate_min,
    )

    package = run_packaging_stage(
        seed_title=seed_title,
        lang=lang,
        selected_path=selected_for_package,
        grid_path=layout.grid_path,
        clues_path=clues_path,
        puzzle_path=output_dir / "puzzle.json",
        attribution_path=output_dir / "attribution.json",
        diagnostics_path=output_dir / "diagnostics_package.json",
        gate_min=gate_min,
    )

    return {
        "scoring": scoring,
        "selection": selection,
        "terms": terms,
        "gate": gate,
        "rescue": rescue_result,
        "clues": clues,
        "layout": layout,
        "package": package,
        "output_dir": output_dir,
    }
