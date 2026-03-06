from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .cache import DiskCache
from .artifact_models import (
    GridArtifact,
    GridSlotRecord,
    PuzzleArtifact,
    SelectedCandidatesArtifact,
)
from .diagnostics import build_seed_stage_diagnostics, write_json
from .k_selector import select_k
from .semantic import build_pairwise_matrix, build_tfidf_vectors, cosine_similarity, mmr_rank
from .term_extractor import (
    extract_terms_lead_bold,
    extract_terms_nltk,
    extract_terms_spacy,
    extract_terms_title_tokens,
    genericness_penalty,
    merge_terms,
    shape_penalty,
    crosswordability_score,
    term_frequency_across_docs,
)
from .text_normalize import tokenize
from .vocab_gate import evaluate_vocab_gate
from .clue_builder import (
    clue_pass_extract,
    clue_pass_extract_with_offset,
    clue_pass_mask_trim,
    clue_pass_validate,
    enforce_diversity,
    split_sentences,
)
from .provenance import build_oldid_url, has_packageable_clue_provenance, validate_clue_provenance
from .wikidata_client import WikidataClient
from .crossword_csp import build_slots, render_grid, solve_crossword
from .csp_heuristics import (
    build_solver_vocabulary,
    evaluate_quality_gate,
    QUALITY_GATE_MAX_FILLER_RATIO,
    QUALITY_GATE_MIN_CLUED_RATIO,
    QUALITY_GATE_MIN_FILL_PERCENT,
    run_template_trial,
)
from .lexicon import (
    load_lexicon_scores,
    load_word_list,
    lexicon_score_for_token,
    lexicon_score_for_tokens,
)
from .topology import (
    auto_block_long_slots,
    build_grid,
    get_templates,
    score_template_from_length_hist,
    select_best_template,
)
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
    term_length_hists: list[dict[int, int]]
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
class TopologyStageResult:
    diagnostics_path: Path
    diagnostics: dict


@dataclass
class SolveStageResult:
    grid_path: Path
    diagnostics_path: Path
    diagnostics: dict


@dataclass
class PackagingStageResult:
    puzzle_path: Path
    attribution_path: Path
    diagnostics_path: Path
    diagnostics: dict


def _term_length_histogram(
    text: str,
    *,
    lang: str,
    min_len: int = 4,
    max_len: int = 12,
) -> dict[int, int]:
    if not text:
        return {}
    tokens = set(tokenize(text, lang=lang))
    hist: dict[int, int] = {}
    for token in tokens:
        length = len(token)
        if length < min_len or length > max_len:
            continue
        hist[length] = hist.get(length, 0) + 1
    return hist


def _length_histogram_from_answers(answers: set[str]) -> dict[int, int]:
    hist: dict[int, int] = {}
    for answer in answers:
        hist[len(answer)] = hist.get(len(answer), 0) + 1
    return hist


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


def _clue_class_rank(clue_class: str) -> int:
    if clue_class == "source_backed":
        return 2
    if clue_class == "template_fallback":
        return 1
    if clue_class == "synthetic_filler":
        return 0
    return -1


def _is_packageable_clue(clue_row: dict) -> bool:
    text = (clue_row.get("clue") or "").strip()
    return bool(text) and has_packageable_clue_provenance(clue_row)


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


def _support_sentence_count(row: dict) -> int:
    return _safe_int(row.get("support_sentence_count"), default=0)


def _title_support_value(row: dict) -> float:
    return _safe_float(row.get("title_support_score"), default=0.0)


def _is_supported_term(row: dict) -> bool:
    support_keys = {
        "lead_bold_signal",
        "title_support_score",
        "source_title_count",
        "support_sentence_count",
    }
    if not any(key in row for key in support_keys):
        return True
    return bool(
        row.get("lead_bold_signal")
        or _title_support_value(row) > 0.0
        or _safe_int(row.get("source_title_count"), default=0) >= 2
        or _support_sentence_count(row) > 0
    )


def _unique_term_inventory_by_length(terms: list[dict]) -> dict[int, int]:
    by_length: dict[int, set[str]] = {}
    for term in terms:
        normalized = (term.get("normalized_answer") or "").strip().upper()
        if not normalized:
            continue
        by_length.setdefault(len(normalized), set()).add(normalized)
    return {length: len(values) for length, values in by_length.items()}


def _supported_term_inventory_by_length(terms: list[dict]) -> dict[int, int]:
    by_length: dict[int, set[str]] = {}
    for term in terms:
        if not _is_supported_term(term):
            continue
        normalized = (term.get("normalized_answer") or "").strip().upper()
        if not normalized:
            continue
        by_length.setdefault(len(normalized), set()).add(normalized)
    return {length: len(values) for length, values in by_length.items()}


def _supported_inventory_targets(size: int) -> dict[int, int]:
    if size >= 15:
        return {4: 12, 5: 12}
    return {}


def _should_expand_selection_inventory(
    *,
    size: int,
    terms: list[dict],
    solve_diagnostics: dict,
    preferred_fill_target: float = 0.85,
) -> bool:
    if size < 15:
        return False
    fill_percent = _safe_float(solve_diagnostics.get("fill_percent"), default=0.0)
    if fill_percent >= preferred_fill_target and solve_diagnostics.get("fill_status") != "failed":
        return False
    inventory = _supported_term_inventory_by_length(terms)
    targets = _supported_inventory_targets(size)
    return any(inventory.get(length, 0) < target for length, target in targets.items())


def _quality_rescue_target_count(
    *,
    size: int,
    current_selected_count: int,
    terms: list[dict],
) -> int:
    base_target = max(current_selected_count, 12 if size >= 15 else current_selected_count)
    if size < 15:
        return base_target

    inventory = _supported_term_inventory_by_length(terms)
    targets = _supported_inventory_targets(size)
    shortage_extra = min(
        4,
        max(
            *(max(0, target - inventory.get(length, 0)) for length, target in targets.items()),
            0,
        ),
    )
    if current_selected_count >= 12 and shortage_extra > 0:
        return min(16, base_target + shortage_extra)
    return base_target


def _should_relax_min_df_for_quality_rescue(
    *,
    size: int,
    min_df: int,
    terms: list[dict],
    solve_diagnostics: dict,
    preferred_fill_target: float,
) -> bool:
    if min_df <= 1:
        return False
    if size < 15:
        return False
    fill_percent = _safe_float(solve_diagnostics.get("fill_percent"), default=0.0)
    unfilled_short_slot_count = _safe_int(
        solve_diagnostics.get("unfilled_short_slot_count"),
        default=0,
    )
    if fill_percent >= preferred_fill_target and unfilled_short_slot_count <= 2:
        return False
    inventory = _supported_term_inventory_by_length(terms)
    targets = _supported_inventory_targets(size)
    return any(inventory.get(length, 0) < target for length, target in targets.items())


def _should_retry_solve_budget(
    *,
    solve_diagnostics: dict,
    preferred_fill_target: float,
) -> bool:
    fill_percent = _safe_float(solve_diagnostics.get("fill_percent"), default=0.0)
    unfilled_short_slot_count = _safe_int(
        solve_diagnostics.get("unfilled_short_slot_count"),
        default=0,
    )
    return fill_percent < preferred_fill_target or unfilled_short_slot_count > 2


def _load_selected_titles(path: str | Path) -> list[str]:
    import json

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(payload.get("selected_titles", []))


def _write_selected_titles(
    *,
    path: str | Path,
    seed_title: str,
    lang: str,
    titles: list[str],
) -> Path:
    payload = {
        "seed_title": seed_title,
        "lang": lang,
        "selected_k": len(titles),
        "selected_titles": titles,
    }
    return write_json(path, payload)


def _expand_selected_titles_from_scores(
    *,
    selected_titles: list[str],
    candidate_scores_path: str | Path,
    target_count: int,
) -> list[str]:
    import csv

    expanded = list(selected_titles)
    if len(expanded) >= target_count or not Path(candidate_scores_path).exists():
        return expanded

    with Path(candidate_scores_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            title = (row.get("title") or "").strip()
            if not title or title in expanded:
                continue
            expanded.append(title)
            if len(expanded) >= target_count:
                break
    return expanded


def _solve_result_rank(diagnostics: dict) -> tuple[int, float, int, float, float, float]:
    status = diagnostics.get("fill_status")
    if status == "complete":
        status_rank = 2
    elif status == "partial":
        status_rank = 1
    else:
        status_rank = 0
    fill_percent = _safe_float(diagnostics.get("fill_percent"), default=0.0)
    source_backed_ratio = _safe_float(
        diagnostics.get("source_backed_entry_ratio"),
        default=0.0,
    )
    long_slot_theme_ratio = _safe_float(
        diagnostics.get("long_slot_theme_ratio"),
        default=0.0,
    )
    unfilled_short_slot_count = _safe_int(
        diagnostics.get("unfilled_short_slot_count"),
        default=0,
    )
    filler_ratio = _safe_float(
        (diagnostics.get("filler") or {}).get("used_ratio"),
        default=1.0,
    )
    return (status_rank, fill_percent, -unfilled_short_slot_count, source_backed_ratio, long_slot_theme_ratio, -filler_ratio)


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
        min_len=min_len,
        max_len=max_len,
        min_alpha_ratio=min_alpha_ratio,
        stopwords=get_stopwords(lang),
    )
    return {term.normalized_answer for term in merged if term.normalized_answer}


def _fetch_best_extract_with_revid(client: WikiClient, title: str) -> dict:
    try:
        full_payload = client.fetch_page_extract_with_revid(title, intro_only=False)
    except Exception:
        full_payload = {"title": title, "page_id": None, "extract": "", "revid": None}
    if full_payload.get("extract") and full_payload.get("revid"):
        return full_payload
    try:
        intro_payload = client.fetch_page_extract_with_revid(title, intro_only=True)
    except Exception:
        intro_payload = {"title": title, "page_id": None, "extract": "", "revid": None}
    lead_revid = None
    if not full_payload.get("revid") and not intro_payload.get("revid"):
        try:
            lead_payload = client.fetch_lead_wikitext(title)
        except Exception:
            lead_payload = {"title": title, "page_id": None, "revid": None, "wikitext": ""}
        lead_revid = lead_payload.get("revid")
    return {
        "title": full_payload.get("title") or intro_payload.get("title") or title,
        "page_id": full_payload.get("page_id") or intro_payload.get("page_id"),
        "extract": full_payload.get("extract") or intro_payload.get("extract") or "",
        "revid": full_payload.get("revid") or intro_payload.get("revid") or lead_revid,
    }


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
) -> dict:
    clue_word_count = len(clue_text.split())
    clue_score = round(1.0 - abs(clue_word_count - 9) / max(9, clue_word_count), 4)
    return {
        "answer": answer,
        "normalized_answer": normalized_answer or answer_key_value,
        "clue": clue_text,
        "clue_score": clue_score,
        "clue_class": clue_class,
        "source_method": source_method,
        "source_page": source_page,
        "revid": revid,
        "sentence_offset": sentence_offset,
        "oldid_url": build_oldid_url(source_page, revid, lang=lang),
    }


def _clue_candidate_rank(row: dict) -> tuple[float, int, int, str]:
    clue_text = str(row.get("clue") or "")
    sentence_offset = _safe_int(row.get("sentence_offset"), default=9999)
    masked_length_score = -abs(len(clue_text.split()) - 9)
    return (
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
    term_source_method: str,
    lang: str,
    min_words: int,
    max_words: int,
    nlp: object | None,
) -> list[dict]:
    import re

    key = _answer_key(answer, normalized_answer, lang=lang)
    if not key:
        return []

    candidates: list[dict] = []
    seen: set[tuple[str, int, str]] = set()
    for title in source_titles:
        payload = source_payloads.get(title, {})
        source_title = payload.get("title", title) or title
        extract = payload.get("extract", "")
        revid = payload.get("revid")
        for sentence_offset, candidate_sentence in enumerate(split_sentences(extract)):
            if not re.search(re.escape(answer), candidate_sentence, re.IGNORECASE):
                continue
            clue_text = clue_pass_mask_trim(candidate_sentence, answer, max_words=max_words)
            if not clue_pass_validate(
                clue_text,
                answer,
                min_words=min_words,
                nlp=nlp,
                original_sentence=candidate_sentence,
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
                )
            )

    candidates.sort(key=_clue_candidate_rank, reverse=True)
    return candidates


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
        links_payload = client.fetch_links(seed_title, max_links=max_links)
        links = links_payload["links"]
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
            extract_payloads, extract_errors = client.fetch_page_extracts_concurrent(
                parent_titles,
                intro_only=True,
                max_concurrency=8,
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
                max_concurrency=8,
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
    keep_threshold: float = 0.2,
    borderline_threshold: float = 0.1,
    weights: dict[str, float] | None = None,
    lexicon_path: str | Path | None = "data/lexicon/combined_wordfreq.txt",
    lexicon_weight: float = 0.08,
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
            term_length_hists=[],
            term_answer_inventories=[],
        )

    cache = DiskCache(cache_dir)
    api_base = f"https://{lang}.wikipedia.org/w/api.php"
    client = WikiClient(cache, api_base=api_base)

    try:
        seed_extract = client.fetch_page_extract(seed_title, intro_only=True).get("extract", "")
    except Exception as exc:  # pragma: no cover
        seed_extract = ""
        errors.append(f"seed_extract_failed: {exc}")

    nlp, backend_used, backend_errors, _, _ = _resolve_nlp_backend(lang=lang, nlp_backend="auto")
    errors.extend(backend_errors)

    texts = [seed_extract]
    titles = [seed_title]
    term_answer_inventories: list[set[str]] = []
    for candidate in candidates:
        title = candidate["title"]
        try:
            extract = client.fetch_page_extract(title, intro_only=True).get("extract", "")
        except Exception as exc:  # pragma: no cover
            extract = ""
            errors.append(f"candidate_extract_failed:{title}:{exc}")
        try:
            lead_wikitext = client.fetch_lead_wikitext(title).get("wikitext", "")
        except Exception as exc:  # pragma: no cover
            lead_wikitext = ""
            errors.append(f"lead_wikitext_failed:{title}:{exc}")
        titles.append(title)
        texts.append(extract)
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

    term_length_hists = [
        _length_histogram_from_answers(inventory)
        for inventory in term_answer_inventories
    ]
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
    lexicon = load_lexicon_scores(lexicon_path)
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

    ranked_indices = [int(row["index"]) for row in ranked]
    scored_rows: list[dict] = []
    for rank_idx, row in enumerate(ranked):
        idx = int(row["index"])
        rel = rel_scores[idx]
        depth = candidates[idx].get("depth", 1)
        links_back = bool(candidates[idx].get("links_back_to_seed"))
        if rel >= keep_threshold:
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
                "status": status,
                "reason": reason,
            }
        )

    Path(scores_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(scores_path).open("w", encoding="utf-8") as handle:
        handle.write(
            "rank,title,depth,links_back_to_seed,rel_score,lexicon_score,red_score,selection_score,status,reason\n"
        )
        for row in scored_rows:
            handle.write(
                f"{row['rank']},\"{row['title']}\",{row['depth']},{row['links_back_to_seed']},"
                f"{row['rel_score']},{row['lexicon_score']},{row['red_score']},{row['selection_score']},"
                f"{row['status']},{row['reason']}\n"
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
        term_length_hists=term_length_hists,
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
    min_slot_len: int = 3,
    inventory_min_df: int = 2,
    lexicon_path: str | Path | None = "data/lexicon/combined_wordfreq.txt",
    lexicon_weight: float = 0.08,
) -> KSelectionStageResult:
    weights = weights or {"w1": 1.0, "w2": 1.0, "w3": 1.0, "w4": 0.5, "w5": 0.5}
    scoring = run_candidate_scoring_stage(
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

    template_fit_by_k: list[float] = []
    fill_conflict_by_k: list[float] = []
    if term_answer_inventories or scoring.term_length_hists:
        templates = get_templates(size)
        fit_weight = weights.get("w4", 0.5)
        conflict_weight = weights.get("w5", 0.5)
        for k in range(1, max_k_effective + 1):
            selected = ranked_indices[:k]
            if term_answer_inventories:
                inventory_counts: dict[str, int] = {}
                for idx in selected:
                    for answer in term_answer_inventories[idx]:
                        inventory_counts[answer] = inventory_counts.get(answer, 0) + 1
                inventory = {
                    answer
                    for answer, count in inventory_counts.items()
                    if count >= inventory_min_df
                }
                length_hist = _length_histogram_from_answers(inventory)
            else:
                length_hist = {}
                for idx in selected:
                    for length, count in scoring.term_length_hists[idx].items():
                        length_hist[length] = length_hist.get(length, 0) + count
            if not length_hist:
                template_fit_by_k.append(0.0)
                fill_conflict_by_k.append(1.0)
                continue
            max_word_len = max(length_hist.keys(), default=None)
            scored_templates = [
                score_template_from_length_hist(
                    length_hist,
                    template,
                    min_len=min_slot_len,
                    max_word_len=max_word_len,
                    auto_block_long_slots_enabled=True,
                )
                for template in templates
            ]
            # Choose template using the same fit/conflict trade-off as J(K).
            best = max(
                scored_templates,
                key=lambda row: (fit_weight * row["score"]) - (conflict_weight * row["fill_conflict"]),
            )
            template_fit_by_k.append(best["score"])
            fill_conflict_by_k.append(best["fill_conflict"])
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
        template_fit_by_k=template_fit_by_k,
        fill_conflict_by_k=fill_conflict_by_k,
    )

    Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(trace_path).open("w", encoding="utf-8") as handle:
        handle.write("k,jk,delta,coherence,diversity,term_quality,template_fit,fill_conflict,stop\n")
        for row in selection.trace:
            handle.write(
                f"{row['k']},{row['jk']},{row['delta']},{row['coherence']},{row['diversity']},"
                f"{row['term_quality']},{row['template_fit']},{row['fill_conflict']},{row['stop']}\n"
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
            "template_fit_proxy_extracted_answer_inventory",
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


def run_term_extraction_stage(
    *,
    seed_title: str,
    lang: str = "en",
    cache_dir: str | Path = "data/cache/wiki",
    diagnostics_path: str | Path = "outputs/diagnostics_terms.json",
    selected_path: str | Path = "outputs/selected_candidates.json",
    terms_path: str | Path = "outputs/answer_candidates.csv",
    min_len: int = 4,
    max_len: int = 12,
    min_alpha_ratio: float = 0.8,
    min_df: int = 2,
    nlp_backend: str = "auto",
    entity_type_scoring: bool = False,
    wikidata_cache_dir: str | Path = "data/cache/wikidata",
    lexicon_path: str | Path | None = "data/lexicon/combined_wordfreq.txt",
    lexicon_weight: float = 0.15,
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
    intro_extracts: dict[str, str] = {}
    full_payloads: dict[str, dict] = {}
    extracts: dict[str, str] = {}
    for title in selected_titles:
        try:
            intro_extract = client.fetch_page_extract(title, intro_only=True).get("extract", "")
        except Exception as exc:  # pragma: no cover
            intro_extract = ""
            errors.append(f"candidate_extract_failed:{title}:{exc}")
        intro_extracts[title] = intro_extract

        try:
            full_payload = _fetch_best_extract_with_revid(client, title)
        except Exception as exc:  # pragma: no cover
            full_payload = {"title": title, "extract": intro_extract, "revid": None}
            errors.append(f"candidate_full_extract_failed:{title}:{exc}")
        full_payloads[title] = full_payload
        extracts[title] = full_payload.get("extract") or intro_extract

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
    generic_reject_count = 0
    for term, freq in eligible_terms:
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
        generic_penalty = genericness_penalty(
            term.answer,
            term.normalized_answer,
            lang=lang,
            lexicon_score=lexicon_score,
            lead_bold_signal=term.lead_bold_signal,
            title_support_score=title_support_score,
            source_title_count=source_title_count,
            support_sentence_count=support_sentence_count,
        )
        if generic_penalty >= 0.5:
            generic_reject_count += 1
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
            - (0.18 * residual_weight * generic_penalty)
            + (lexicon_weight * lexicon_score)
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
                "genericness_penalty": round(generic_penalty, 4),
                "evidence_tier": evidence_tier,
                "crosswordability_score": round(crossword_score, 4),
                "lexicon_score": round(lexicon_score, 4),
                "shape_penalty": round(shape, 4),
                "answer_score": round(answer_score, 4),
            }
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
                "genericness_penalty",
                "evidence_tier",
                "crosswordability_score",
                "lexicon_score",
                "shape_penalty",
                "answer_score",
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
                    row["genericness_penalty"],
                    row["evidence_tier"],
                    row["crosswordability_score"],
                    row["lexicon_score"],
                    row["shape_penalty"],
                    row["answer_score"],
                ]
            )

    diagnostics = {
        "stage": "term_extraction",
        "lang": lang,
        "seed_title": seed_title,
        "terms_written": True,
        "term_count": len(filtered_terms),
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
        "unsupported_term_reject_count": unsupported_reject_count,
        "generic_term_reject_count": generic_reject_count,
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
    min_required: int = 40,
    max_allowed: int = 250,
) -> VocabGateStageResult:
    errors: list[str] = []
    if not Path(terms_path).exists():
        errors.append("terms_missing")
        diagnostics = {
            "stage": "vocab_gate",
            "lang": lang,
            "seed_title": seed_title,
            "passed": False,
            "reason": "terms_missing",
            "errors": errors,
        }
        final_diagnostics = write_json(diagnostics_path, diagnostics)
        return VocabGateStageResult(diagnostics_path=final_diagnostics, diagnostics=diagnostics)

    lines = Path(terms_path).read_text(encoding="utf-8").splitlines()
    term_count = max(0, len(lines) - 1)
    result = evaluate_vocab_gate(term_count, min_required=min_required, max_allowed=max_allowed)

    diagnostics = {
        "stage": "vocab_gate",
        "lang": lang,
        "seed_title": seed_title,
        "passed": result.passed,
        "reason": result.reason,
        "term_count": result.term_count,
        "min_required": result.min_required,
        "max_allowed": result.max_allowed,
        "errors": errors,
    }
    final_diagnostics = write_json(diagnostics_path, diagnostics)
    return VocabGateStageResult(diagnostics_path=final_diagnostics, diagnostics=diagnostics)


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
    gate_min: int = 40,
    gate_max: int = 250,
    min_len: int = 4,
    max_len: int = 12,
    min_alpha_ratio: float = 0.8,
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

    def _run_terms(min_df_value: int, min_len_value: int) -> TermExtractionStageResult:
        return run_term_extraction_stage(
            seed_title=seed_title,
            lang=lang,
            cache_dir=cache_dir,
            diagnostics_path=terms_diagnostics_path,
            selected_path=selected_path,
            terms_path=terms_path,
            min_len=min_len_value,
            max_len=max_len,
            min_alpha_ratio=min_alpha_ratio,
            min_df=min_df_value,
        )

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

    result = _run_terms(min_df_value=2, min_len_value=min_len)
    gate = evaluate_vocab_gate(len(result.terms), min_required=gate_min, max_allowed=gate_max)
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
    result = _run_terms(min_df_value=1, min_len_value=min_len)
    gate = evaluate_vocab_gate(len(result.terms), min_required=gate_min, max_allowed=gate_max)
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
    promoted_titles: list[str] = []
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
            promoted_titles = borderline
            selected_titles = selected_titles + borderline
            selected_path = _write_selected(Path(selected_override_path), selected_titles)
    else:
        steps.append({"action": "promote_borderline", "value": 0, "note": "candidate_scores_missing"})

    result = run_term_extraction_stage(
        seed_title=seed_title,
        lang=lang,
        cache_dir=cache_dir,
        diagnostics_path=terms_diagnostics_path,
        selected_path=selected_path,
        terms_path=terms_path,
        min_len=min_len,
        max_len=max_len,
        min_alpha_ratio=min_alpha_ratio,
        min_df=1,
    )
    gate = evaluate_vocab_gate(len(result.terms), min_required=gate_min, max_allowed=gate_max)
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

    # Two-hop expansion: re-run seed ingestion with two-hop enabled and retry terms.
    steps.append({"action": "enable_two_hop", "value": True})
    two_hop_result = run_seed_ingestion_stage(
        seed_title=seed_title,
        lang=lang,
        cache_dir=cache_dir,
        diagnostics_path=str(terms_diagnostics_path),
        expansion="one_hop_plus_bounded_two_hop",
        include_backlinks=False,
        max_two_hop_parents=5,
        max_two_hop_links=20,
        max_candidates=100,
    )
    extra_titles = [
        c["title"]
        for c in two_hop_result.diagnostics.get("candidates", [])
        if c.get("depth") == 2 and c["title"] not in selected_titles
    ]
    if extra_titles:
        expanded_titles = selected_titles + extra_titles[:10]
        _write_selected(Path(selected_override_path), expanded_titles)
        result = run_term_extraction_stage(
            seed_title=seed_title,
            lang=lang,
            cache_dir=cache_dir,
            diagnostics_path=terms_diagnostics_path,
            selected_path=selected_override_path,
            terms_path=terms_path,
            min_len=min_len,
            max_len=max_len,
            min_alpha_ratio=min_alpha_ratio,
            min_df=1,
        )
        gate = evaluate_vocab_gate(len(result.terms), min_required=gate_min, max_allowed=gate_max)
        if gate.passed:
            diagnostics = {
                "stage": "rescue_ladder",
                "lang": lang,
                "seed_title": seed_title,
                "passed": True,
                "reason": "enable_two_hop",
                "steps": steps,
                "gate": gate.__dict__,
                "errors": errors,
            }
            final_diagnostics = write_json(diagnostics_path, diagnostics)
            return {"passed": True, "diagnostics": diagnostics, "diagnostics_path": final_diagnostics}

    steps.append({"action": "shorten_min_len", "value": 3})
    result = run_term_extraction_stage(
        seed_title=seed_title,
        lang=lang,
        cache_dir=cache_dir,
        diagnostics_path=terms_diagnostics_path,
        selected_path=selected_path,
        terms_path=terms_path,
        min_len=3,
        max_len=max_len,
        min_alpha_ratio=min_alpha_ratio,
        min_df=1,
    )
    gate = evaluate_vocab_gate(len(result.terms), min_required=gate_min, max_allowed=gate_max)
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

    diagnostics = {
        "stage": "rescue_ladder",
        "lang": lang,
        "seed_title": seed_title,
        "passed": False,
        "reason": "insufficient_vocabulary",
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
    import re

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
                }
            )

    source_cache: dict[str, dict] = {}

    def load_source_payload(title: str) -> dict:
        if title not in source_cache:
            try:
                source_cache[title] = _fetch_best_extract_with_revid(client, title)
            except Exception as exc:  # pragma: no cover
                source_cache[title] = {"title": title, "extract": "", "revid": None}
                errors.append(f"clue_extract_failed:{title}:{exc}")
        return source_cache[title]
    candidate_rows_by_key: dict[str, list[dict]] = {}
    term_by_key: dict[str, dict] = {}
    unique_term_keys: set[str] = set()

    for term in terms:
        answer = term["answer"]
        normalized_answer = term.get("normalized_answer", "")
        key = _answer_key(answer, normalized_answer, lang=lang)
        if not key:
            continue
        unique_term_keys.add(key)
        term_by_key.setdefault(key, term)
        if key in candidate_rows_by_key:
            continue
        source_titles = [title for title in term["source_titles"].split("|") if title]
        for title in source_titles:
            payload = load_source_payload(title)
        candidate_rows_by_key[key] = _collect_source_backed_clue_candidates(
            answer=answer,
            normalized_answer=normalized_answer,
            source_titles=source_titles,
            source_payloads=source_cache,
            term_source_method=term["source_method"],
            lang=lang,
            min_words=min_words,
            max_words=max_words,
            nlp=nlp,
        )

    extracted_by_key = {
        key: rows[0]
        for key, rows in candidate_rows_by_key.items()
        if rows
    }
    extracted_clues = list(extracted_by_key.values())
    diverse_initial, _ = enforce_diversity(
        extracted_clues,
        max_per_bucket=diversity_cap,
        definitional_cap=3,
    )
    kept_initial_keys = {
        _answer_key(clue.get("answer", ""), clue.get("normalized_answer", ""), lang=lang)
        for clue in diverse_initial
    }
    rejected_keys = {key for key in extracted_by_key if key not in kept_initial_keys}

    retry_by_key: dict[str, dict] = {}
    for key in rejected_keys:
        rows = candidate_rows_by_key.get(key, [])
        if len(rows) < 2:
            continue
        retry_by_key[key] = rows[1]

    retry_clues = list(retry_by_key.values())
    retried_attempted = len(retry_clues)
    retried_keys = set(retry_by_key.keys())
    if retry_clues:
        combined = retry_clues + diverse_initial
        diverse_final, final_rejected = enforce_diversity(
            combined,
            max_per_bucket=diversity_cap,
            definitional_cap=3,
        )
        retried_kept = sum(
            1
            for clue in diverse_final
            if _answer_key(clue.get("answer", ""), clue.get("normalized_answer", ""), lang=lang) in retried_keys
        )
    else:
        diverse_final = diverse_initial
        final_rejected = []
        retried_kept = 0

    diverse_final_by_key: dict[str, dict] = {}
    for clue in diverse_final:
        key = _answer_key(clue.get("answer", ""), clue.get("normalized_answer", ""), lang=lang)
        if key and key not in diverse_final_by_key:
            diverse_final_by_key[key] = clue

    final_by_key: dict[str, dict] = dict(extracted_by_key)
    final_by_key.update(retry_by_key)
    final_by_key.update(diverse_final_by_key)

    fallback_template_added = 0
    fallback_template_failed = 0
    fallback_text = "Theme-related entry from the selected source article."
    for key in unique_term_keys:
        if key in final_by_key:
            continue
        term = term_by_key.get(key)
        if term is None:
            fallback_template_failed += 1
            continue
        source_titles = [title for title in term["source_titles"].split("|") if title]
        fallback_title = ""
        fallback_revid = None
        for title in source_titles:
            payload = load_source_payload(title)
            if not fallback_title:
                fallback_title = payload.get("title", title) or title
            if fallback_revid is None:
                fallback_revid = payload.get("revid")
            if fallback_title and fallback_revid:
                break
        if not fallback_title:
            fallback_template_failed += 1
            continue
        final_by_key[key] = {
            "answer": term["answer"],
            "normalized_answer": term.get("normalized_answer") or key,
            "clue": fallback_text,
            "clue_score": 0.05,
            "clue_class": "template_fallback",
            "source_method": term.get("source_method") or "template_fallback",
            "source_page": fallback_title,
            "revid": fallback_revid,
            "sentence_offset": -1,
            "oldid_url": build_oldid_url(fallback_title, fallback_revid, lang=lang),
        }
        fallback_template_added += 1

    clues: list[dict] = []
    seen_keys: set[str] = set()
    for term in terms:
        key = _answer_key(term["answer"], term.get("normalized_answer", ""), lang=lang)
        if not key or key in seen_keys:
            continue
        clue_row = final_by_key.get(key)
        if clue_row is None:
            continue
        clues.append(clue_row)
        seen_keys.add(key)

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
    diagnostics = {
        "stage": "clue_extraction",
        "lang": lang,
        "seed_title": seed_title,
        "clues_written": True,
        "term_count": len(unique_term_keys),
        "extracted_clue_count": len(extracted_clues),
        "diverse_clue_count": len(diverse_final),
        "clue_count": len(clues),
        "spacy_version": clue_spacy_version,
        "spacy_model_version": clue_spacy_model_version,
        "diversity_rejected_count": len(rejected_keys),
        "diversity_retried_attempted": retried_attempted,
        "diversity_retried_count": retried_kept,
        "diversity_rejected_after_retry": len(final_rejected),
        "fallback_template_added": fallback_template_added,
        "fallback_template_failed": fallback_template_failed,
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


def run_topology_selection_stage(
    *,
    seed_title: str,
    lang: str = "en",
    terms_path: str | Path = "outputs/answer_candidates.csv",
    diagnostics_path: str | Path = "outputs/diagnostics_topology.json",
    size: int = 15,
    min_slot_len: int = 3,
    require_gate: bool = True,
    gate_min: int = 40,
    gate_max: int = 250,
) -> TopologyStageResult:
    errors: list[str] = []
    if not Path(terms_path).exists():
        errors.append("terms_missing")
        diagnostics = {
            "stage": "topology_selection",
            "lang": lang,
            "seed_title": seed_title,
            "selected_template": None,
            "errors": errors,
        }
        final_diagnostics = write_json(diagnostics_path, diagnostics)
        return TopologyStageResult(diagnostics_path=final_diagnostics, diagnostics=diagnostics)

    vocabulary = build_solver_vocabulary(
        terms_path=terms_path,
        lang=lang,
        filler_path=None,
        filler_min_len=min_slot_len,
        filler_max_len=size,
        filler_max_per_length=0,
        filler_weight=0.0,
    )
    words = vocabulary.words
    ranking_words = vocabulary.themed_words
    inventory_source = "themed"
    if vocabulary.clue_answers_available:
        ranking_words = sorted(
            vocabulary.clue_answers,
            key=lambda word: (vocabulary.word_scores.get(word, 0.0), word),
            reverse=True,
        )
        inventory_source = "clued"

    gate_result = evaluate_vocab_gate(len(words), min_required=gate_min, max_allowed=gate_max)
    if require_gate and not gate_result.passed:
        errors.append(f"vocab_gate_failed:{gate_result.reason}")
        diagnostics = {
            "stage": "topology_selection",
            "lang": lang,
            "seed_title": seed_title,
            "selected_template": None,
            "gate": gate_result.__dict__,
            "errors": errors,
        }
        final_diagnostics = write_json(diagnostics_path, diagnostics)
        return TopologyStageResult(diagnostics_path=final_diagnostics, diagnostics=diagnostics)

    min_word_len = min((len(word) for word in ranking_words), default=None)
    max_word_len = max((len(word) for word in ranking_words), default=None)
    effective_min_slot_len = min_slot_len
    if min_word_len is not None:
        effective_min_slot_len = max(min_slot_len, min_word_len)

    selection = select_best_template(
        ranking_words,
        size=size,
        min_len=effective_min_slot_len,
        max_word_len=max_word_len,
        auto_block_long_slots_enabled=True,
    )
    diagnostics = {
        "stage": "topology_selection",
        "lang": lang,
        "seed_title": seed_title,
        "selected_template": selection["selected"],
        "scored": selection["scored"],
        "gate": gate_result.__dict__,
        "min_slot_len": min_slot_len,
        "effective_min_slot_len": effective_min_slot_len,
        "min_word_len": min_word_len,
        "max_word_len": max_word_len,
        "inventory_source": inventory_source,
        "ranking_word_count": len(ranking_words),
        "errors": errors,
    }
    final_diagnostics = write_json(diagnostics_path, diagnostics)
    return TopologyStageResult(diagnostics_path=final_diagnostics, diagnostics=diagnostics)


def _slot_pattern(rendered_grid: list[list[str]], slot) -> str:
    letters: list[str] = []
    for row, col in slot.cells:
        cell = rendered_grid[row][col]
        letters.append(cell if cell not in {".", "#"} else ".")
    return "".join(letters)


def _matches_pattern(word: str, pattern: str) -> bool:
    if len(word) != len(pattern):
        return False
    return all(pattern_char == "." or pattern_char == word_char for word_char, pattern_char in zip(word, pattern))


def _count_unfilled_short_slots(slots: list, assignments: dict[int, str], *, max_len: int = 5) -> int:
    return sum(1 for slot in slots if slot.id not in assignments and slot.length <= max_len)


def _apply_short_slot_completion(
    *,
    rendered_grid: list[list[str]],
    slots: list,
    assignments: dict[int, str],
    candidate_words: set[str],
    word_scores: dict[str, float],
    render_grid_fn,
    max_len: int = 5,
    allow_ranked_choice: bool = False,
) -> tuple[dict[int, str], list[list[str]], list[dict[str, int | str]]]:
    assignments_out = dict(assignments)
    used_words = set(assignments_out.values())
    working_grid = [list(row) for row in rendered_grid]
    additions: list[dict[str, int | str]] = []
    candidate_buckets: dict[int, list[str]] = {}
    for word in candidate_words:
        candidate_buckets.setdefault(len(word), []).append(word)

    while True:
        best_choice: tuple[int, int, float, int, object, str] | None = None
        for slot in slots:
            if slot.id in assignments_out or slot.length > max_len:
                continue
            pattern = _slot_pattern(working_grid, slot)
            anchored = sum(1 for char in pattern if char != ".")
            if anchored == 0:
                continue
            matches = [
                word
                for word in candidate_buckets.get(slot.length, [])
                if word not in used_words and _matches_pattern(word, pattern)
            ]
            if not matches:
                continue
            matches.sort(key=lambda word: (word_scores.get(word, 0.0), word), reverse=True)
            if len(matches) != 1:
                if not allow_ranked_choice or anchored < 2 or len(matches) > 3:
                    continue
                top_score = word_scores.get(matches[0], 0.0)
                second_score = word_scores.get(matches[1], 0.0)
                if top_score - second_score < 0.08:
                    continue
            choice = (
                len(matches),
                -anchored,
                -word_scores.get(matches[0], 0.0),
                slot.id,
                slot,
                matches[0],
            )
            if best_choice is None or choice < best_choice:
                best_choice = choice

        if best_choice is None:
            break

        slot = best_choice[4]
        word = best_choice[5]
        assignments_out[slot.id] = word
        used_words.add(word)
        additions.append(
            {
                "slot_id": slot.id,
                "word": word,
                "length": slot.length,
            }
        )
        working_grid = render_grid_fn(working_grid, slots, assignments_out)

    return assignments_out, working_grid, additions


def _summarize_assignment_state(
    *,
    rendered_grid: list[list[str]],
    slots: list,
    assignments: dict[int, str],
    themed_set: set[str],
    clue_answers: set[str],
    source_backed_answers: set[str],
    fallback_only_answers: set[str],
    clue_answers_available: bool,
) -> dict:
    assigned_ids = set(assignments)
    fill_count = sum(1 for row in rendered_grid for cell in row if cell not in (".", "#"))
    total_cells = sum(1 for row in rendered_grid for cell in row if cell != "#")
    fill_percent = 0.0 if total_cells == 0 else fill_count / total_cells
    unfilled_slots = [
        {
            "slot_id": slot.id,
            "direction": slot.direction,
            "length": slot.length,
            "position": slot.cells[0],
        }
        for slot in slots
        if slot.id not in assigned_ids
    ]
    long_slot_ids = {slot.id for slot in slots if slot.length >= 6}
    assigned_count = len(assignments)
    themed_assigned_count = sum(1 for word in assignments.values() if word in themed_set)
    if clue_answers_available:
        clued_assigned_count = sum(1 for word in assignments.values() if word in clue_answers)
    else:
        clued_assigned_count = assigned_count
    filler_used_count = max(0, assigned_count - themed_assigned_count)
    filler_used_ratio = 0.0 if assigned_count == 0 else filler_used_count / assigned_count
    clued_entry_ratio = 0.0 if assigned_count == 0 else clued_assigned_count / assigned_count
    source_backed_entry_count = sum(1 for word in assignments.values() if word in source_backed_answers)
    source_backed_entry_ratio = 0.0 if assigned_count == 0 else source_backed_entry_count / assigned_count
    fallback_only_assigned_count = sum(1 for word in assignments.values() if word in fallback_only_answers)
    fallback_only_entry_ratio = 0.0 if assigned_count == 0 else fallback_only_assigned_count / assigned_count
    long_slot_assigned_count = sum(1 for slot_id in assignments if slot_id in long_slot_ids)
    long_slot_non_theme_count = sum(
        1 for slot_id, word in assignments.items() if slot_id in long_slot_ids and word not in themed_set
    )
    fallback_only_long_count = sum(
        1 for slot_id, word in assignments.items() if slot_id in long_slot_ids and word in fallback_only_answers
    )
    fallback_only_long_ratio = (
        0.0 if long_slot_assigned_count == 0 else fallback_only_long_count / long_slot_assigned_count
    )
    long_slot_theme_ratio = (
        1.0 if long_slot_assigned_count == 0 else (long_slot_assigned_count - long_slot_non_theme_count) / long_slot_assigned_count
    )
    unfilled_short_slot_count = sum(1 for slot in slots if slot.id not in assigned_ids and slot.length <= 5)
    quality_objective = (
        (1.35 * fill_percent)
        + (0.75 * (0.0 if assigned_count == 0 else themed_assigned_count / assigned_count))
        + (0.90 * source_backed_entry_ratio)
        + (0.25 * clued_entry_ratio)
        + (0.35 * long_slot_theme_ratio)
        - (1.10 * filler_used_ratio)
        - (0.75 * fallback_only_entry_ratio)
        - (0.90 * fallback_only_long_ratio)
        - (0.10 * unfilled_short_slot_count)
        - (0.70 * long_slot_non_theme_count)
    )
    return {
        "fill_count": fill_count,
        "total_cells": total_cells,
        "fill_percent": fill_percent,
        "unfilled_slots": unfilled_slots,
        "assigned_count": assigned_count,
        "themed_assigned_count": themed_assigned_count,
        "clued_assigned_count": clued_assigned_count,
        "filler_used_count": filler_used_count,
        "filler_used_ratio": filler_used_ratio,
        "clued_entry_ratio": clued_entry_ratio,
        "source_backed_entry_count": source_backed_entry_count,
        "source_backed_entry_ratio": source_backed_entry_ratio,
        "fallback_only_assigned_count": fallback_only_assigned_count,
        "fallback_only_entry_ratio": fallback_only_entry_ratio,
        "long_slot_assigned_count": long_slot_assigned_count,
        "long_slot_non_theme_count": long_slot_non_theme_count,
        "fallback_only_long_count": fallback_only_long_count,
        "fallback_only_long_ratio": fallback_only_long_ratio,
        "long_slot_theme_ratio": long_slot_theme_ratio,
        "unfilled_short_slot_count": unfilled_short_slot_count,
        "quality_objective": quality_objective,
    }


def run_csp_solve_stage(
    *,
    seed_title: str,
    lang: str = "en",
    terms_path: str | Path = "outputs/answer_candidates.csv",
    diagnostics_path: str | Path = "outputs/diagnostics_csp.json",
    grid_path: str | Path = "outputs/grid.json",
    size: int = 15,
    min_slot_len: int = 3,
    template_name: str | None = None,
    max_steps: int = 20000,
    min_domain: int = 1,
    max_restarts: int = 2,
    random_seed: int = 13,
    use_ac3: bool = True,
    beam_width: int = 32,
    enable_local_repair: bool = True,
    repair_steps: int = 300,
    template_trials: int = 3,
    filler_path: str | Path | None = "data/lexicon/filler_words.txt",
    filler_min_len: int = 3,
    filler_max_len: int = 12,
    filler_max_per_length: int = 1200,
    filler_weight: float = 0.01,
    use_rust: bool | None = None,
    template_priority_names: list[str] | None = None,
    require_gate: bool = True,
    gate_min: int = 40,
    gate_max: int = 250,
    preferred_fill_target: float = 0.85,
) -> SolveStageResult:
    errors: list[str] = []
    if not Path(terms_path).exists():
        errors.append("terms_missing")
        diagnostics = {
            "stage": "csp_solve",
            "lang": lang,
            "seed_title": seed_title,
            "solved": False,
            "errors": errors,
        }
        final_diagnostics = write_json(diagnostics_path, diagnostics)
        return SolveStageResult(
            grid_path=Path(grid_path),
            diagnostics_path=final_diagnostics,
            diagnostics=diagnostics,
        )

    vocabulary = build_solver_vocabulary(
        terms_path=terms_path,
        lang=lang,
        filler_path=filler_path,
        filler_min_len=filler_min_len,
        filler_max_len=filler_max_len,
        filler_max_per_length=filler_max_per_length,
        filler_weight=filler_weight,
    )
    word_scores = vocabulary.word_scores
    words = vocabulary.words
    themed_words = vocabulary.themed_words
    themed_set = vocabulary.themed_set
    clue_answers = vocabulary.clue_answers
    clue_answers_available = vocabulary.clue_answers_available
    source_backed_answers = getattr(vocabulary, "source_backed_answers", set())
    template_fallback_answers = getattr(vocabulary, "template_fallback_answers", set())
    fallback_only_answers = getattr(vocabulary, "fallback_only_answers", set())
    unsupported_answers = getattr(vocabulary, "unsupported_answers", set())
    filler_words = vocabulary.filler_words
    filler_raw_count = vocabulary.filler_raw_count
    filler_added = vocabulary.filler_added
    filler_limit_per_length = vocabulary.filler_limit_per_length
    filler_weight = vocabulary.filler_weight
    long_filler_weight = vocabulary.long_filler_weight
    effective_filler_max_len = getattr(vocabulary, "effective_filler_max_len", filler_max_len)
    min_word_len = vocabulary.min_word_len
    max_word_len = vocabulary.max_word_len

    gate_result = evaluate_vocab_gate(len(themed_words), min_required=gate_min, max_allowed=gate_max)
    if require_gate and not gate_result.passed:
        errors.append(f"vocab_gate_failed:{gate_result.reason}")
        diagnostics = {
            "stage": "csp_solve",
            "lang": lang,
            "seed_title": seed_title,
            "solved": False,
            "gate": gate_result.__dict__,
            "errors": errors,
        }
        final_diagnostics = write_json(diagnostics_path, diagnostics)
        return SolveStageResult(
            grid_path=Path(grid_path),
            diagnostics_path=final_diagnostics,
            diagnostics=diagnostics,
        )

    effective_min_slot_len = min_slot_len
    if min_word_len is not None:
        effective_min_slot_len = max(min_slot_len, min_word_len)
    if clue_answers_available:
        min_clue_word_len = min((len(answer) for answer in clue_answers), default=effective_min_slot_len)
        effective_min_slot_len = max(effective_min_slot_len, min_clue_word_len)

    templates = get_templates(size)
    template_trials = max(1, template_trials)
    if template_name:
        templates_to_try = [next((t for t in templates if t.name == template_name), templates[0])]
        scored_templates = []
    else:
        ranking_words = themed_words
        if source_backed_answers:
            ranking_words = sorted(
                source_backed_answers,
                key=lambda word: (word_scores.get(word, 0.0), word),
                reverse=True,
            )
        elif clue_answers_available:
            ranking_words = sorted(
                clue_answers,
                key=lambda word: (word_scores.get(word, 0.0), word),
                reverse=True,
            )
        selection = select_best_template(
            ranking_words,
            size=size,
            min_len=effective_min_slot_len,
            max_word_len=max_word_len,
            auto_block_long_slots_enabled=True,
        )
        scored_templates = selection.get("scored", [])
        ordered_names = [row["template"] for row in scored_templates]
        if template_priority_names:
            priority_order = {
                name: index for index, name in enumerate(template_priority_names)
            }
            default_order = {
                name: index for index, name in enumerate(ordered_names)
            }
            ordered_names.sort(
                key=lambda name: (
                    priority_order.get(name, len(priority_order)),
                    default_order.get(name, len(default_order)),
                )
            )
        templates_to_try = [
            next(t for t in templates if t.name == name)
            for name in ordered_names[:template_trials]
        ]

    def _status_rank(status: str) -> int:
        if status == "complete":
            return 2
        if status == "partial":
            return 1
        return 0

    import os

    solver_backend = "python"
    solver = solve_crossword
    use_rust_solver = use_rust
    if use_rust_solver is None:
        use_rust_solver = os.getenv("CROSSWORD_USE_RUST", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    if use_rust_solver:
        try:
            import rust_csp

            solver = rust_csp.solve_crossword
            solver_backend = "rust"
        except Exception as exc:
            errors.append(f"rust_solver_unavailable:{exc}")

    template_trials_log: list[dict] = []
    best_trial: dict | None = None
    best_rank: tuple[int, float, int, int, float, float, float, int, float, int, int] | None = None

    for idx, template in enumerate(templates_to_try):
        trial_seed = random_seed + (idx * 101)
        trial = run_template_trial(
            template=template,
            trial_seed=trial_seed,
            solver=solver,
            build_grid_fn=build_grid,
            auto_block_long_slots_fn=auto_block_long_slots,
            build_slots_fn=build_slots,
            render_grid_fn=render_grid,
            words=words,
            themed_words=themed_words,
            themed_set=themed_set,
            clue_answers=clue_answers,
            source_backed_answers=source_backed_answers,
            fallback_only_answers=fallback_only_answers,
            clue_answers_available=clue_answers_available,
            word_scores=word_scores,
            long_filler_weight=long_filler_weight,
            max_word_len=max_word_len,
            effective_min_slot_len=effective_min_slot_len,
            min_domain=min_domain,
            max_steps=max_steps,
            max_restarts=max_restarts,
            use_ac3=use_ac3,
            beam_width=beam_width,
            enable_local_repair=enable_local_repair,
            repair_steps=repair_steps,
        )
        trial_quality_gate_passed, trial_quality_gate_reasons = evaluate_quality_gate(
            fill_percent=trial["fill_percent"],
            invalid_slots=trial["invalid_slots"],
            filler_used_ratio=trial["filler_used_ratio"],
            clued_entry_ratio=trial["clued_entry_ratio"],
            clue_answers_available=clue_answers_available,
            long_slot_non_theme_count=trial["long_slot_non_theme_count"],
        )
        trial_source_backed_entry_ratio = _safe_float(
            trial.get("source_backed_entry_ratio"),
            default=0.0,
        )
        trial_fallback_only_assigned_count = _safe_int(
            trial.get("fallback_only_assigned_count"),
            default=0,
        )
        trial_fallback_only_long_count = _safe_int(
            trial.get("fallback_only_long_count"),
            default=0,
        )
        trial_long_slot_theme_ratio = _safe_float(
            trial.get("long_slot_theme_ratio"),
            default=0.0,
        )
        trial_unfilled_short_slot_count = _safe_int(
            trial.get("unfilled_short_slot_count"),
            default=0,
        )
        template_trials_log.append(
            {
                "template": template.name,
                "fill_status": trial["fill_status"],
                "fill_percent": trial["fill_percent"],
                "fill_count": trial["fill_count"],
                "total_cells": trial["total_cells"],
                "slot_count": len(trial["slots"]),
                "active_slot_count": len(trial["active_slots"]),
                "pruned_slot_count": len(trial["slots"]) - len(trial["active_slots"]),
                "steps": trial["result"]["steps"],
                "restarts": trial["result"].get("restarts", 0),
                "implicit_added_count": trial["implicit_added_count"],
                "removed_assignments_count": len(trial["removed_assignments"]),
                "invalid_slots_count": len(trial["invalid_slots"]),
                "solved_final": trial["solved_final"],
                "quality_objective": trial["quality_objective"],
                "filler_used_ratio": trial["filler_used_ratio"],
                "clued_entry_ratio": trial["clued_entry_ratio"],
                "source_backed_entry_ratio": trial_source_backed_entry_ratio,
                "long_slot_non_theme_count": trial["long_slot_non_theme_count"],
                "long_slot_theme_ratio": trial_long_slot_theme_ratio,
                "unfilled_short_slot_count": trial_unfilled_short_slot_count,
                "fallback_only_assigned_count": trial_fallback_only_assigned_count,
                "fallback_only_long_count": trial_fallback_only_long_count,
                "quality_gate": {
                    "passed": trial_quality_gate_passed,
                    "reasons": trial_quality_gate_reasons,
                },
                "phase_a": trial["phase_a"],
                "auto_block": {
                    "max_word_len": max_word_len,
                    "added_block_count": len(trial["auto_block"]["added_blocks"]),
                    "added_blocks": trial["auto_block"]["added_blocks"],
                    "long_slot_count_before": len(trial["auto_block"]["long_slots_before"]),
                    "long_slot_count_after": len(trial["auto_block"]["long_slots_after"]),
                    "iterations": trial["auto_block"]["iterations"],
                },
                "errors": trial["trial_errors"],
            }
        )

        rank = (
            1 if trial_quality_gate_passed else 0,
            trial["fill_percent"],
            -len(trial["invalid_slots"]),
            -trial_unfilled_short_slot_count,
            trial_source_backed_entry_ratio,
            trial_long_slot_theme_ratio,
            -trial["filler_used_ratio"],
            -trial_fallback_only_assigned_count,
            trial["quality_objective"],
            -trial["long_slot_non_theme_count"],
            _status_rank(trial["fill_status"]),
            trial["fill_count"],
        )
        if best_trial is None or rank > best_rank:
            best_trial = trial
            best_rank = rank
        if (
            trial["fill_status"] == "complete"
            and trial_quality_gate_passed
        ):
            break

    assert best_trial is not None
    template = best_trial["template"]
    rendered = [list(row) for row in best_trial["rendered"]]
    result = dict(best_trial["result"])
    slots = best_trial["slots"]
    active_slots = best_trial["active_slots"]
    auto_block = best_trial["auto_block"]
    implicit_added_count = best_trial["implicit_added_count"]
    invalid_slots = best_trial["invalid_slots"]
    unclued_removed_count = best_trial["unclued_removed_count"]
    completion_log = {
        "preferred_fill_target": preferred_fill_target,
        "source_backed_added": [],
        "template_fallback_added": [],
    }
    current_assignments = {
        int(slot_id): str(word)
        for slot_id, word in result.get("assignments", {}).items()
    }
    short_unfilled_before = _count_unfilled_short_slots(slots, current_assignments)
    if short_unfilled_before > 0:
        current_assignments, rendered, source_backed_additions = _apply_short_slot_completion(
            rendered_grid=rendered,
            slots=slots,
            assignments=current_assignments,
            candidate_words=source_backed_answers,
            word_scores=word_scores,
            render_grid_fn=render_grid,
            max_len=5,
            allow_ranked_choice=False,
        )
        completion_log["source_backed_added"] = source_backed_additions
        if _count_unfilled_short_slots(slots, current_assignments) > 0:
            current_assignments, rendered, template_fallback_additions = _apply_short_slot_completion(
                rendered_grid=rendered,
                slots=slots,
                assignments=current_assignments,
                candidate_words=template_fallback_answers,
                word_scores=word_scores,
                render_grid_fn=render_grid,
                max_len=5,
                allow_ranked_choice=True,
            )
            completion_log["template_fallback_added"] = template_fallback_additions

    completion_applied = bool(completion_log["source_backed_added"] or completion_log["template_fallback_added"])
    if completion_applied:
        result["assignments"] = current_assignments
        summary = _summarize_assignment_state(
            rendered_grid=rendered,
            slots=slots,
            assignments=current_assignments,
            themed_set=themed_set,
            clue_answers=clue_answers,
            source_backed_answers=source_backed_answers,
            fallback_only_answers=fallback_only_answers,
            clue_answers_available=clue_answers_available,
        )
        fill_percent = summary["fill_percent"]
        fill_status = "complete" if len(current_assignments) == len(slots) and not invalid_slots else ("partial" if current_assignments else "failed")
        unfilled_slots = summary["unfilled_slots"]
        solved_final = bool(result.get("solved", False)) or (len(current_assignments) == len(slots) and not invalid_slots)
        assigned_count = summary["assigned_count"]
        themed_assigned_count = summary["themed_assigned_count"]
        clued_assigned_count = summary["clued_assigned_count"]
        filler_used_count = summary["filler_used_count"]
        filler_used_ratio = summary["filler_used_ratio"]
        clued_entry_ratio = summary["clued_entry_ratio"]
        source_backed_entry_count = summary["source_backed_entry_count"]
        source_backed_entry_ratio = summary["source_backed_entry_ratio"]
        fallback_only_assigned_count = summary["fallback_only_assigned_count"]
        fallback_only_entry_ratio = summary["fallback_only_entry_ratio"]
        long_slot_assigned_count = summary["long_slot_assigned_count"]
        long_slot_non_theme_count = summary["long_slot_non_theme_count"]
        fallback_only_long_count = summary["fallback_only_long_count"]
        fallback_only_long_ratio = summary["fallback_only_long_ratio"]
        long_slot_theme_ratio = summary["long_slot_theme_ratio"]
        unfilled_short_slot_count = summary["unfilled_short_slot_count"]
        quality_objective = summary["quality_objective"]
    else:
        fill_percent = best_trial["fill_percent"]
        fill_status = best_trial["fill_status"]
        unfilled_slots = best_trial["unfilled_slots"]
        solved_final = best_trial["solved_final"]
        assigned_count = best_trial["assigned_count"]
        themed_assigned_count = best_trial["themed_assigned_count"]
        clued_assigned_count = best_trial["clued_assigned_count"]
        filler_used_count = best_trial["filler_used_count"]
        filler_used_ratio = best_trial["filler_used_ratio"]
        clued_entry_ratio = best_trial["clued_entry_ratio"]
        source_backed_entry_count = _safe_int(best_trial.get("source_backed_entry_count"), default=0)
        source_backed_entry_ratio = _safe_float(best_trial.get("source_backed_entry_ratio"), default=0.0)
        fallback_only_assigned_count = _safe_int(best_trial.get("fallback_only_assigned_count"), default=0)
        fallback_only_entry_ratio = _safe_float(best_trial.get("fallback_only_entry_ratio"), default=0.0)
        long_slot_assigned_count = best_trial["long_slot_assigned_count"]
        long_slot_non_theme_count = best_trial["long_slot_non_theme_count"]
        fallback_only_long_count = _safe_int(best_trial.get("fallback_only_long_count"), default=0)
        fallback_only_long_ratio = _safe_float(best_trial.get("fallback_only_long_ratio"), default=0.0)
        long_slot_theme_ratio = best_trial["long_slot_theme_ratio"]
        unfilled_short_slot_count = _safe_int(best_trial.get("unfilled_short_slot_count"), default=0)
        quality_objective = best_trial["quality_objective"]
    quality_gate_passed, quality_gate_reasons = evaluate_quality_gate(
        fill_percent=fill_percent,
        invalid_slots=invalid_slots,
        filler_used_ratio=filler_used_ratio,
        clued_entry_ratio=clued_entry_ratio,
        clue_answers_available=clue_answers_available,
        long_slot_non_theme_count=long_slot_non_theme_count,
    )
    if not quality_gate_passed:
        fill_status = "failed"
        solved_final = False
        errors.append("quality_gate_failed")
    if unclued_removed_count > 0:
        errors.append(f"unclued_assignments_present:{unclued_removed_count}")

    grid_artifact = GridArtifact(
        seed_title=seed_title,
        lang=lang,
        template=template.name,
        size=size,
        min_slot_len=min_slot_len,
        effective_min_slot_len=effective_min_slot_len,
        grid=rendered,
        assignments=result["assignments"],
        slots=[
            GridSlotRecord(
                id=slot.id,
                direction=slot.direction,
                length=slot.length,
                cells=slot.cells,
            )
            for slot in slots
        ],
        auto_block={
            "max_word_len": max_word_len,
            "added_block_count": len(auto_block["added_blocks"]),
            "added_blocks": auto_block["added_blocks"],
            "long_slot_count_before": len(auto_block["long_slots_before"]),
            "long_slot_count_after": len(auto_block["long_slots_after"]),
            "iterations": auto_block["iterations"],
        },
        filler={
            "path": str(filler_path) if filler_path is not None else None,
            "loaded": bool(filler_words),
            "min_len": filler_min_len,
            "max_len": effective_filler_max_len,
            "max_per_length": filler_limit_per_length,
            "weight": filler_weight,
            "word_count": len(filler_words),
            "raw_word_count": filler_raw_count,
            "added_count": filler_added,
            "used_count": filler_used_count,
            "used_ratio": filler_used_ratio,
            "strict_filter_enabled": True,
        },
        invalid_slots=invalid_slots,
        fill_status=fill_status,
        fill_percent=fill_percent,
        unfilled_slots=unfilled_slots,
        extras={
            "implicit_assignments_added": implicit_added_count,
            "removed_assignments": best_trial["removed_assignments"],
            "completion": completion_log,
            "quality_gate": {
                "passed": quality_gate_passed,
                "reasons": quality_gate_reasons,
                "objective": quality_objective,
                "min_fill_percent": QUALITY_GATE_MIN_FILL_PERCENT,
                "preferred_fill_target": preferred_fill_target,
                "max_filler_ratio": QUALITY_GATE_MAX_FILLER_RATIO,
                "min_clued_ratio": QUALITY_GATE_MIN_CLUED_RATIO if clue_answers_available else None,
            },
            "assigned_count": assigned_count,
            "themed_assigned_count": themed_assigned_count,
            "clued_assigned_count": clued_assigned_count,
            "clue_answers_available": clue_answers_available,
            "clued_entry_ratio": clued_entry_ratio,
            "source_backed_entry_count": source_backed_entry_count,
            "source_backed_entry_ratio": source_backed_entry_ratio,
            "fallback_only_assigned_count": fallback_only_assigned_count,
            "fallback_only_entry_ratio": fallback_only_entry_ratio,
            "long_slot_assigned_count": long_slot_assigned_count,
            "long_slot_non_theme_count": long_slot_non_theme_count,
            "fallback_only_long_count": fallback_only_long_count,
            "fallback_only_long_ratio": fallback_only_long_ratio,
            "long_slot_theme_ratio": long_slot_theme_ratio,
            "unfilled_short_slot_count": unfilled_short_slot_count,
            "unsupported_answers_count": len(unsupported_answers),
        },
    )
    payload = grid_artifact.to_dict()

    if best_trial["trial_errors"]:
        errors.extend(best_trial["trial_errors"])
    final_grid = write_json(grid_path, payload)
    diagnostics = {
        "stage": "csp_solve",
        "lang": lang,
        "seed_title": seed_title,
        "solved": solved_final,
        "steps": result["steps"],
        "restarts": result.get("restarts", 0),
        "fill_percent": fill_percent,
        "fill_status": fill_status,
        "template": template.name,
        "template_trials": template_trials_log,
        "gate": gate_result.__dict__,
        "min_slot_len": min_slot_len,
        "effective_min_slot_len": effective_min_slot_len,
        "min_word_len": min_word_len,
        "max_word_len": max_word_len,
        "slot_count": len(slots),
        "active_slot_count": len(active_slots),
        "pruned_slot_count": len(slots) - len(active_slots),
        "max_restarts": max_restarts,
        "random_seed": random_seed,
        "use_ac3": use_ac3,
        "beam_width": beam_width,
        "preferred_fill_target": preferred_fill_target,
        "local_repair_enabled": enable_local_repair,
        "repair_steps": repair_steps,
        "local_repair_applied": result.get("local_repair_applied", False),
        "completion": completion_log,
        "word_score_count": len(word_scores),
        "themed_word_count": len(themed_words),
        "filler": payload["filler"],
        "quality_gate": payload["quality_gate"],
        "auto_block": payload["auto_block"],
        "implicit_assignments_added": implicit_added_count,
        "removed_assignments_count": len(best_trial["removed_assignments"]),
        "invalid_slots_count": len(invalid_slots),
        "invalid_slots": invalid_slots,
        "assigned_count": assigned_count,
        "themed_assigned_count": themed_assigned_count,
        "clued_assigned_count": clued_assigned_count,
        "clue_answers_available": clue_answers_available,
        "clue_answers_count": len(clue_answers),
        "clued_entry_ratio": clued_entry_ratio,
        "source_backed_answers_count": len(source_backed_answers),
        "source_backed_entry_count": source_backed_entry_count,
        "source_backed_entry_ratio": source_backed_entry_ratio,
        "filler_used_ratio": filler_used_ratio,
        "fallback_only_answers_count": len(fallback_only_answers),
        "fallback_only_assigned_count": fallback_only_assigned_count,
        "fallback_only_entry_ratio": fallback_only_entry_ratio,
        "long_slot_assigned_count": long_slot_assigned_count,
        "long_slot_non_theme_count": long_slot_non_theme_count,
        "fallback_only_long_count": fallback_only_long_count,
        "fallback_only_long_ratio": fallback_only_long_ratio,
        "long_slot_theme_ratio": long_slot_theme_ratio,
        "unfilled_short_slot_count": unfilled_short_slot_count,
        "unsupported_answers_count": len(unsupported_answers),
        "quality_objective": quality_objective,
        "unclued_removed_count": unclued_removed_count,
        "unfilled_slots": unfilled_slots,
        "solver_backend": solver_backend,
        "errors": errors,
    }
    final_diagnostics = write_json(diagnostics_path, diagnostics)
    return SolveStageResult(
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
) -> PackagingStageResult:
    errors: list[str] = []

    selected_titles: list[str] = []
    selected_k = 0
    selected_artifact: SelectedCandidatesArtifact | None = None
    if Path(selected_path).exists():
        try:
            import json

            payload = json.loads(Path(selected_path).read_text(encoding="utf-8"))
            selected_artifact = SelectedCandidatesArtifact.from_dict(payload)
            selected_titles = selected_artifact.selected_titles
            selected_k = selected_artifact.selected_k
        except Exception as exc:  # pragma: no cover
            errors.append(f"selected_parse_failed:{exc}")
    else:
        errors.append("selected_missing")

    grid_payload: dict = {}
    grid_artifact: GridArtifact | None = None
    if Path(grid_path).exists():
        try:
            import json

            grid_payload = json.loads(Path(grid_path).read_text(encoding="utf-8"))
            grid_artifact = GridArtifact.from_dict(grid_payload)
        except Exception as exc:  # pragma: no cover
            errors.append(f"grid_parse_failed:{exc}")
    else:
        errors.append("grid_missing")

    clues: list[dict] = []
    if Path(clues_path).exists():
        import csv

        with Path(clues_path).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                clues.append(row)
    else:
        errors.append("clues_missing")

    clue_by_normalized: dict[str, dict] = {}
    for clue in clues:
        normalized = clue.get("normalized_answer") or ""
        if normalized:
            existing = clue_by_normalized.get(normalized)
            candidate_rank = (
                1 if _is_packageable_clue(clue) else 0,
                _clue_class_rank(_infer_clue_class(clue)),
            )
            existing_rank = (
                -1,
                -1,
            ) if existing is None else (
                1 if _is_packageable_clue(existing) else 0,
                _clue_class_rank(_infer_clue_class(existing)),
            )
            if existing is None or candidate_rank > existing_rank:
                clue_by_normalized[normalized] = clue

    def _has_valid_clue(clue_row: dict) -> bool:
        return _is_packageable_clue(clue_row)

    template_name = grid_artifact.template if grid_artifact is not None else grid_payload.get("template")
    size = grid_artifact.size if grid_artifact is not None else grid_payload.get("size", 15)
    if grid_artifact is not None:
        min_slot_len = grid_artifact.effective_min_slot_len or grid_artifact.min_slot_len
    else:
        min_slot_len = grid_payload.get("effective_min_slot_len") or grid_payload.get("min_slot_len") or 3

    across_entries: list[dict] = []
    down_entries: list[dict] = []
    unclued_assigned_slots: list[dict] = []
    entry_clue_classes: list[str] = []
    used_clues: list[dict] = []
    if grid_artifact is not None:
        normalized_assignments = dict(grid_artifact.assignments)
    else:
        assignments = grid_payload.get("assignments", {})
        normalized_assignments = {int(k): v for k, v in assignments.items()}
    slot_records: list[dict] = []

    if grid_artifact is not None:
        slot_records = [slot.to_dict() for slot in grid_artifact.slots]
    else:
        raw_slots = grid_payload.get("slots")
        if isinstance(raw_slots, list):
            for row in raw_slots:
                try:
                    slot_records.append(
                        {
                            "id": int(row["id"]),
                            "direction": str(row["direction"]),
                            "length": int(row["length"]),
                            "cells": [tuple(cell) for cell in row["cells"]],
                        }
                    )
                except Exception:
                    continue

    if not slot_records and template_name:
        templates = get_templates(size)
        template = next((t for t in templates if t.name == template_name), None)
        if template is None:
            errors.append(f"template_missing:{template_name}")
        else:
            grid = build_grid(template)
            slots = build_slots(grid, min_len=min_slot_len)
            slot_records = [
                {
                    "id": slot.id,
                    "direction": slot.direction,
                    "length": slot.length,
                    "cells": slot.cells,
                }
                for slot in slots
            ]

    for slot in slot_records:
        answer = normalized_assignments.get(slot["id"])
        if not answer:
            continue
        clue = clue_by_normalized.get(answer, {})
        if not _has_valid_clue(clue):
            unclued_assigned_slots.append(
                {
                    "slot_id": slot["id"],
                    "direction": slot["direction"],
                    "length": slot["length"],
                    "position": slot["cells"][0],
                    "answer": answer,
                }
            )
            continue
        entry = {
            "slot_id": slot["id"],
            "answer": answer,
            "clue": clue.get("clue", ""),
            "row": slot["cells"][0][0],
            "col": slot["cells"][0][1],
            "length": slot["length"],
            "clue_class": _infer_clue_class(clue),
        }
        entry_clue_classes.append(entry["clue_class"])
        used_clues.append(clue)
        if slot["direction"] == "across":
            across_entries.append(entry)
        else:
            down_entries.append(entry)

    fill_status = grid_artifact.fill_status if grid_artifact is not None else grid_payload.get("fill_status", "failed")
    fill_percent = grid_artifact.fill_percent if grid_artifact is not None else grid_payload.get("fill_percent", 0.0)
    unfilled_slots = grid_artifact.unfilled_slots if grid_artifact is not None else grid_payload.get("unfilled_slots", [])
    assigned_slot_count = len(normalized_assignments)
    clued_entry_count = len(across_entries) + len(down_entries)
    source_backed_entry_count = sum(1 for clue_class in entry_clue_classes if clue_class == "source_backed")
    fallback_only_entry_count = sum(1 for clue_class in entry_clue_classes if clue_class == "template_fallback")
    synthetic_filler_clue_count = sum(1 for clue_class in entry_clue_classes if clue_class == "synthetic_filler")
    clued_entry_ratio = 0.0 if assigned_slot_count == 0 else clued_entry_count / assigned_slot_count
    source_backed_entry_ratio = 0.0 if assigned_slot_count == 0 else source_backed_entry_count / assigned_slot_count
    fallback_only_entry_ratio = 0.0 if assigned_slot_count == 0 else fallback_only_entry_count / assigned_slot_count
    puzzle_status = "ok"
    if not selected_titles or not clues:
        puzzle_status = "insufficient_vocabulary"
    elif clued_entry_ratio < 0.90:
        puzzle_status = "insufficient_clues"
        errors.append("clue_coverage_failed")
    elif fill_status == "failed":
        puzzle_status = "insufficient_quality"

    provenance_missing = validate_clue_provenance(used_clues)
    if provenance_missing:
        errors.append("provenance_incomplete")
    if synthetic_filler_clue_count > 0:
        errors.append("synthetic_filler_disallowed")
        if puzzle_status == "ok":
            puzzle_status = "insufficient_clues"

    puzzle_model = PuzzleArtifact(
        seed_title=seed_title,
        lang=lang,
        selected_articles=selected_titles,
        selected_k=selected_k,
        grid_template_id=template_name,
        grid_cells=grid_artifact.grid if grid_artifact is not None else grid_payload.get("grid", []),
        across_entries=across_entries,
        down_entries=down_entries,
        fill_status=fill_status,
        fill_percent=fill_percent,
        unfilled_slots=unfilled_slots,
        puzzle_status=puzzle_status,
        diagnostics={
            "errors": errors,
            "used_clue_provenance_missing_count": len(provenance_missing),
            "assigned_slot_count": assigned_slot_count,
            "clued_entry_count": clued_entry_count,
            "clued_entry_ratio": clued_entry_ratio,
            "used_source_backed_entry_count": source_backed_entry_count,
            "used_source_backed_entry_ratio": source_backed_entry_ratio,
            "used_template_fallback_entry_count": fallback_only_entry_count,
            "used_template_fallback_entry_ratio": fallback_only_entry_ratio,
            "source_backed_entry_count": source_backed_entry_count,
            "source_backed_entry_ratio": source_backed_entry_ratio,
            "fallback_only_entry_count": fallback_only_entry_count,
            "fallback_only_entry_ratio": fallback_only_entry_ratio,
            "unclued_assigned_slots_count": len(unclued_assigned_slots),
            "unclued_assigned_slots": unclued_assigned_slots,
            "packaged_synthetic_filler_count": synthetic_filler_clue_count,
            "synthetic_filler_clue_count": synthetic_filler_clue_count,
        },
        attribution=used_clues,
    )
    puzzle_payload = puzzle_model.to_dict()

    final_puzzle = write_json(puzzle_path, puzzle_payload)
    final_attribution = write_json(attribution_path, {"clues": used_clues})

    diagnostics = {
        "stage": "packaging",
        "lang": lang,
        "seed_title": seed_title,
        "selected_k": selected_k,
        "clue_count": len(clues),
        "fill_status": fill_status,
        "puzzle_status": puzzle_status,
        "assigned_slot_count": assigned_slot_count,
        "clued_entry_count": clued_entry_count,
        "clued_entry_ratio": clued_entry_ratio,
        "used_source_backed_entry_count": source_backed_entry_count,
        "used_source_backed_entry_ratio": source_backed_entry_ratio,
        "used_template_fallback_entry_count": fallback_only_entry_count,
        "used_template_fallback_entry_ratio": fallback_only_entry_ratio,
        "source_backed_entry_count": source_backed_entry_count,
        "source_backed_entry_ratio": source_backed_entry_ratio,
        "fallback_only_entry_count": fallback_only_entry_count,
        "fallback_only_entry_ratio": fallback_only_entry_ratio,
        "unclued_assigned_slots_count": len(unclued_assigned_slots),
        "packaged_synthetic_filler_count": synthetic_filler_clue_count,
        "synthetic_filler_clue_count": synthetic_filler_clue_count,
        "errors": errors,
        "used_clue_provenance_missing_count": len(provenance_missing),
        "provenance_missing_count": len(provenance_missing),
    }
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
    max_len: int = 12,
    min_alpha_ratio: float = 0.8,
    min_df: int = 2,
    nlp_backend: str = "auto",
    entity_type_scoring: bool = False,
    gate_min: int = 40,
    gate_max: int = 250,
    rescue: bool = True,
    clue_min_words: int = 6,
    clue_max_words: int = 12,
    diversity_cap: int = 2,
    size: int = 15,
    min_slot_len: int = 3,
    template_name: str | None = None,
    max_steps: int = 20000,
    min_domain: int = 1,
    max_restarts: int = 2,
    random_seed: int = 13,
    use_ac3: bool = True,
    beam_width: int = 32,
    enable_local_repair: bool = True,
    repair_steps: int = 300,
    template_trials: int = 3,
    filler_path: str | Path | None = "data/lexicon/filler_words.txt",
    filler_min_len: int = 3,
    filler_max_len: int = 12,
    filler_max_per_length: int = 1200,
    filler_weight: float = 0.01,
    use_rust: bool | None = None,
    skip_gate: bool = False,
    use_topology: bool = False,
    preferred_fill_target: float = 0.85,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scores_path = output_dir / "candidate_scores.csv"
    diagnostics_scores = output_dir / "diagnostics_scores.json"
    selected_path = output_dir / "selected_candidates.json"
    trace_path = output_dir / "k_selection_trace.csv"
    diagnostics_k = output_dir / "diagnostics_k.json"
    terms_path = output_dir / "answer_candidates.csv"
    diagnostics_terms = output_dir / "diagnostics_terms.json"
    diagnostics_gate = output_dir / "diagnostics_vocab_gate.json"
    clues_path = output_dir / "clues.csv"
    diagnostics_clues = output_dir / "diagnostics_clues.json"
    diagnostics_rescue = output_dir / "diagnostics_rescue.json"
    diagnostics_terms_rescue = output_dir / "diagnostics_terms_rescue.json"
    selected_rescue_path = output_dir / "selected_candidates_rescue.json"
    selected_quality_rescue_path = output_dir / "selected_candidates_quality_rescue.json"
    grid_path = output_dir / "grid.json"
    diagnostics_csp = output_dir / "diagnostics_csp.json"
    diagnostics_topology = output_dir / "diagnostics_topology.json"
    puzzle_path = output_dir / "puzzle.json"
    attribution_path = output_dir / "attribution.json"
    diagnostics_package = output_dir / "diagnostics_package.json"

    scoring = run_candidate_scoring_stage(
        seed_title=seed_title,
        lang=lang,
        cache_dir=cache_dir,
        diagnostics_path=diagnostics_scores,
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
        weights=candidate_weights,
        lexicon_path=lexicon_path,
        lexicon_weight=candidate_lexicon_weight,
    )

    selection = run_k_selection_stage(
        seed_title=seed_title,
        lang=lang,
        cache_dir=cache_dir,
        diagnostics_path=diagnostics_k,
        scores_path=scores_path,
        trace_path=trace_path,
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
        min_slot_len=min_slot_len,
        inventory_min_df=min_df,
        weights=k_weights,
        lexicon_path=lexicon_path,
        lexicon_weight=candidate_lexicon_weight,
    )

    terms = run_term_extraction_stage(
        seed_title=seed_title,
        lang=lang,
        cache_dir=cache_dir,
        diagnostics_path=diagnostics_terms,
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
    )

    gate = run_vocab_gate_stage(
        seed_title=seed_title,
        lang=lang,
        diagnostics_path=diagnostics_gate,
        terms_path=terms_path,
        min_required=gate_min,
        max_allowed=gate_max,
    )

    selected_for_package = selected_path
    rescue_result = None
    if rescue and not gate.diagnostics.get("passed", False):
        rescue_result = run_rescue_ladder(
            seed_title=seed_title,
            lang=lang,
            cache_dir=cache_dir,
            selected_path=selected_path,
            terms_path=terms_path,
            diagnostics_path=diagnostics_rescue,
            candidate_scores_path=scores_path,
            terms_diagnostics_path=diagnostics_terms_rescue,
            selected_override_path=selected_rescue_path,
            gate_min=gate_min,
            gate_max=gate_max,
            min_len=min_len,
            max_len=max_len,
            min_alpha_ratio=min_alpha_ratio,
        )
        if selected_rescue_path.exists():
            selected_for_package = selected_rescue_path
        gate = run_vocab_gate_stage(
            seed_title=seed_title,
            lang=lang,
            diagnostics_path=diagnostics_gate,
            terms_path=terms_path,
            min_required=gate_min,
            max_allowed=gate_max,
        )

    clues = run_clue_extraction_stage(
        seed_title=seed_title,
        lang=lang,
        cache_dir=cache_dir,
        diagnostics_path=diagnostics_clues,
        terms_path=terms_path,
        clues_path=clues_path,
        min_words=clue_min_words,
        max_words=clue_max_words,
        diversity_cap=diversity_cap,
    )

    topology = run_topology_selection_stage(
        seed_title=seed_title,
        lang=lang,
        terms_path=terms_path,
        diagnostics_path=diagnostics_topology,
        size=size,
        min_slot_len=min_slot_len,
        require_gate=not skip_gate,
        gate_min=gate_min,
        gate_max=gate_max,
    )

    template_for_csp = template_name
    template_priority_names: list[str] | None = None
    if template_for_csp is None and use_topology:
        template_priority_names = [
            row.get("template")
            for row in topology.diagnostics.get("scored", [])
            if row.get("template")
        ]
    current_template_priority_names = template_priority_names

    solve = run_csp_solve_stage(
        seed_title=seed_title,
        lang=lang,
        terms_path=terms_path,
        diagnostics_path=diagnostics_csp,
        grid_path=grid_path,
        size=size,
        min_slot_len=min_slot_len,
        template_name=template_for_csp,
        max_steps=max_steps,
        min_domain=min_domain,
        max_restarts=max_restarts,
        random_seed=random_seed,
        use_ac3=use_ac3,
        beam_width=beam_width,
        enable_local_repair=enable_local_repair,
        repair_steps=repair_steps,
        template_trials=template_trials,
        filler_path=filler_path,
        filler_min_len=filler_min_len,
        filler_max_len=filler_max_len,
        filler_max_per_length=filler_max_per_length,
        filler_weight=filler_weight,
        use_rust=use_rust,
        template_priority_names=current_template_priority_names,
        require_gate=not skip_gate,
        gate_min=gate_min,
        gate_max=gate_max,
        preferred_fill_target=preferred_fill_target,
    )

    if (
        rescue
        and template_name is None
        and _should_expand_selection_inventory(
            size=size,
            terms=terms.terms,
            solve_diagnostics=solve.diagnostics,
            preferred_fill_target=preferred_fill_target,
        )
    ):
        current_selected_titles = _load_selected_titles(selected_for_package)
        expanded_titles = _expand_selected_titles_from_scores(
            selected_titles=current_selected_titles,
            candidate_scores_path=scores_path,
            target_count=_quality_rescue_target_count(
                size=size,
                current_selected_count=len(current_selected_titles),
                terms=terms.terms,
            ),
        )
        rescue_min_df = (
            max(1, min_df - 1)
            if _should_relax_min_df_for_quality_rescue(
                size=size,
                min_df=min_df,
                terms=terms.terms,
                solve_diagnostics=solve.diagnostics,
                preferred_fill_target=preferred_fill_target,
            )
            else min_df
        )
        if len(expanded_titles) > len(current_selected_titles) or rescue_min_df != min_df:
            expanded_selected_path = _write_selected_titles(
                path=selected_quality_rescue_path,
                seed_title=seed_title,
                lang=lang,
                titles=expanded_titles,
            )
            expanded_terms = run_term_extraction_stage(
                seed_title=seed_title,
                lang=lang,
                cache_dir=cache_dir,
                diagnostics_path=diagnostics_terms_rescue,
                selected_path=expanded_selected_path,
                terms_path=terms_path,
                min_len=min_len,
                max_len=max_len,
                min_alpha_ratio=min_alpha_ratio,
                min_df=rescue_min_df,
                nlp_backend=nlp_backend,
                entity_type_scoring=entity_type_scoring,
                wikidata_cache_dir=wikidata_cache_dir,
                lexicon_path=lexicon_path,
                lexicon_weight=term_lexicon_weight,
            )
            expanded_gate = run_vocab_gate_stage(
                seed_title=seed_title,
                lang=lang,
                diagnostics_path=diagnostics_gate,
                terms_path=terms_path,
                min_required=gate_min,
                max_allowed=gate_max,
            )
            if skip_gate or expanded_gate.diagnostics.get("passed", False):
                expanded_clues = run_clue_extraction_stage(
                    seed_title=seed_title,
                    lang=lang,
                    cache_dir=cache_dir,
                    diagnostics_path=diagnostics_clues,
                    terms_path=terms_path,
                    clues_path=clues_path,
                    min_words=clue_min_words,
                    max_words=clue_max_words,
                    diversity_cap=diversity_cap,
                )
                expanded_topology = run_topology_selection_stage(
                    seed_title=seed_title,
                    lang=lang,
                    terms_path=terms_path,
                    diagnostics_path=diagnostics_topology,
                    size=size,
                    min_slot_len=min_slot_len,
                    require_gate=not skip_gate,
                    gate_min=gate_min,
                    gate_max=gate_max,
                )
                expanded_template_for_csp = template_name
                expanded_template_priority_names: list[str] | None = None
                if expanded_template_for_csp is None and use_topology:
                    expanded_template_priority_names = [
                        row.get("template")
                        for row in expanded_topology.diagnostics.get("scored", [])
                        if row.get("template")
                    ]
                expanded_solve = run_csp_solve_stage(
                    seed_title=seed_title,
                    lang=lang,
                    terms_path=terms_path,
                    diagnostics_path=diagnostics_csp,
                    grid_path=grid_path,
                    size=size,
                    min_slot_len=min_slot_len,
                    template_name=expanded_template_for_csp,
                    max_steps=max_steps,
                    min_domain=min_domain,
                    max_restarts=max_restarts,
                    random_seed=random_seed,
                    use_ac3=use_ac3,
                    beam_width=beam_width,
                    enable_local_repair=enable_local_repair,
                    repair_steps=repair_steps,
                    template_trials=template_trials,
                    filler_path=filler_path,
                    filler_min_len=filler_min_len,
                    filler_max_len=filler_max_len,
                    filler_max_per_length=filler_max_per_length,
                    filler_weight=filler_weight,
                    use_rust=use_rust,
                    template_priority_names=expanded_template_priority_names,
                    require_gate=not skip_gate,
                    gate_min=gate_min,
                    gate_max=gate_max,
                    preferred_fill_target=preferred_fill_target,
                )
                if _solve_result_rank(expanded_solve.diagnostics) > _solve_result_rank(solve.diagnostics):
                    selected_for_package = expanded_selected_path
                    terms = expanded_terms
                    gate = expanded_gate
                    clues = expanded_clues
                    topology = expanded_topology
                    solve = expanded_solve
                    current_template_priority_names = expanded_template_priority_names

    if _should_retry_solve_budget(
        solve_diagnostics=solve.diagnostics,
        preferred_fill_target=preferred_fill_target,
    ):
        retried_solve = run_csp_solve_stage(
            seed_title=seed_title,
            lang=lang,
            terms_path=terms_path,
            diagnostics_path=diagnostics_csp,
            grid_path=grid_path,
            size=size,
            min_slot_len=min_slot_len,
            template_name=template_name,
            max_steps=max(max_steps * 2, max_steps + 4000),
            min_domain=min_domain,
            max_restarts=max(max_restarts + 1, 2),
            random_seed=random_seed + 907,
            use_ac3=use_ac3,
            beam_width=max(48, beam_width * 2),
            enable_local_repair=enable_local_repair,
            repair_steps=repair_steps,
            template_trials=max(template_trials, 3),
            filler_path=filler_path,
            filler_min_len=filler_min_len,
            filler_max_len=filler_max_len,
            filler_max_per_length=filler_max_per_length,
            filler_weight=filler_weight,
            use_rust=use_rust,
            template_priority_names=current_template_priority_names,
            require_gate=not skip_gate,
            gate_min=gate_min,
            gate_max=gate_max,
            preferred_fill_target=preferred_fill_target,
        )
        if _solve_result_rank(retried_solve.diagnostics) > _solve_result_rank(solve.diagnostics):
            solve = retried_solve

    package = run_packaging_stage(
        seed_title=seed_title,
        lang=lang,
        selected_path=selected_for_package,
        grid_path=grid_path,
        clues_path=clues_path,
        puzzle_path=puzzle_path,
        attribution_path=attribution_path,
        diagnostics_path=diagnostics_package,
    )

    return {
        "scoring": scoring,
        "selection": selection,
        "terms": terms,
        "gate": gate,
        "rescue": rescue_result,
        "clues": clues,
        "topology": topology,
        "solve": solve,
        "package": package,
        "output_dir": output_dir,
    }
