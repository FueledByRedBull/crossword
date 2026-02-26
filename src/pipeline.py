from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .cache import DiskCache
from .diagnostics import build_seed_stage_diagnostics, write_json
from .k_selector import select_k
from .semantic import build_pairwise_matrix, build_tfidf_vectors, cosine_similarity, mmr_rank
from .term_extractor import (
    extract_terms_lead_bold,
    extract_terms_nltk,
    extract_terms_spacy,
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
)
from .provenance import build_oldid_url, validate_clue_provenance
from .wikidata_client import WikidataClient
from .crossword_csp import build_slots, render_grid, solve_crossword
from .lexicon import load_lexicon_scores, lexicon_score_for_token, lexicon_score_for_tokens
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


def _safe_float(value, *, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


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
            for title in parent_titles:
                try:
                    extract = client.fetch_page_extract(title, intro_only=True).get("extract", "")
                except Exception as exc:  # pragma: no cover - runtime/network path
                    extract = ""
                    errors.append(f"two_hop_parent_extract_failed:{title}:{exc}")
                parent_extracts.append(extract)

            vectors = build_tfidf_vectors([seed_extract] + parent_extracts, lang=lang)
            seed_vec = vectors[0]
            rel_scores = [cosine_similarity(seed_vec, vec) for vec in vectors[1:]]
            ranked = sorted(
                zip(parent_titles, rel_scores), key=lambda item: item[1], reverse=True
            )
            limit = max_two_hop_parents or len(ranked)
            two_hop_parents = [title for title, _ in ranked[:limit]]

        for parent_title in two_hop_parents:
            try:
                payload = client.fetch_links(parent_title, max_links=max_two_hop_links)
            except Exception as exc:  # pragma: no cover - runtime/network path
                errors.append(f"two_hop_links_failed:{parent_title}:{exc}")
                continue
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
    lexicon_path: str | Path | None = "data/lexicon/scowl_wordfreq.txt",
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
        )

    cache = DiskCache(cache_dir)
    api_base = f"https://{lang}.wikipedia.org/w/api.php"
    client = WikiClient(cache, api_base=api_base)

    try:
        seed_extract = client.fetch_page_extract(seed_title, intro_only=True).get("extract", "")
    except Exception as exc:  # pragma: no cover
        seed_extract = ""
        errors.append(f"seed_extract_failed: {exc}")

    texts = [seed_extract]
    titles = [seed_title]
    for candidate in candidates:
        title = candidate["title"]
        try:
            extract = client.fetch_page_extract(title, intro_only=True).get("extract", "")
        except Exception as exc:  # pragma: no cover
            extract = ""
            errors.append(f"candidate_extract_failed:{title}:{exc}")
        titles.append(title)
        texts.append(extract)

    term_length_hists = [
        _term_length_histogram(text, lang=lang, min_len=4, max_len=12) for text in texts[1:]
    ]
    term_counts = [sum(hist.values()) for hist in term_length_hists]
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
    lexicon_path: str | Path | None = "data/lexicon/scowl_wordfreq.txt",
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
    max_k_effective = min(max_k or len(ranked_indices), len(ranked_indices))

    template_fit_by_k: list[float] = []
    fill_conflict_by_k: list[float] = []
    if scoring.term_length_hists:
        templates = get_templates(size)
        fit_weight = weights.get("w4", 0.5)
        conflict_weight = weights.get("w5", 0.5)
        for k in range(1, max_k_effective + 1):
            selected = ranked_indices[:k]
            length_hist: dict[int, int] = {}
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
        min_k=min_k,
        max_k=max_k,
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
    selected_payload = {
        "seed_title": seed_title,
        "lang": lang,
        "selected_k": selection.selected_k,
        "selected_titles": selected_titles,
    }
    final_selected = write_json(selected_path, selected_payload)

    diagnostics = {
        "stage": "k_selection",
        "lang": lang,
        "seed_title": seed_title,
        "selected_k": selection.selected_k,
        "trace_written": True,
        "selected_written": True,
        "weights": weights,
        "thresholds": {"keep": keep_threshold, "borderline": borderline_threshold},
        "errors": mapping_errors,
        "notes": [
            "pairwise_similarity_cosine",
            "term_quality_proxy_unique_token_count",
            "template_fit_proxy_length_hist",
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
    lexicon_path: str | Path | None = "data/lexicon/scowl_wordfreq.txt",
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
            selected_titles = selected.get("selected_titles", [])
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

    nlp_backend = nlp_backend.lower()
    allowed_backends = {"auto", "spacy", "nltk"}
    if nlp_backend not in allowed_backends:
        errors.append(f"unsupported_nlp_backend:{nlp_backend}")
        nlp_backend = "auto"

    nlp = None
    backend_used = None
    if nlp_backend in {"auto", "spacy"}:
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

        spacy_version = getattr(spacy, "__version__", None) if spacy is not None else None
        spacy_model_version = None
        if nlp is not None:
            try:
                spacy_model_version = nlp.meta.get("version")
            except Exception:  # pragma: no cover
                pass

    if backend_used is None and nlp_backend in {"auto", "nltk"}:
        try:
            import nltk  # noqa: F401

            backend_used = "nltk"
        except Exception as exc:  # pragma: no cover
            errors.append(f"nltk_unavailable:{exc}")
            backend_used = None

    lead_terms: dict[str, list[str]] = {}
    extracts: dict[str, str] = {}
    for title in selected_titles:
        try:
            extract = client.fetch_page_extract(title, intro_only=True).get("extract", "")
        except Exception as exc:  # pragma: no cover
            extract = ""
            errors.append(f"candidate_extract_failed:{title}:{exc}")
        extracts[title] = extract

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

    from .term_extractor import get_stopwords

    stopwords = get_stopwords(lang)
    merged = merge_terms(
        term_lists,
        min_len=min_len,
        max_len=max_len,
        min_alpha_ratio=min_alpha_ratio,
        stopwords=stopwords,
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

    filtered_terms = []
    for term, freq in eligible_terms:
        theme_score = 0.0 if max_df == 0 else freq / max_df
        shape = shape_penalty(term.normalized_answer)
        crossword_score = crosswordability_score(term.normalized_answer)
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
        residual_weight = max(0.0, 1.0 - lexicon_weight)
        answer_score = (
            (0.4 * residual_weight * theme_score)
            + (0.3 * residual_weight * entity_score)
            + (0.2 * residual_weight * lead_bold_bonus)
            + (0.1 * residual_weight * crossword_score)
            + (lexicon_weight * lexicon_score)
        )

        filtered_terms.append(
            {
                "answer": term.answer,
                "normalized_answer": term.normalized_answer,
                "length": term.length,
                "source_method": term.source_method,
                "lead_bold_signal": term.lead_bold_signal,
                "source_titles": "|".join(sorted(term.source_titles)),
                "source_mentions": sorted(term.source_titles),
                "doc_frequency": freq,
                "theme_score": round(theme_score, 4),
                "entity_type_score": round(entity_score, 4),
                "crosswordability_score": round(crossword_score, 4),
                "lexicon_score": round(lexicon_score, 4),
                "shape_penalty": round(shape, 4),
                "answer_score": round(answer_score, 4),
            }
        )

    filtered_terms.sort(
        key=lambda row: (
            row["answer_score"],
            row["doc_frequency"],
            row["crosswordability_score"],
            row["length"],
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
                "source_titles",
                "doc_frequency",
                "theme_score",
                "entity_type_score",
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
                    row["source_titles"],
                    row["doc_frequency"],
                    row["theme_score"],
                    row["entity_type_score"],
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
    max_allowed: int = 80,
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
    gate_max: int = 80,
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

    terms = []
    with Path(terms_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            terms.append(
                {
                    "answer": row["answer"],
                    "normalized_answer": row.get("normalized_answer", ""),
                    "source_titles": row.get("source_titles", ""),
                    "source_method": row.get("source_method", ""),
                }
            )

    clues: list[dict] = []
    source_cache: dict[str, dict] = {}
    for term in terms:
        answer = term["answer"]
        normalized_answer = term.get("normalized_answer", "")
        source_method = term["source_method"]
        source_titles = [t for t in term["source_titles"].split("|") if t]
        sentence = None
        sentence_offset = None
        source_title = None
        source_revid = None
        for title in source_titles:
            if title not in source_cache:
                try:
                    source_cache[title] = client.fetch_page_extract_with_revid(
                        title, intro_only=True
                    )
                except Exception as exc:  # pragma: no cover
                    source_cache[title] = {"title": title, "extract": "", "revid": None}
                    errors.append(f"clue_extract_failed:{title}:{exc}")
            payload = source_cache[title]
            extract = payload.get("extract", "")
            sentence, sentence_offset = clue_pass_extract_with_offset(extract, answer)
            if sentence:
                source_title = payload.get("title", title)
                source_revid = payload.get("revid")
                break

        if not sentence:
            continue
        clue = clue_pass_mask_trim(sentence, answer, max_words=max_words)
        if not clue_pass_validate(
            clue, answer, min_words=min_words, nlp=nlp, original_sentence=sentence
        ):
            continue
        # clue_score: proxy based on word count closeness to ideal range (6-12 words).
        clue_word_count = len(clue.split())
        clue_score = round(
            1.0 - abs(clue_word_count - 9) / max(9, clue_word_count), 4
        )
        clues.append(
            {
                "answer": answer,
                "normalized_answer": normalized_answer,
                "clue": clue,
                "clue_score": clue_score,
                "source_method": source_method,
                "source_page": source_title or "",
                "revid": source_revid,
                "sentence_offset": sentence_offset,
                "oldid_url": build_oldid_url(source_title or "", source_revid, lang=lang),
            }
        )

    clues, rejected_answers = enforce_diversity(clues, max_per_bucket=diversity_cap, definitional_cap=3)

    # Re-extraction pass for diversity-rejected answers (PLAN §5.5 Pass 4).
    # Try alternate sentences from source articles, preserving provenance.
    if rejected_answers:
        rejected_set = set(rejected_answers)
        existing_answers = {c.get("normalized_answer", c["answer"]).upper() for c in clues}
        for term in terms:
            answer = term["answer"]
            normalized_answer = term.get("normalized_answer", "")
            if answer not in rejected_set:
                continue
            if normalized_answer.upper() in existing_answers:
                continue
            source_titles_list = [t for t in term["source_titles"].split("|") if t]
            retry_sentence = None
            retry_offset = None
            retry_title = None
            retry_revid = None
            for title in source_titles_list:
                if title not in source_cache:
                    continue
                payload = source_cache[title]
                extract = payload.get("extract", "")
                # Try extracting all sentences, skip the one already used
                from .clue_builder import split_sentences as _split_sentences
                all_sentences = _split_sentences(extract)
                for s_offset, candidate_sentence in enumerate(all_sentences):
                    if answer.lower() not in candidate_sentence.lower():
                        continue
                    trial_clue = clue_pass_mask_trim(candidate_sentence, answer, max_words=max_words)
                    if clue_pass_validate(
                        trial_clue, answer, min_words=min_words, nlp=nlp, original_sentence=candidate_sentence
                    ):
                        retry_sentence = candidate_sentence
                        retry_offset = s_offset
                        retry_title = payload.get("title", title)
                        retry_revid = payload.get("revid")
                        break
                if retry_sentence:
                    break
            if retry_sentence and retry_title is not None:
                retry_clue = clue_pass_mask_trim(retry_sentence, answer, max_words=max_words)
                clue_word_count = len(retry_clue.split())
                clue_score = round(
                    1.0 - abs(clue_word_count - 9) / max(9, clue_word_count), 4
                )
                clues.append(
                    {
                        "answer": answer,
                        "normalized_answer": normalized_answer,
                        "clue": retry_clue,
                        "clue_score": clue_score,
                        "source_method": term["source_method"],
                        "source_page": retry_title,
                        "revid": retry_revid,
                        "sentence_offset": retry_offset,
                        "oldid_url": build_oldid_url(retry_title, retry_revid, lang=lang),
                    }
                )
                existing_answers.add(normalized_answer.upper())

    Path(clues_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(clues_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "answer",
                "normalized_answer",
                "clue",
                "clue_score",
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
                    row.get("source_method", ""),
                    row.get("source_page", ""),
                    row.get("revid", ""),
                    row.get("sentence_offset", ""),
                    row.get("oldid_url", ""),
                ]
            )

    provenance_missing = validate_clue_provenance(clues)
    diagnostics = {
        "stage": "clue_extraction",
        "lang": lang,
        "seed_title": seed_title,
        "clues_written": True,
        "clue_count": len(clues),
        "spacy_version": clue_spacy_version,
        "spacy_model_version": clue_spacy_model_version,
        "diversity_rejected_count": len(rejected_answers),
        "diversity_retried_count": len(clues) - (len(clues) - len([
            a for a in rejected_answers
            if any(c.get("answer") == a for c in clues)
        ])),
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
    gate_max: int = 80,
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

    import csv

    words: list[str] = []
    word_scores: dict[str, float] = {}
    with Path(terms_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            normalized = (row.get("normalized_answer") or "").strip().upper()
            if not normalized:
                continue
            answer_score = _safe_float(row.get("answer_score"), default=0.0)
            lexicon_score = _safe_float(row.get("lexicon_score"), default=0.0)
            crossword_score = _safe_float(row.get("crosswordability_score"), default=0.0)
            composite = (0.6 * answer_score) + (0.25 * lexicon_score) + (0.15 * crossword_score)
            previous = word_scores.get(normalized)
            if previous is None or composite > previous:
                word_scores[normalized] = composite
    words = sorted(word_scores, key=lambda word: (word_scores[word], word), reverse=True)

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

    min_word_len = min((len(word) for word in words), default=None)
    max_word_len = max((len(word) for word in words), default=None)
    effective_min_slot_len = min_slot_len
    if min_word_len is not None:
        effective_min_slot_len = max(min_slot_len, min_word_len)

    selection = select_best_template(
        words,
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
        "errors": errors,
    }
    final_diagnostics = write_json(diagnostics_path, diagnostics)
    return TopologyStageResult(diagnostics_path=final_diagnostics, diagnostics=diagnostics)


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
    require_gate: bool = True,
    gate_min: int = 40,
    gate_max: int = 80,
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

    import csv

    words: list[str] = []
    word_scores: dict[str, float] = {}
    with Path(terms_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            normalized = (row.get("normalized_answer") or "").strip().upper()
            if not normalized:
                continue
            answer_score = _safe_float(row.get("answer_score"), default=0.0)
            lexicon_score = _safe_float(row.get("lexicon_score"), default=0.0)
            crossword_score = _safe_float(row.get("crosswordability_score"), default=0.0)
            composite = (0.6 * answer_score) + (0.25 * lexicon_score) + (0.15 * crossword_score)
            previous = word_scores.get(normalized)
            if previous is None or composite > previous:
                word_scores[normalized] = composite
    words = sorted(word_scores, key=lambda word: (word_scores[word], word), reverse=True)

    gate_result = evaluate_vocab_gate(len(words), min_required=gate_min, max_allowed=gate_max)
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

    min_word_len = min((len(word) for word in words), default=None)
    max_word_len = max((len(word) for word in words), default=None)
    effective_min_slot_len = min_slot_len
    if min_word_len is not None:
        effective_min_slot_len = max(min_slot_len, min_word_len)

    templates = get_templates(size)
    if template_name:
        template = next((t for t in templates if t.name == template_name), templates[0])
    else:
        selection = select_best_template(
            words,
            size=size,
            min_len=effective_min_slot_len,
            max_word_len=max_word_len,
            auto_block_long_slots_enabled=True,
        )
        template = next(t for t in templates if t.name == selection["selected"])

    grid = build_grid(template)
    auto_block = auto_block_long_slots(grid, max_slot_len=max_word_len, symmetric=True)
    grid = auto_block["grid"]
    slots = build_slots(grid, min_len=effective_min_slot_len)
    domains = {slot.id: [word for word in words if len(word) == slot.length] for slot in slots}
    active_slots = [slot for slot in slots if len(domains[slot.id]) >= min_domain]

    if not active_slots:
        errors.append("no_active_slots")
        result = {"solved": False, "assignments": {}, "steps": 0, "restarts": 0}
    else:
        result = solve_crossword(
            grid,
            active_slots,
            words,
            min_len=effective_min_slot_len,
            max_steps=max_steps,
            max_restarts=max_restarts,
            random_seed=random_seed,
            use_ac3=use_ac3,
            word_scores=word_scores,
            beam_width=beam_width,
            enable_local_repair=enable_local_repair,
            repair_steps=repair_steps,
        )

    rendered = render_grid(grid, active_slots, result["assignments"])

    payload = {
        "seed_title": seed_title,
        "lang": lang,
        "template": template.name,
        "size": size,
        "min_slot_len": min_slot_len,
        "effective_min_slot_len": effective_min_slot_len,
        "grid": rendered,
        "assignments": result["assignments"],
        "slots": [
            {
                "id": slot.id,
                "direction": slot.direction,
                "length": slot.length,
                "cells": slot.cells,
            }
            for slot in slots
        ],
        "auto_block": {
            "max_word_len": max_word_len,
            "added_block_count": len(auto_block["added_blocks"]),
            "added_blocks": auto_block["added_blocks"],
            "long_slot_count_before": len(auto_block["long_slots_before"]),
            "long_slot_count_after": len(auto_block["long_slots_after"]),
            "iterations": auto_block["iterations"],
        },
        "fill_status": None,
        "fill_percent": 0.0,
        "unfilled_slots": [],
    }

    fill_count = sum(1 for row in rendered for cell in row if cell not in (".", "#"))
    total_cells = sum(1 for row in rendered for cell in row if cell != "#")
    fill_percent = 0.0 if total_cells == 0 else fill_count / total_cells
    unfilled_slots = []
    assigned_ids = set(result["assignments"].keys())
    for slot in slots:
        if slot.id in assigned_ids:
            continue
        unfilled_slots.append(
            {
                "slot_id": slot.id,
                "direction": slot.direction,
                "length": slot.length,
                "position": slot.cells[0],
            }
        )

    if result["solved"] and active_slots:
        fill_status = "complete"
    elif result["assignments"]:
        fill_status = "partial"
    else:
        fill_status = "failed"

    payload["fill_status"] = fill_status
    payload["fill_percent"] = fill_percent
    payload["unfilled_slots"] = unfilled_slots
    final_grid = write_json(grid_path, payload)
    diagnostics = {
        "stage": "csp_solve",
        "lang": lang,
        "seed_title": seed_title,
        "solved": result["solved"],
        "steps": result["steps"],
        "restarts": result.get("restarts", 0),
        "fill_percent": fill_percent,
        "fill_status": fill_status,
        "template": template.name,
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
        "local_repair_enabled": enable_local_repair,
        "repair_steps": repair_steps,
        "local_repair_applied": result.get("local_repair_applied", False),
        "word_score_count": len(word_scores),
        "auto_block": payload["auto_block"],
        "unfilled_slots": unfilled_slots,
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
    if Path(selected_path).exists():
        try:
            import json

            payload = json.loads(Path(selected_path).read_text(encoding="utf-8"))
            selected_titles = payload.get("selected_titles", [])
            selected_k = payload.get("selected_k", len(selected_titles))
        except Exception as exc:  # pragma: no cover
            errors.append(f"selected_parse_failed:{exc}")
    else:
        errors.append("selected_missing")

    grid_payload: dict = {}
    if Path(grid_path).exists():
        try:
            import json

            grid_payload = json.loads(Path(grid_path).read_text(encoding="utf-8"))
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
            clue_by_normalized[normalized] = clue

    template_name = grid_payload.get("template")
    size = grid_payload.get("size", 15)
    min_slot_len = grid_payload.get("effective_min_slot_len") or grid_payload.get("min_slot_len") or 3

    across_entries: list[dict] = []
    down_entries: list[dict] = []
    assignments = grid_payload.get("assignments", {})
    normalized_assignments = {int(k): v for k, v in assignments.items()}
    slot_records: list[dict] = []

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
        entry = {
            "slot_id": slot["id"],
            "answer": answer,
            "clue": clue.get("clue", ""),
            "row": slot["cells"][0][0],
            "col": slot["cells"][0][1],
            "length": slot["length"],
        }
        if slot["direction"] == "across":
            across_entries.append(entry)
        else:
            down_entries.append(entry)

    fill_status = grid_payload.get("fill_status", "failed")
    fill_percent = grid_payload.get("fill_percent", 0.0)
    unfilled_slots = grid_payload.get("unfilled_slots", [])
    puzzle_status = "ok"
    if not selected_titles or not clues:
        puzzle_status = "insufficient_vocabulary"

    provenance_missing = validate_clue_provenance(clues)
    if provenance_missing:
        errors.append("provenance_incomplete")

    puzzle_payload = {
        "seed_title": seed_title,
        "lang": lang,
        "selected_articles": selected_titles,
        "selected_k": selected_k,
        "grid_template_id": template_name,
        "grid_cells": grid_payload.get("grid", []),
        "across_entries": across_entries,
        "down_entries": down_entries,
        "fill_status": fill_status,
        "fill_percent": fill_percent,
        "unfilled_slots": unfilled_slots,
        "puzzle_status": puzzle_status,
        "diagnostics": {
            "errors": errors,
            "provenance_missing_count": len(provenance_missing),
        },
        "attribution": clues,
    }

    final_puzzle = write_json(puzzle_path, puzzle_payload)
    final_attribution = write_json(attribution_path, {"clues": clues})

    diagnostics = {
        "stage": "packaging",
        "lang": lang,
        "seed_title": seed_title,
        "selected_k": selected_k,
        "clue_count": len(clues),
        "fill_status": fill_status,
        "puzzle_status": puzzle_status,
        "errors": errors,
        "provenance_missing_count": len(provenance_missing),
    }
    final_diagnostics = write_json(diagnostics_path, diagnostics)
    return PackagingStageResult(
        puzzle_path=final_puzzle,
        attribution_path=final_attribution,
        diagnostics_path=final_diagnostics,
        diagnostics=diagnostics,
    )
