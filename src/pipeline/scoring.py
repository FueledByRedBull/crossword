from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from ..cache import DiskCache
from ..artifact_models import SelectedCandidatesArtifact
from ..diagnostics import write_json
from ..k_selector import select_k
from ..semantic import build_pairwise_matrix, build_tfidf_vectors, cosine_similarity, mmr_rank
from ..term_extractor import (
    _TITLE_DISAMBIGUATOR,
    extract_terms_lead_bold,
    extract_terms_spacy,
    extract_terms_title_tokens,
    merge_terms,
    title_to_answer,
)
from ..text_normalize import is_sensitive, strip_diacritics, tokenize
from ..clue_builder import answer_pattern, split_sentences
from ..lexicon import load_lexicon_scores, lexicon_score_for_tokens
from ..wiki_client import WikiClient

from .common import (
    ScoringStageResult,
    KSelectionStageResult,
    _resolve_nlp_backend,
    _fetch_extract_with_revid,
    _prefetch,
    _uses_seed_word,
    _name_mentions,
)
from .seed import run_seed_ingestion_stage


@lru_cache(maxsize=1)
def _plain_words(text: str) -> str:
    return " ".join(re.findall(r"\w+", strip_diacritics(text).casefold()))


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


def _names_seed_word(extract: str, seed_title: str, *, lang: str, lexicon: dict[str, float], nlp: object | None) -> bool:
    """Whether the text uses a seed word, other than another form of it that the tagger calls an adverb.

    "thermodynamically stable" names no field, but a seed word in its own written form ("μαθηματικά" for Μαθηματικά)
    always counts, whatever its tag. Without a tagger every use counts.
    """
    if nlp is None:
        return _uses_seed_word(extract, seed_title, lang=lang, lexicon=lexicon)
    seed_words = tokenize(seed_title, lang=lang)
    return any(
        _uses_seed_word(token.text, seed_title, lang=lang, lexicon=lexicon)
        and (token.pos_ != "ADV" or any(word in seed_words for word in tokenize(token.text, lang=lang)))
        for token in nlp(extract)
    )


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
    from ..term_extractor import get_stopwords

    term_lists: list[list] = []
    if backend_used == "spacy" and nlp is not None:
        term_lists.append(extract_terms_spacy(nlp(extract), source_title=title, lang=lang))
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


def _empty_scoring_result(
    seed_title: str,
    lang: str,
    errors: list[str],
    weights: dict[str, float],
    diagnostics_path: str | Path,
    scores_path: str | Path,
) -> ScoringStageResult:
    """The stage result for a seed with no candidates: nothing scored, and the diagnostics saying so."""
    diagnostics = {
        "stage": "candidate_scoring",
        "lang": lang,
        "seed_title": seed_title,
        "errors": errors,
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


def _fetch_seed_texts(errors: list[str], client: WikiClient, seed_title: str) -> tuple[str, str]:
    """The seed's intro extract and its full text."""
    try:
        seed_extract = client.fetch_page_extract(seed_title, intro_only=True).get("extract", "")
    except RuntimeError as exc:  # pragma: no cover
        seed_extract = ""
        errors.append(f"seed_extract_failed: {exc}")
    seed_text = _fetch_extract_with_revid(errors, client, seed_title, full=True)["extract"]
    return seed_extract, seed_text


def _fetch_candidate_page(errors: list[str], client: WikiClient, title: str) -> tuple[dict, str]:
    """The intro payload and the lead wikitext of a candidate page; a failed request leaves its part empty."""
    try:
        payload = client.fetch_page_extract(title, intro_only=True)
    except RuntimeError as exc:  # pragma: no cover
        payload = {}
        errors.append(f"candidate_extract_failed:{title}:{exc}")
    try:
        lead_wikitext = client.fetch_lead_wikitext(title).get("wikitext", "")
    except RuntimeError as exc:  # pragma: no cover
        lead_wikitext = ""
        errors.append(f"lead_wikitext_failed:{title}:{exc}")
    return payload, lead_wikitext


def _term_scores(term_answer_inventories: list[set[str]]) -> list[float]:
    """Each candidate's answer count as a share of the largest count."""
    term_counts = [len(inventory) for inventory in term_answer_inventories]
    max_term_count = max(term_counts) if term_counts else 1
    max_term_count = max(1, max_term_count)
    return [count / max_term_count for count in term_counts]


def _title_lexicon_scores(titles: list[str], lexicon: dict[str, float], lang: str) -> list[float]:
    """The lexicon score of each candidate title; the seed title, first in titles, is left out."""
    title_lexicon_scores: list[float] = []
    for title in titles[1:]:
        title_tokens = tokenize(title, lang=lang)
        title_lexicon_scores.append(lexicon_score_for_tokens(title_tokens, lexicon))
    return title_lexicon_scores


def _rank_candidates(
    candidates: list[dict],
    rel_scores: list[float],
    pairwise: list[list[float]],
    title_lexicon_scores: list[float],
    lexicon_weight: float,
    weights: dict[str, float],
) -> list[dict]:
    """The candidates in MMR order, relevance boosted by the title lexicon score."""
    depth_penalties = [1.0 if c.get("depth", 1) == 2 else 0.0 for c in candidates]
    backlink_bonus = [1.0 if c.get("links_back_to_seed") else 0.0 for c in candidates]

    ranking_rel_scores = [
        rel + (lexicon_weight * lex_score)
        for rel, lex_score in zip(rel_scores, title_lexicon_scores)
    ]

    return mmr_rank(
        rel_scores=ranking_rel_scores,
        pairwise=pairwise,
        depth_penalties=depth_penalties,
        backlink_bonus=backlink_bonus,
        a=weights["a"],
        b=weights["b"],
        c=weights["c"],
        d=weights["d"],
    )


def _candidate_status(group: int) -> tuple[str, str]:
    """The status of a ranked candidate and the reason for it: a sensitive or off-topic page is cut, an on-topic one never is.

    Similarity to the seed text runs low when the seed article is short or broad (most on-topic links of the Greek
    Mathematics article score under 0.1), so it only ranks the on-topic pages. Among them nothing but the cap on
    title answers cuts.
    """
    if group == 3:
        return "CUT", "cut_sensitive"
    if group == 2:
        return "CUT", "cut_off_topic"
    return "KEEP", "on_topic"


def _write_scores_csv(scores_path: str | Path, scored_rows: list[dict]) -> None:
    """Writes the scored candidates to the scores CSV, one row per candidate."""
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
    backlink_titles: list[str] | None = None,
    extra_titles: list[str] | None = None,
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
    if candidates:
        # Pages that link to the seed are one hop away; two-hop pages are not linked by the seed, so no back-link.
        candidates = (
            candidates
            + [{"title": t, "depth": 1, "links_back_to_seed": True} for t in backlink_titles or []]
            + [{"title": t, "depth": 2, "links_back_to_seed": False} for t in extra_titles or []]
        )
    if not candidates:
        return _empty_scoring_result(
            seed_title, lang, seed_result.diagnostics.get("errors", []), weights, diagnostics_path, scores_path
        )

    cache = DiskCache(cache_dir)
    api_base = f"https://{lang}.wikipedia.org/w/api.php"
    client = WikiClient(cache, api_base=api_base)

    candidate_titles = [candidate["title"] for candidate in candidates]
    _prefetch(errors, client.prefetch_intro_extracts, [seed_title, *candidate_titles])
    _prefetch(errors, client.prefetch_lead_wikitext, candidate_titles)

    seed_extract, seed_text = _fetch_seed_texts(errors, client, seed_title)

    nlp, backend_used, backend_errors, _, _ = _resolve_nlp_backend(lang=lang, nlp_backend="auto")
    errors.extend(backend_errors)

    lexicon = load_lexicon_scores(lexicon_path)
    texts = [seed_extract]
    titles = [seed_title]
    term_answer_inventories: list[set[str]] = []
    intro_on_topic: list[bool] = []
    mentions_it: list[bool] = []
    backlink_only = set(backlink_titles or [])
    seed_mentions: list[int] = []
    sensitive: list[bool] = []
    for candidate in candidates:
        title = candidate["title"]
        payload, lead_wikitext = _fetch_candidate_page(errors, client, title)
        extract = payload.get("extract", "")
        titles.append(title)
        texts.append(extract)
        # A derived form ("Roman") counts only where the page is defined; later on it is a passing mention.
        first_sentence = (split_sentences(extract) or [""])[0]
        sensitive.append(
            any(is_sensitive(text, lang, seed_title) for text in (title, payload.get("title") or title, first_sentence))
        )
        # A page that merely links to the seed is about it only when its first sentence uses a seed word, and a page
        # that mentions it (second tier, like the pages the seed names) when a later sentence uses it, unless the
        # use is another form of it tagged as an adverb.
        in_intro = _uses_seed_word(extract, seed_title, lang=lang, lexicon=lexicon)
        in_first = _uses_seed_word(first_sentence, seed_title, lang=lang, lexicon=lexicon) or _uses_seed_word(
            first_sentence, seed_title, lang=lang, lexicon=lexicon, derived=True
        )
        if title in backlink_only:
            intro_on_topic.append(in_first)
            mentions_it.append(
                in_intro and not in_first and _names_seed_word(extract, seed_title, lang=lang, lexicon=lexicon, nlp=nlp)
            )
        else:
            intro_on_topic.append(in_intro or in_first)
            mentions_it.append(False)
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

    term_scores = _term_scores(term_answer_inventories)

    vectors = build_tfidf_vectors(texts, lang=lang)
    seed_vec = vectors[0]
    candidate_vecs = vectors[1:]
    rel_scores = [cosine_similarity(seed_vec, vec) for vec in candidate_vecs]
    pairwise = build_pairwise_matrix(candidate_vecs)

    lexicon_weight = max(0.0, min(1.0, lexicon_weight))
    title_lexicon_scores = _title_lexicon_scores(titles, lexicon, lang)

    ranked = _rank_candidates(candidates, rel_scores, pairwise, title_lexicon_scores, lexicon_weight, weights)

    # A linked page is on topic when its intro uses a seed word or the seed article names it twice. Pages whose
    # intro does is first, then pages the seed merely names, then the rest, then the sensitive pages; MMR order
    # within each group (stable).
    group = [
        3 if blocked else 0 if intro else 1 if mentions >= 2 or mentioned else 2
        for blocked, intro, mentions, mentioned in zip(sensitive, intro_on_topic, seed_mentions, mentions_it)
    ]
    ranked.sort(key=lambda row: group[int(row["index"])])

    # The sensitive pages rank last and are no candidate for the selection, which pairs these indices with the rows.
    ranked_indices = [int(row["index"]) for row in ranked if group[int(row["index"])] < 3]
    scored_rows: list[dict] = []
    for rank_idx, row in enumerate(ranked):
        idx = int(row["index"])
        rel = rel_scores[idx]
        depth = candidates[idx].get("depth", 1)
        links_back = bool(candidates[idx].get("links_back_to_seed"))
        status, reason = _candidate_status(group[idx])
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

    _write_scores_csv(scores_path, scored_rows)

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
