from __future__ import annotations

from pathlib import Path

from ..cache import DiskCache
from ..diagnostics import write_json
from ..vocab_gate import VocabGateResult, evaluate_vocab_gate
from ..wiki_client import WikiClient

from .common import VocabGateStageResult, ClueStageResult, _clued_answer_count, _read_csv_rows
from .scoring import run_candidate_scoring_stage, run_k_selection_stage
from .clues import run_clue_extraction_stage
from .answers import run_term_extraction_stage, run_vocab_gate_stage
from .packaging import run_layout_stage, run_packaging_stage


# Widening for a thin seed: link lists of at most 6 selected articles (one request each), at most 100 new pages
# (five intro batches of 20), 6 of them selected. Pages that link to the seed: at most 300 new ones (15 intro
# batches of 20). The 6 best new on-topic pages join the selection and further ones replace selected articles that
# were cut.
_RESCUE_PARENTS = 6
_RESCUE_NEW_PAGES = 100
_RESCUE_SELECTED = 6
_RESCUE_BACKLINKS = 300


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


def _load_selected_titles(path: Path, errors: list[str]) -> list[str]:
    """The selected titles of a selection file; errors says why one could not be parsed."""
    if not path.exists():
        return []
    try:
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload.get("selected_titles", [])
    except (OSError, ValueError) as exc:  # pragma: no cover
        errors.append(f"selected_parse_failed:{exc}")
        return []


def _write_selected(path: Path, titles: list[str], seed_title: str, lang: str) -> Path:
    """Writes a selection file for the ladder and returns its path."""
    payload = {
        "seed_title": seed_title,
        "lang": lang,
        "selected_k": len(titles),
        "selected_titles": titles,
    }
    return write_json(path, payload)


def _next_ranked_titles(candidate_scores_path: str | Path, selected_titles: list[str], errors: list[str]) -> list[str]:
    """The 3 best-ranked on-topic candidates of the scores CSV (its rows are in rank order) that are not selected yet."""
    import csv

    promoted = []
    with Path(candidate_scores_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            errors.append("candidate_scores_header_missing")
        for row in reader:
            title = (row.get("title") or "").strip()
            status = (row.get("status") or "").strip().upper()
            if not title:
                continue
            if status == "KEEP" and title not in selected_titles and title not in promoted:
                promoted.append(title)
            if len(promoted) >= 3:
                break
    return promoted


def _join_new_pages(selected_titles: list[str], scores: list[dict], new_titles: set[str]) -> list[str]:
    """The selection after a widening step: the best 6 new pages that are not cut join it, and each further one
    replaces a selected article the scores cut, the last selected first."""
    fresh = [row["title"] for row in scores if row["title"] in new_titles and row["status"] != "CUT"]
    cut = [title for title in selected_titles if any(r["title"] == title and r["status"] == "CUT" for r in scores)]
    replacements = fresh[_RESCUE_SELECTED:][: len(cut)]
    leaving = cut[len(cut) - len(replacements) :]
    return [title for title in selected_titles if title not in leaving] + fresh[:_RESCUE_SELECTED] + replacements


def _backlink_titles(
    candidate_scores_path: str | Path,
    seed_title: str,
    lang: str,
    cache_dir: str | Path,
    max_backlinks: int | None,
    errors: list[str],
) -> list[str]:
    """The pages that link to the seed and are not scored yet, an even sample of them when there are too many."""
    client = WikiClient(DiskCache(cache_dir), api_base=f"https://{lang}.wikipedia.org/w/api.php")
    try:
        resolved_title = client.fetch_links(seed_title)["seed_title"]  # as the seed stage asks, after redirects
        backlinks = client.fetch_backlinks(resolved_title, max_backlinks=max_backlinks)
    except RuntimeError as exc:
        errors.append(f"backlink_fetch_failed:{exc}")
        return []
    known = {row["title"] for row in _read_csv_rows(candidate_scores_path)} | {seed_title, resolved_title}
    fresh = sorted(backlinks - known)
    if len(fresh) > _RESCUE_BACKLINKS:
        fresh = [fresh[i * len(fresh) // _RESCUE_BACKLINKS] for i in range(_RESCUE_BACKLINKS)]
    return fresh


def _widening_titles(
    candidate_scores_path: str | Path,
    selected_titles: list[str],
    seed_title: str,
    lang: str,
    cache_dir: str | Path,
    errors: list[str],
) -> list[str]:
    """The pages the on-topic selected articles link to that the scores do not list yet."""
    scores = _read_csv_rows(candidate_scores_path)
    on_topic_intros = {row["title"] for row in scores if row.get("intro_on_topic") == "True"}
    parents = [title for title in selected_titles if title in on_topic_intros][:_RESCUE_PARENTS]
    client = WikiClient(DiskCache(cache_dir), api_base=f"https://{lang}.wikipedia.org/w/api.php")
    return _linked_titles(client, parents, {row["title"] for row in scores} | {seed_title}, errors)


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

    def _finish(passed: bool, reason: str, gate: VocabGateResult) -> dict:
        """The ladder's result, its diagnostics written with the gate that decided it."""
        diagnostics = {
            "stage": "rescue_ladder",
            "lang": lang,
            "seed_title": seed_title,
            "passed": passed,
            "reason": reason,
            "steps": steps,
            "gate": gate.__dict__,
            "errors": errors,
        }
        final_diagnostics = write_json(diagnostics_path, diagnostics)
        return {"passed": passed, "diagnostics": diagnostics, "diagnostics_path": final_diagnostics}

    selected_titles = _load_selected_titles(Path(selected_path), errors)
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
        return _finish(True, "baseline_passed", gate)

    steps.append({"action": "loosen_min_df", "value": 1})
    gate = _run_gate(min_df_value=1, min_len_value=min_len)
    if gate.passed:
        return _finish(True, "loosen_min_df", gate)

    # Promote the next on-topic candidates by rank (up to 3) into the selected set.
    if Path(candidate_scores_path).exists():
        promoted = _next_ranked_titles(candidate_scores_path, selected_titles, errors)
        steps.append({"action": "promote_next_ranked", "value": len(promoted)})
        if promoted:
            selected_titles = selected_titles + promoted
            selected_path = _write_selected(Path(selected_override_path), selected_titles, seed_title, lang)
    else:
        steps.append({"action": "promote_next_ranked", "value": 0, "note": "candidate_scores_missing"})

    gate = _run_gate(min_df_value=1, min_len_value=min_len)
    if gate.passed:
        return _finish(True, "promote_next_ranked", gate)

    # A stub article links mostly to sibling topics, but the pages that link to it are on topic: the unscored ones are
    # scored as one-hop pages. The best join the selected articles and all of them can be answers. Without the
    # scoring arguments, or with backlinks off, nothing is scored.
    backlink_titles: list[str] = []
    if scoring_args is not None and scoring_args.get("include_backlinks", True):
        backlink_titles = _backlink_titles(
            candidate_scores_path, seed_title, lang, cache_dir, scoring_args.get("max_backlinks"), errors
        )
    steps.append({"action": "widen_from_backlinks", "value": len(backlink_titles)})
    if backlink_titles:
        widened = run_candidate_scoring_stage(
            **scoring_args,
            backlink_titles=backlink_titles,
            diagnostics_path=Path(diagnostics_path).with_name("diagnostics_scores_rescue.json"),
            scores_path=Path(selected_override_path).with_name("candidate_scores_rescue.csv"),
        )
        candidate_scores_path = widened.scores_path
        selected_titles = _join_new_pages(selected_titles, widened.scores, set(backlink_titles))
        selected_path = _write_selected(Path(selected_override_path), selected_titles, seed_title, lang)
        gate = _run_gate(min_df_value=1, min_len_value=min_len)
        if gate.passed:
            return _finish(True, "widen_from_backlinks", gate)

    # Two-hop: the pages the on-topic selected articles link to are scored like the seed's own links, on top of the
    # backlink pages. The best join the selected articles and all of them can be answers.
    if scoring_args is not None:
        extra_titles = _widening_titles(candidate_scores_path, selected_titles, seed_title, lang, cache_dir, errors)
        steps.append({"action": "widen_from_selected", "value": len(extra_titles)})
        if extra_titles:
            widened = run_candidate_scoring_stage(
                **scoring_args,
                backlink_titles=backlink_titles,
                extra_titles=extra_titles,
                diagnostics_path=Path(diagnostics_path).with_name("diagnostics_scores_rescue.json"),
                scores_path=Path(selected_override_path).with_name("candidate_scores_rescue.csv"),
            )
            candidate_scores_path = widened.scores_path
            selected_titles = _join_new_pages(selected_titles, widened.scores, set(extra_titles))
            selected_path = _write_selected(Path(selected_override_path), selected_titles, seed_title, lang)
            gate = _run_gate(min_df_value=1, min_len_value=min_len)
            if gate.passed:
                return _finish(True, "widen_from_selected", gate)

    steps.append({"action": "shorten_min_len", "value": 3})
    gate = _run_gate(min_df_value=1, min_len_value=3)
    if gate.passed:
        return _finish(True, "shorten_min_len", gate)

    # Too few clued answers for the gate, but enough for the smallest grid the layout accepts: the caller goes on.
    thin = gate.term_count >= min_words
    if thin:
        steps.append({"action": "accept_thin_pool", "value": gate.term_count})
    return _finish(False, "accept_thin_pool" if thin else "insufficient_vocabulary", gate)


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
