from __future__ import annotations

from pathlib import Path

from ..cache import DiskCache
from ..artifact_models import SelectedCandidatesArtifact
from ..diagnostics import write_json
from ..term_extractor import (
    MergeDiagnostics,
    TermCandidate,
    extract_terms_lead_bold,
    extract_terms_spacy,
    extract_terms_title_tokens,
    GENERIC_GRID_JUNK_EN,
    merge_plural_duplicates,
    merge_terms,
    shape_penalty,
    crosswordability_score,
    term_frequency_across_docs,
    title_to_answer,
)
from ..text_normalize import is_sensitive, tokenize
from ..vocab_gate import evaluate_vocab_gate
from ..wikidata_client import WikidataClient
from ..lexicon import load_lexicon_scores, lexicon_score_for_token
from ..wiki_client import WikiClient

from .common import (
    TermExtractionStageResult,
    VocabGateStageResult,
    _clued_answer_count,
    _resolve_nlp_backend,
    _fetch_extract_with_revid,
    _prefetch,
)
from .clues import _collect_source_backed_clue_candidates
from .titles import _build_title_answer_rows


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


_MAX_ANSWER_POOL = 200


def _selected_titles_for_terms(selected_path: str | Path, errors: list[str]) -> list[str]:
    """Titles of the selected articles; errors says why there are none."""
    if Path(selected_path).exists():
        selected_payload = Path(selected_path).read_text(encoding="utf-8")
        try:
            import json

            selected = json.loads(selected_payload)
            selected_artifact = SelectedCandidatesArtifact.from_dict(selected)
            selected_titles = selected_artifact.selected_titles
        except ValueError as exc:  # pragma: no cover
            errors.append(f"selected_parse_failed:{exc}")
            selected_titles = []
    else:
        selected_titles = []
        errors.append("selected_candidates_missing")
    return selected_titles


def _unwritten_terms_result(
    seed_title: str, lang: str, errors: list[str], diagnostics_path: str | Path, terms_path: str | Path
) -> TermExtractionStageResult:
    """The stage result for a run with no selected articles: no terms, and the diagnostics saying so."""
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


def _collect_page_texts(
    errors: list[str], client: WikiClient, selected_titles: list[str]
) -> tuple[dict[str, list[str]], dict[str, dict], dict[str, str]]:
    """The lead-bold terms, the full-text payloads and the extracts of the selected articles, by title."""
    lead_terms: dict[str, list[str]] = {}
    full_payloads: dict[str, dict] = {}
    extracts: dict[str, str] = {}
    _prefetch(errors, client.prefetch_lead_wikitext, selected_titles)
    for title in selected_titles:
        try:
            intro_extract = client.fetch_page_extract(title, intro_only=True).get("extract", "")
        except RuntimeError as exc:  # pragma: no cover
            intro_extract = ""
            errors.append(f"candidate_extract_failed:{title}:{exc}")

        full_payload = _fetch_extract_with_revid(errors, client, title, full=True)
        if not full_payload["extract"]:  # the plain intro serves term extraction, but has no revision id
            full_payload = {"title": title, "extract": intro_extract, "revid": None}
        full_payloads[title] = full_payload
        extracts[title] = full_payload["extract"]

        try:
            lead_wikitext = client.fetch_lead_wikitext(title).get("wikitext", "")
        except RuntimeError as exc:  # pragma: no cover
            lead_wikitext = ""
            errors.append(f"lead_wikitext_failed:{title}:{exc}")
        lead_terms[title] = client.extract_lead_bold_terms(lead_wikitext)
    return lead_terms, full_payloads, extracts


def _extracted_term_lists(
    extracts: dict[str, str], lead_terms: dict[str, list[str]], *, nlp: object | None, backend_used: str | None, lang: str
) -> list:
    """The term lists of every extractor: spaCy over the extracts, then the lead-bold and title terms of each page."""
    term_lists = []
    if backend_used == "spacy" and nlp is not None:
        for title, extract in extracts.items():
            doc = nlp(extract)
            term_lists.append(extract_terms_spacy(doc, source_title=title, lang=lang))

    for title, bold_terms in lead_terms.items():
        term_lists.append(extract_terms_lead_bold(bold_terms, source_title=title, lang=lang))
        term_lists.append(extract_terms_title_tokens(title, source_title=title, lang=lang))
    return term_lists


def _merge_term_lists(
    term_lists: list, *, lang: str, min_len: int, max_len: int, min_alpha_ratio: float
) -> tuple[list[TermCandidate], MergeDiagnostics]:
    """The term lists merged into one candidate per answer, with the counts of what the merge rejected."""
    from ..term_extractor import get_stopwords

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
    return merged, merge_diagnostics


def _terms_in_enough_pages(
    merged: list[TermCandidate], extracts: dict[str, str], *, lang: str, min_df: int
) -> list[tuple]:
    """The merged terms found in at least min_df of the extracts, each with its document frequency."""
    df = term_frequency_across_docs(list(extracts.values()), lang=lang)
    eligible_terms: list[tuple] = []
    for term in merged:
        freq = df.get(term.normalized_answer, 0)
        if freq < min_df:
            continue
        eligible_terms.append((term, freq))
    return eligible_terms


def _support_sentence_count(
    term: TermCandidate,
    support_sentence_count_by_key: dict[str, int],
    *,
    full_payloads: dict[str, dict],
    seed_title: str,
    lang: str,
    nlp: object | None,
) -> int:
    """How many source-backed clue sentences the term has, counted once per answer."""
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
    return support_sentence_count


def _entity_type_score(
    term: TermCandidate,
    wikidata_client: WikidataClient | None,
    entity_scores: dict[str, float],
    *,
    lang: str,
    errors: list[str],
) -> tuple[float, bool]:
    """The term's entity-type score and whether a lookup just found a preferred type; entity_scores caches per answer."""
    preferred_entity_types = {"Q115949945", "Q107715", "Q47574"}
    entity_score = 0.0
    entity_hit = False
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
                        entity_hit = True
            except RuntimeError as exc:  # pragma: no cover - network path
                errors.append(f"wikidata_lookup_failed:{term.answer}:{exc}")
            entity_scores[cache_key] = entity_score
    return entity_score, entity_hit


def _evidence_tier(
    term: TermCandidate, title_support_score: float, support_sentence_count: int, source_title_count: int
) -> tuple[str, float]:
    """The evidence tier of a supported term and the score that tier is worth."""
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
    return evidence_tier, evidence_tier_score


def _ranked_answer_pool(title_rows: list[dict], filtered_terms: list[dict]) -> list[dict]:
    """The title answers and the other terms with plural duplicates merged, best first, cut to the pool size."""
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
    return filtered_terms


def _write_terms_csv(terms_path: str | Path, filtered_terms: list[dict]) -> None:
    """Writes the answer pool to the terms CSV, one row per answer."""
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

    selected_titles = _selected_titles_for_terms(selected_path, errors)
    if not selected_titles:
        return _unwritten_terms_result(seed_title, lang, errors, diagnostics_path, terms_path)

    nlp, backend_used, backend_errors, spacy_version, spacy_model_version = _resolve_nlp_backend(
        lang=lang,
        nlp_backend=nlp_backend,
    )
    errors.extend(backend_errors)

    lead_terms, full_payloads, extracts = _collect_page_texts(errors, client, selected_titles)

    term_lists = _extracted_term_lists(extracts, lead_terms, nlp=nlp, backend_used=backend_used, lang=lang)

    merged, merge_diagnostics = _merge_term_lists(
        term_lists, lang=lang, min_len=min_len, max_len=max_len, min_alpha_ratio=min_alpha_ratio
    )

    eligible_terms = _terms_in_enough_pages(merged, extracts, lang=lang, min_df=min_df)

    max_df = max((freq for _, freq in eligible_terms), default=1)
    entity_scores: dict[str, float] = {}
    entity_hits = 0
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
        if (
            not term.lead_bold_signal
            or term.normalized_answer in GENERIC_GRID_JUNK_EN
            or is_sensitive(term.answer, lang, seed_title)
        ):
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
        support_sentence_count = _support_sentence_count(
            term,
            support_sentence_count_by_key,
            full_payloads=full_payloads,
            seed_title=seed_title,
            lang=lang,
            nlp=nlp,
        )
        support_sentence_score = min(1.0, support_sentence_count / 3.0)
        lexicon_score = lexicon_score_for_token(term.normalized_answer, lexicon)
        if lexicon_score > 0.0:
            lexicon_hits += 1
        entity_score, entity_hit = _entity_type_score(term, wikidata_client, entity_scores, lang=lang, errors=errors)
        entity_hits += entity_hit

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
        evidence_tier, evidence_tier_score = _evidence_tier(
            term, title_support_score, support_sentence_count, source_title_count
        )
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
    # A selected redirect ("Αυτί") is another spelling of the page whose title already answers ("Αφτί").
    answered = {row["source_titles"].split("|")[0] for row in title_rows}
    spellings = {
        pair[1]
        for title, payload in full_payloads.items()
        if payload["title"] != title
        and payload["title"] in answered
        and (pair := title_to_answer(title, lang=lang, max_len=max_len))
    }
    filtered_terms = _ranked_answer_pool(
        title_rows, [row for row in filtered_terms if row["normalized_answer"] not in spellings]
    )

    _write_terms_csv(terms_path, filtered_terms)

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
