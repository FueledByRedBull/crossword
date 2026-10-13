from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from ..cache import DiskCache
from ..artifact_models import SelectedCandidatesArtifact
from ..diagnostics import write_json
from ..term_extractor import _TITLE_DISAMBIGUATOR, get_stopwords, title_short_name
from ..clue_builder import (
    _WORD,
    MAX_CLUE_WORDS,
    _fold,
    answer_pattern,
    clue_pass_mask_trim,
    clue_pass_validate,
    definition_clue,
    enforce_diversity,
    follow_up_definition,
    has_verb,
    is_generic_definition,
    is_thin_definition,
    lead_word_counts,
    name_the_person,
    split_sentences,
    who_definition,
)
from ..provenance import build_oldid_url, validate_clue_provenance
from ..text_normalize import is_sensitive
from ..wiki_client import WikiClient

from .common import ClueStageResult, _safe_float, _safe_int, _fetch_extract_with_revid, _prefetch, _answer_key


def _build_clue_row(
    *,
    answer: str,
    normalized_answer: str,
    answer_key_value: str,
    clue_text: str,
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
        "clue_class": "source_backed",
        "clue_style": clue_style,
        "source_method": source_method,
        "source_page": source_page,
        "revid": revid,
        "sentence_offset": sentence_offset,
        "oldid_url": build_oldid_url(source_page, revid, lang=lang),
    }


def _intro_weights(payload: dict, answer: str, lead_counts: Counter[str], pool_size: int, lang: str) -> dict[str, float]:
    """The rarity weight of each content word in the intro of the answer's own page (a full extract's intro ends at its first section title)."""
    intro = " ".join(split_sentences(payload.get("extract", "").split("\n\n\n")[0]))
    stopwords = get_stopwords(lang)
    return {
        word: math.log((pool_size + 1) / (lead_counts[word] + 1))
        for word in set(_WORD.findall(intro.lower())) - set(_WORD.findall(answer.lower()))
        if _fold(word).upper() not in stopwords
    }


def _aboutness(clue_text: str, intro_weights: dict[str, float]) -> float:
    """How much of the intro the clue repeats: the summed rarity of its distinct words that the intro has too."""
    return round(sum(intro_weights.get(word, 0.0) for word in set(_WORD.findall(clue_text.lower()))), 6)


def _clue_candidate_rank(row: dict, seed_title: str, intro_weights: dict[str, float]) -> tuple[bool, bool, float, float, int, int, str]:
    clue_text = str(row.get("clue") or "")
    sentence_offset = _safe_int(row.get("sentence_offset"), default=9999)
    masked_length_score = -abs(len(clue_text.split()) - 9)
    return (
        not clue_text.startswith("...") and not clue_text.endswith("..."),
        row.get("source_page") == seed_title,
        _aboutness(clue_text, intro_weights),
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
    person: bool = False,
    intro_weights: dict[str, float] | None = None,
) -> list[dict]:
    """Fill-in candidates, complete sentences first, then the seed article's before other pages'.

    Among those, the sentence that repeats more of the answer's own intro (`intro_weights`) comes first: "owe their useful
    properties to conservation of ____" says more about "Angular momentum" than "a consequence of the ____ of the orbital".

    A fragment is a window that is not a whole sentence and not a single clause-boundary cut; the
    term stage still counts them as support, a clue never uses one. With `full_name` the answer is a
    surname and the blank may stand for the whole name: "hired ____ as" from "hired Wayne Shorter as".
    The answer of a `person` is a name: a sentence with it in lower case ("a shorter set") is no candidate.
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
            if not pattern.search(candidate_sentence) or (
                person and not all(match.group()[0].isupper() for match in pattern.finditer(candidate_sentence))
            ):
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
                    source_method=term_source_method,
                    source_page=source_title,
                    revid=revid,
                    sentence_offset=sentence_offset,
                    lang=lang,
                    clue_style="fill_in_blank",
                )
            )

    candidates.sort(key=lambda row: _clue_candidate_rank(row, seed_title, intro_weights or {}), reverse=True)
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
    person: bool = False,
) -> list[dict]:
    """Clue an article-title answer with its own article's lead sentence, subject removed.

    The second, longer cut is there for generic definitions ("An American physicist"): it may
    reach the relative clause that says who the person was. A definition that stays generic is dropped; for a
    `person` the generic one is joined with what the second or third sentence says they did ("An English
    physicist who studied the nature of heat"), and that is as good as a first-sentence definition.
    When none is left, the second sentence is tried ("He was one of the most influential figures in jazz piano."):
    a `last_resort` row, used after the fill-ins. So is the person's thin first-sentence definition (too few informative
    words) with the given names of the page and a blank for the surname ("An American jazz drummer (Art ____)"),
    when nothing else clues the answer.
    """
    sentences = split_sentences(payload.get("extract", ""))
    rows: list[dict] = []
    named: str | None = None

    def definition_rows(texts: list[tuple[str, int]], last_resort: bool) -> list[dict]:
        return [
            {
                **_build_clue_row(
                    answer=answer,
                    normalized_answer=normalized_answer,
                    answer_key_value=answer_key_value,
                    clue_text=clue_text,
                    source_method=term_source_method,
                    source_page=payload.get("title", title) or title,
                    revid=payload.get("revid"),
                    sentence_offset=source,
                    lang=lang,
                    clue_style="definition",
                ),
                "last_resort": last_resort,
            }
            for clue_text, source in texts
        ]

    # The second sentence is the last resort of an answer whose first one gave no definition.
    for offset, cut in enumerate((definition_clue, follow_up_definition)):
        if rows or offset >= len(sentences):
            break
        texts: list[tuple[str, int]] = []  # (clue, the sentence it comes from)
        generic: dict[int, str] = {}  # the valid but generic cuts, by their word limit
        for max_words in (16, MAX_CLUE_WORDS):
            clue_text = cut(sentences[offset], answer=answer, max_words=max_words, nlp=nlp, lang=lang)
            if (
                clue_text
                and clue_text not in [text for text, _ in texts]
                and clue_pass_validate(clue_text, answer, min_words=3, nlp=nlp, lang=lang)
            ):
                if is_generic_definition(clue_text, lead_counts, pool_size, nlp, seed_title=seed_title):
                    generic[max_words] = clue_text
                else:
                    texts.append((clue_text, offset))
        if person and offset == 0 and generic and not texts:
            for later, sentence in enumerate(sentences[1:3], 1):
                for max_words, head in generic.items():
                    clue_text = who_definition(head, sentence, answer, max_words=max_words, nlp=nlp, lang=lang)
                    if (
                        clue_text
                        and clue_pass_validate(clue_text, answer, min_words=3, nlp=nlp, lang=lang)
                        and not is_generic_definition(clue_text, lead_counts, pool_size, nlp, seed_title=seed_title)
                    ):
                        texts = [(clue_text, later)]
                        break
                if texts:
                    break
            if not texts:
                named = next(
                    (
                        clue
                        for head in sorted(generic.values(), key=len, reverse=True)
                        if is_thin_definition(head, lead_counts, pool_size, nlp, seed_title=seed_title)
                        and (clue := name_the_person(head, title, answer))
                        and clue_pass_validate(clue, answer, min_words=3, nlp=nlp, lang=lang)
                    ),
                    None,
                )
        rows = definition_rows(texts, cut is follow_up_definition)
    if named and not rows:
        rows = definition_rows([(named, 0)], True)
    return rows


# A Greek page opens with a line that sends the reader to another article for the other meaning:
# "Για το σύστημα χρονολόγησης δείτε: Ημερολόγιο (σύστημα χρονολόγησης)".
_SEE_OTHER_ARTICLE = re.compile(r"Για [^\n]* δείτε")


def _unwritten_clues_result(
    seed_title: str, lang: str, errors: list[str], diagnostics_path: str | Path, clues_path: str | Path
) -> ClueStageResult:
    """The stage result for a run without a terms file: no clues, and the diagnostics saying so."""
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


def _load_clue_nlp(lang: str, errors: list[str]) -> tuple[object | None, str | None, str | None]:
    """The spaCy pipeline for the clues with its version and its model's version; each is None when spaCy is missing."""
    nlp = None
    clue_spacy_version = None
    clue_spacy_model_version = None
    try:
        import spacy

        model_name = "en_core_web_sm" if lang == "en" else "el_core_news_sm"
        nlp = spacy.load(model_name)
        clue_spacy_version = getattr(spacy, "__version__", None)
        clue_spacy_model_version = nlp.meta.get("version")
    except (ImportError, OSError) as exc:  # pragma: no cover
        errors.append(f"clue_spacy_unavailable:{exc}")
    return nlp, clue_spacy_version, clue_spacy_model_version


def _read_term_rows(terms_path: str | Path) -> list[dict[str, str]]:
    """The answers of the terms CSV with the columns the clue stage reads."""
    import csv

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
    return terms


def _selected_titles_for_clues(selected_path: str | Path, errors: list[str]) -> list[str]:
    """Titles of the selected articles; errors says why they could not be read."""
    import json

    selected_titles: list[str] = []
    try:
        selected = json.loads(Path(selected_path).read_text(encoding="utf-8"))
        selected_titles = SelectedCandidatesArtifact.from_dict(selected).selected_titles
    except (OSError, ValueError) as exc:
        errors.append(f"selected_unavailable:{exc}")
    return selected_titles


def _lead_sentences(terms: list[dict[str, str]], load_source_payload: Callable[[str], dict]) -> list[str]:
    """The first sentence of the page behind each answer that is an article title."""
    return [
        sentences[0]
        for term in terms
        if term["source_method"] == "article_title" and term["source_titles"]
        if (sentences := split_sentences(load_source_payload(term["source_titles"].split("|")[0]).get("extract", "")))
    ]


def _write_clues_csv(clues_path: str | Path, clues: list[dict]) -> None:
    """Writes the chosen clues to the clues CSV, one row per answer."""
    import csv

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


def _choose_clue(
    candidates: list[dict],
    used_texts: set[str],
    used_sentences: set[tuple[str, int | None]],
    fill_clues: list[dict],
    *,
    nlp: object | None,
    diversity_cap: int,
) -> tuple[dict | None, int]:
    """The first usable candidate and how many the diversity cap turned down; the choice is recorded in the sets and fill_clues."""
    diversity_rejected_count = 0
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
        return row, diversity_rejected_count
    return None, diversity_rejected_count


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
        return _unwritten_clues_result(seed_title, lang, errors, diagnostics_path, clues_path)

    cache = DiskCache(cache_dir)
    api_base = f"https://{lang}.wikipedia.org/w/api.php"
    client = WikiClient(cache, api_base=api_base)
    nlp, clue_spacy_version, clue_spacy_model_version = _load_clue_nlp(lang, errors)

    terms = _read_term_rows(terms_path)

    # The seed and the selected articles are cited from their full text; every other page from its intro.
    selected_titles = _selected_titles_for_clues(selected_path, errors)
    full_titles = {seed_title, *selected_titles}
    pool_titles = {title for term in terms for title in term["source_titles"].split("|") if title}
    _prefetch(errors, client.prefetch_intro_extracts, sorted(pool_titles - full_titles), with_revid=True)

    source_cache: dict[str, dict] = {}

    def load_source_payload(title: str) -> dict:
        if title not in source_cache:
            source_cache[title] = _fetch_extract_with_revid(errors, client, title, full=title in full_titles)
        return source_cache[title]

    leads = _lead_sentences(terms, load_source_payload)
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
        person = term["page_is_person"] == "True"
        last_word = bool(page) and (title_short_name(page, lang=lang, person=person) or ("",))[0] == answer
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
        # The intro of an off-topic page is about another sense of the word: nothing ranks by it.
        intro_weights = (
            _intro_weights(source_cache[page], answer, lead_counts, len(leads), lang) if page and not other_sense else {}
        )
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
            person=person,
            intro_weights=intro_weights,
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
                    person=person,
                )
            # A follow-up (the second sentence) or a named definition is for an answer nothing else clues: it comes after the fill-ins.
            candidates = (
                [row for row in definitions if not row["last_resort"]]
                + candidates
                + [row for row in definitions if row["last_resort"]]
            )

        candidates = [row for row in candidates if not is_sensitive(row["clue"], lang, seed_title)]
        chosen, rejected_for_diversity = _choose_clue(
            candidates, used_texts, used_sentences, fill_clues, nlp=nlp, diversity_cap=diversity_cap
        )
        diversity_rejected_count += rejected_for_diversity
        if chosen is not None:
            clues.append(chosen)
            clued_keys.add(key)

    _write_clues_csv(clues_path, clues)

    provenance_missing = validate_clue_provenance(clues)
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
