from __future__ import annotations

import re
from pathlib import Path

from ..term_extractor import (
    answer_cleanliness_score,
    clueability_score,
    _TITLE_DISAMBIGUATOR,
    shape_penalty,
    crosswordability_score,
    title_short_name,
    title_to_answer,
)
from ..clue_builder import answer_pattern, has_verb, lead_subject, split_sentences
from ..lexicon import lexicon_score_for_token
from ..wiki_client import WikiClient

from .common import (
    _safe_float,
    _safe_int,
    _fetch_extract_with_revid,
    _prefetch,
    _uses_seed_word,
    _name_mentions,
    _shared_words,
)
from .clues import _collect_source_backed_clue_candidates


# A page about a person opens with life dates, "born" or "fl." in a parenthesis: "(1840 – 1926)", "(born 1946)",
# "(138–78 BC)", "(fl. 259–269)". A war or a period has dates too, but none that span a life:
# "(149–146 BC)", "(1350 – 1620)".
_PARENTHESIS = re.compile(r"\((?:[^()]|\([^()]*\))*\)")
_YEAR = re.compile(r"(?<!\d)((?:AD|CE) ?)?(\d{1,4})(?!\d)(?: ?(BCE?|[\u03c0\u03bc]\. ?\u03a7))?")


def _life_span(dates: str) -> int:
    """Years from the earliest to the latest year of a parenthesis: 60 for "(138–78 BC)", 86 for "(1840 – 1926)"."""
    years, bc = [], False
    for ad, number, era in reversed(_YEAR.findall(dates)):  # a year with no era takes the next one's
        if ad or era:
            bc = era[:1] in ("B", "\u03c0")
        if ad or era or int(number) > 31:  # a smaller number is a day
            years.append(-int(number) if bc else int(number))
    return max(years) - min(years) if years else 0


def _is_person(text: str, *, lang: str) -> bool:
    subject = lead_subject(text[:600], lang=lang)
    parentheses = _PARENTHESIS.findall(subject)
    if any(re.search(r"\b(?:born|fl)\b", dates) for dates in parentheses):
        return True
    # A war can last a life too, but no person opens on an article: "The Samnite Wars (343–290 BC)".
    return not subject.startswith(("The ", "A ", "An ")) and any(
        15 <= _life_span(dates) <= 110 for dates in parentheses
    )


def _seed_short_names(
    intros: dict[str, str], seed_title: str, seed_text: str, *, lang: str, has_seed_fill_in
) -> dict[str, tuple[str, str, int, bool]]:
    """Pages the seed calls by the last word of their title: page title -> (display, normalized, mentions, person).

    A person by the surname: "Monet" for "Claude Monet" when the seed names it alone at least twice, or when the
    person's own page does after its first sentence ("Kelvin" for "Lord Kelvin"). `mentions` counts the seed's.
    A surname after a lower-case word is none ("Vincent van Gogh", "Cato the Younger"), nor is a word of the seed
    title, and one only the page uses never takes another page's title ("Kelvin", the unit). Of several people
    with one surname, the one whose full name the seed uses more often than all the others' together takes it;
    with no such person ("Williams") nobody does. Only a Greek person is named this way, and only by the title's
    own form: the seed writes other pages in an ending of its own.

    Any other page by a word the seed uses as a proper name, capitalised in mid-sentence and alone at least twice:
    "Chicago" for "Music of Chicago", not "Rhythm" for "I Got Rhythm". The seed's sentence is its only clue, so
    `has_seed_fill_in` must find one. A word that ends two such pages is a class noun ("Empire") and nobody's
    name. A surname, another page's own title and a word of the seed title are not free.
    """
    people: dict[str, list[tuple[str, tuple[str, str]]]] = {}
    others: dict[str, list[tuple[str, tuple[str, str]]]] = {}
    for title, intro in intros.items():
        person = _is_person(intro, lang=lang)
        pair = title_short_name(title, lang=lang, person=person)
        if pair and title != seed_title:
            (people if person else others).setdefault(pair[1], []).append((title, pair))
    titled = {pair[1] for title in intros if (pair := title_to_answer(title, lang=lang))}
    names = {}
    for rivals in people.values():
        phrases = {title: _TITLE_DISAMBIGUATOR.sub("", title).strip() for title, _ in rivals}
        named = {title: len(answer_pattern(phrase).findall(seed_text)) for title, phrase in phrases.items()}
        for title, pair in rivals:
            if (len(rivals) > 1 and 2 * named[title] <= sum(named.values())) or phrases[title].split()[-2].islower():
                continue
            if pair[0].casefold() in seed_title.casefold().split():  # a word of the seed title is not free
                continue
            mentions = _name_mentions(pair[0], seed_text, part_of=(phrases[title],))
            own = " ".join(split_sentences(intros[title])[1:])
            if mentions >= 2 or (pair[1] not in titled and _name_mentions(pair[0], own, part_of=(phrases[title],))):
                names[title] = (*pair, mentions, True)
    if lang != "en":  # a Greek page that is no person keeps its title: the seed writes its inflected form
        return names
    taken = set(people) | titled
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
        except RuntimeError as exc:
            errors.append(f"title_answer_extract_failed:{title}:{exc}")
            continue
        # A redirect ("Big bands") is named after the page it leads to.
        page_title = title if title == seed_title else payload.get("title") or title
        intro = (payload.get("extract") or "").strip()
        pages.append([title, rel_score, intro_on_topic, seed_mentions, page_title, intro])

    def evidence(page: list) -> tuple[bool, float]:
        return page[0] == seed_title, page[1] + 0.1 * min(page[3], 10) + (0.5 if page[2] else 0.0)

    pages.sort(key=evidence, reverse=True)
    seed_payload = _fetch_extract_with_revid(errors, client, seed_title, full=True)

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
        if page[4] in short_names and short_names[page[4]][2] >= 2:
            # The seed's mentions of the answer itself; a surname only the page uses keeps the page's count.
            page[3] = short_names[page[4]][2]
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
