from __future__ import annotations

import re
from collections import Counter
from typing import NamedTuple

from .term_extractor import get_stopwords
from .text_normalize import strip_diacritics, tokenize


def _fold(word: str) -> str:
    """Case and accents carry no meaning when words are compared with a word list."""
    return strip_diacritics(word).lower()


# No split after initials or common abbreviations ("John F. Kennedy", "Charles Mingus Jr.", "300 π.Χ.").
_SENTENCE_END = re.compile(
    r"(?<=[.!?])(?<!\b[^\W\d_]\.)(?<!\bJr\.)(?<!\bSr\.)(?<!\bSt\.)(?<!\bDr\.)(?<!\bMr\.)(?<!\bVol\.)\s+"
)


_PARENTHETICAL = re.compile(r"\s*\([^()]*\)")


def strip_parentheticals(text: str) -> str:
    """Drop (nested) parentheses: dates, translations and the "(; born 1926)" left by removed pronunciations."""
    while (stripped := _PARENTHETICAL.sub("", text)) != text:
        text = stripped
    return text


# A full extract sets each section title on its own line after two blank lines.
_TRAILING_SECTIONS = re.compile(
    r"\n\n\n(?:See also|Notes|References|Citations|Sources|Bibliography|Further reading|External links"
    r"|Works|Publications|Selected works|Discography|Filmography"
    r"|Δείτε [Εε]πίσης|Παραπομπές(?: και παρατηρήσεις)?|Αναφορές|Σημειώσεις|Πηγές|Βιβλιογραφία|Προτεινόμενη βιβλιογραφία"
    r"|Περισσότερες πληροφορίες|Εξωτερικοί σύνδεσμοι|Έργα|Δημοσιεύσεις|Δισκογραφία|Ταινιογραφία)(?:\n|$)"
)


def strip_trailing_sections(text: str) -> str:
    """Cut a full extract at its first reference section: bibliographies and link lists make poor clues."""
    match = _TRAILING_SECTIONS.search(text)
    return text[: match.start()] if match else text


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    text = strip_parentheticals(text.replace("\u00ad", ""))
    return [sentence.strip() for sentence in _SENTENCE_END.split(text) if sentence.strip()]


BLANK = "____"
MAX_CLUE_WORDS = 18


def answer_pattern(answer: str) -> re.Pattern[str]:
    """Whole-word (hyphenated compounds excluded), case-insensitive; spaces match any whitespace."""
    body = r"\s+".join(re.escape(part) for part in answer.split())
    return re.compile(rf"(?<![\w-]){body}(?![\w-])", re.IGNORECASE)


def mask_answer(sentence: str, answer: str) -> str:
    if not sentence or not answer:
        return sentence
    return answer_pattern(answer).sub(BLANK, sentence)


def select_sentence(sentences: list[str], answer: str) -> str | None:
    pattern = answer_pattern(answer)
    for sentence in sentences:
        if pattern.search(sentence):
            return sentence
    return None


def trim_clause(sentence: str, max_words: int = 12) -> tuple[str, bool]:
    """Return (clue, clean): the sentence if it fits MAX_CLUE_WORDS, else a window of at most `max_words` words.

    Clean means a sentence that reads as one, or the start of one cut once at a clause boundary:
    a window that opens mid-sentence never is. The longest clean window wins; when there is none
    the window is centred on the blank and is not clean.
    """
    words = sentence.split()

    def at_boundary(idx: int) -> bool:
        if idx == 0:  # a heading, list entry or formula remnant does not open like a sentence
            return sentence[0].isupper() or sentence[0].isdigit() or sentence[0] in "_\"'(«"
        if idx == len(words):
            return sentence.rstrip("\"')").endswith((".", "!", "?"))
        return words[idx - 1][-1] in ",;:"

    if len(words) <= max(max_words, MAX_CLUE_WORDS):
        return " ".join(words), at_boundary(0) and at_boundary(len(words))
    blank = next((idx for idx, word in enumerate(words) if BLANK in word), None)
    if blank is None:
        return " ".join(words[:max_words]), False
    earliest = max(0, blank - max_words + 1)
    latest = min(blank, len(words) - max_words)
    centered = min(max(blank - max_words // 2, earliest), latest)

    def rank(window: tuple[int, int]) -> tuple[bool, int, bool, int]:
        start, end = window
        clean = start == 0 and at_boundary(0) and at_boundary(end)
        return (not clean, start - end, not at_boundary(start), abs(start - centered))

    start, end = min(
        ((s, e) for s in range(earliest, blank + 1) for e in range(blank + 1, min(len(words), s + max_words) + 1)),
        key=rank,
    )
    clue = " ".join(words[start:end])
    if start > 0:
        clue = f"... {clue}"
    if end < len(words):
        clue = f"{clue} ..."
    return clue, not rank((start, end))[0]


def clue_pass_extract(text: str, answer: str) -> str | None:
    sentence, _ = clue_pass_extract_with_offset(text, answer)
    return sentence


def clue_pass_extract_with_offset(text: str, answer: str) -> tuple[str | None, int | None]:
    pattern = answer_pattern(answer)
    for idx, sentence in enumerate(split_sentences(text)):
        if pattern.search(sentence):
            return sentence, idx
    return None, None


def clue_pass_mask_trim(sentence: str, answer: str, max_words: int = 12) -> tuple[str, bool]:
    clue, clean = trim_clause(mask_answer(sentence, answer), max_words=max_words)
    # An opened quotation that never closes is a fragment.
    return clue, clean and clue.count('"') % 2 == 0 and clue.count("«") == clue.count("»")


def _words(text: str) -> frozenset[str]:
    return frozenset(_fold(word) for word in text.split())


class _Words(NamedTuple):
    """The word lists `definition_clue` works with in one language. Every word is folded (see _fold)."""

    copulas: frozenset[str]
    # After these verbs the predicate is in the accusative, which reads wrong as a clue when it shows
    # (an article from `accusative_openers`, or a masculine head noun that the tagger marks).
    accusative_copulas: frozenset[str]
    accusative_openers: frozenset[str]
    # Turns ", who was widely regarded as ..." into the participle phrase ", regarded as ...", or None.
    credited: re.Pattern[str] | None
    # A predicate that opens like this ("is significantly influenced by ...", "is also known as ...") defines nothing.
    not_a_noun_phrase: re.Pattern[str]
    # Prepositions that start an optional adjunct, not those that belong to the noun phrase.
    adjunct_prepositions: frozenset[str]
    prepositions: frozenset[str]
    demonstratives: frozenset[str]
    possessives: frozenset[str]
    # Pronouns that may point at a modifier of the removed subject ("the volume of a system ... its state").
    referring: frozenset[str]
    relatives: frozenset[str]
    # Relative pronouns that take the verb of their own clause, and those that introduce a person.
    verbal_relatives: frozenset[str]
    person_relatives: frozenset[str]
    # Function words that belong to a relative clause when they stand right before its pronoun ("της οποίας").
    relative_lead: frozenset[str]
    weak_end: frozenset[str]
    # Words that need the preposition after them, and endings that mark such a word.
    participles: frozenset[str]
    participle_endings: tuple[str, ...]
    # A word with one of these endings opens an adjunct clause, like a comma does.
    gerund_endings: tuple[str, ...]
    determiners: frozenset[str]
    finite_verbs: frozenset[str]
    conjunctions: frozenset[str]
    and_word: str
    between: frozenset[str]
    # A comma segment that opens with one of these ("including ...") is an optional tail.
    tails: frozenset[str]
    # How an answer's last word occurs in the subject: its last `stem_trim` letters may differ, then `endings`.
    stem_trim: int
    endings: str


_EN_ADJUNCT = (
    "in at by for from with during after before near around across through within against under over into"
    " between among since"
)
_EN_RELATIVE = "that which who whose where when"
_EN = _Words(
    copulas=_words("is was are were"),
    accusative_copulas=frozenset(),
    accusative_openers=frozenset(),
    # "... and is considered one of ..." and "..., who was widely regarded as ..." go on as a participle: ", considered one of ...".
    credited=re.compile(
        r",? (?:and|who) (?:is|was) (?:(?:also|\w+ly) )?(?=(?:considered|regarded|known|credited|recogni[sz]ed|remembered)\b)"
    ),
    not_a_noun_phrase=re.compile(r"(?:also|\w+ly|\w{3,}ed)\b"),
    adjunct_prepositions=_words(_EN_ADJUNCT),
    prepositions=_words(f"{_EN_ADJUNCT} of to on"),
    demonstratives=_words("this that these those"),
    possessives=_words("its their his her"),
    referring=_words("it its their"),
    relatives=_words(_EN_RELATIVE),
    verbal_relatives=_words("that which who"),
    person_relatives=_words("who whose"),
    relative_lead=frozenset(),
    weak_end=_words(
        f"{_EN_ADJUNCT} {_EN_RELATIVE} a an the and or but of to as than is was are were be been being has have had"
        " not all first also then often usually typically generally commonly thus so hence while because although"
        " such including especially mainly particularly only once this these those"
    ),
    participles=_words("known born made built found given taken written chosen held"),
    participle_endings=("ed", "ing", "ble"),
    gerund_endings=(),
    determiners=_words("the a an this these its their his her some any each every all"),
    finite_verbs=_words("was is were are had has became also"),
    conjunctions=_words("and or"),
    and_word="and",
    between=_words("between among"),
    tails=_words("including such especially particularly notably mainly mostly while whereas because although"),
    stem_trim=0,
    endings=r"(?:e?s)?",
)

_EL_ARTICLE = "ο η το οι τα του της των τον την τους τις τη"
_EL_INDEFINITE = "ένας μια μία ένα έναν ενός μιας"
# στο = σε + το and so on. Folded, the conjunction "ή" (or) cannot be told from the article "η".
_EL_CONTRACTED = "στο στη στην στον στους στις στα"
_EL_ADJUNCT = (
    f"σε από με για προς κατά μετά μεταξύ χωρίς μέχρι έως εντός εκτός κοντά γύρω παρά υπό επί {_EL_CONTRACTED}"
)
_EL_DEMONSTRATIVE = "αυτός αυτή αυτό αυτοί αυτές αυτά εκείνος εκείνη εκείνο"
_EL_RELATIVE = "που οποίος οποία οποίο οποίοι οποίες οποίου οποίας οποίων οποίον οποίους όπου όταν"
_EL = _Words(
    copulas=_words(
        "είναι ήταν αποτελεί αποτελούν αποτελούσε υπήρξε ονομάζεται ονομάζονται καλείται καλούνται"
        " ορίζεται ορίζονται χαρακτηρίζεται χαρακτηρίζονται"
    ),
    accusative_copulas=_words("αποτελεί αποτελούν αποτελούσε"),
    accusative_openers=_words("έναν τον την τους τις"),
    credited=None,
    not_a_noun_phrase=re.compile(r"(?:επισης|ως|οπως|κυριως|συνηθως|γνωστ\w*|ονομαζ\w*)\b"),
    adjunct_prepositions=_words(_EL_ADJUNCT),
    prepositions=_words(f"{_EL_ADJUNCT} ως μέσω λόγω βάσει"),
    demonstratives=_words(_EL_DEMONSTRATIVE),
    possessives=frozenset(),
    referring=frozenset(),
    relatives=_words(_EL_RELATIVE),
    verbal_relatives=_words(f"που {_EL_RELATIVE}"),
    person_relatives=frozenset(),
    relative_lead=_words(f"{_EL_ARTICLE} {_EL_ADJUNCT} μέσω λόγω βάσει"),
    weak_end=_words(
        f"{_EL_ARTICLE} {_EL_INDEFINITE} {_EL_ADJUNCT} {_EL_RELATIVE} {_EL_DEMONSTRATIVE} και ή ούτε ενώ αλλά ως όπως"
        " ότι αν εάν επειδή"
        " αφού καθώς δηλαδή είτε ώστε μα είναι ήταν έχει έχουν είχε θα να δεν επίσης κυρίως συνήθως ιδιαίτερα μόνο"
    ),
    participles=_words("γνωστός γνωστή γνωστό γνωστοί γνωστές γνωστά υπεύθυνος υπεύθυνη υπεύθυνο"),
    participle_endings=("μενος", "μενη", "μενο", "μενοι", "μενες", "μενα"),
    gerund_endings=("ντας",),
    determiners=_words(
        f"{_EL_ARTICLE} {_EL_INDEFINITE} αυτός αυτή αυτό αυτοί αυτές αυτά κάθε κάποιος κάποια κάποιο κάποιοι κάποιες"
        " όλος όλη όλο όλοι όλες όλα"
    ),
    finite_verbs=_words("είναι ήταν έχει έχουν είχε έγινε υπήρξε αποτελεί αποτελούν θεωρείται ονομάζεται"),
    conjunctions=_words("και"),
    and_word="και",
    between=_words("μεταξύ"),
    tails=_words("συμπεριλαμβανομένων ιδίως ειδικά ειδικότερα όπως δηλαδή ενώ επειδή καθώς αφού αν εάν"),
    stem_trim=2,
    endings=r"\w{0,3}",
)
_WORDS = {"en": _EN, "el": _EL}
_BAD_START = {"VERB", "AUX", "ADV", "ADP", "SCONJ", "CCONJ", "PART"}
_BAD_END = (_BAD_START - {"VERB"}) | {"DET"}


def _ends_in_nationality(kept: list[str], nlp) -> bool:
    """True when the cut stops on a nationality that modifies a noun it cut off ("... organization in Brazilian")."""
    last = [token for token in nlp(" ".join(kept)) if not token.is_punct][-1]
    return last.ent_type_ == "NORP" and last.tag_ != "NNPS"  # "the Romans" is a noun


def _masculine_accusative(words: list[str], nlp) -> bool:
    """True when the first noun or adjective is tagged masculine accusative ("ιδιαίτερο κλάδο")."""
    head = next((token for token in nlp(" ".join(words)) if token.pos_ in {"NOUN", "ADJ", "PROPN"}), None)
    return head is not None and head.morph.get("Case") == ["Acc"] and head.morph.get("Gender") == ["Masc"]


def _clean_prefix(words: list[str], limit: int, lex: _Words, nlp=None) -> list[str] | None:
    """Longest prefix of at most `limit` words that stops at a clean boundary, or None."""

    def prefix(count: int, at_comma: bool, safe: bool) -> list[str] | None:
        kept = words[:count]
        commas = [i for i, word in enumerate(kept[:-1]) if word[-1] in ",;:"]
        # An orphan after the last comma ("..., all", "..., a state") is dropped; "..., and X" closes a list.
        if (
            not at_comma
            and commas
            and len(kept) - commas[-1] - 1 <= 2
            and _fold(kept[commas[-1] + 1]) not in lex.conjunctions
        ):
            kept = kept[: commas[-1] + 1]
        if len(kept) < 5 or _fold(kept[-1].strip(",;:")) in lex.weak_end:
            return None
        # An aside opened after a function word ("a folk instrument that, in its form,") never closes.
        if any(_fold(word.rstrip(",;:")) in lex.weak_end for word in kept[:-1] if word[-1] in ",;:"):
            return None
        lowered = [_fold(word.strip(",;:")) for word in kept]
        # A closing "its X" points back at the removed subject ("coin from its introduction").
        if len(lowered) > 2 and lowered[-2] in lex.possessives and lowered[-3] != "of":
            return None
        # "between X and Y" has to be complete.
        if not all(lex.and_word in lowered[pos:] for pos, word in enumerate(lowered) if word in lex.between):
            return None
        # A tail ("including the legal developments") that the cut left unfinished is dropped as a whole.
        opened = [i + 1 for i, word in enumerate(kept[:-1]) if word[-1] in ",;:"]
        if opened and kept[-1][-1] not in ",;:" and lowered[opened[-1]] in lex.tails:
            return None
        # Cutting inside a relative clause leaves it unfinished, unless the cut closes a list or a place name,
        # or is a comma after a clause that already has its verb ("that studies X, Y, and Z, including ...").
        relatives = [pos for pos, word in enumerate(lowered) if pos and word in lex.relatives]
        if relatives and not safe:
            has_verb = all(
                lowered[pos] in lex.verbal_relatives
                and pos + 1 < len(lowered)
                and lowered[pos + 1] not in lex.determiners
                for pos in relatives
            )
            if not (at_comma and has_verb):
                return None
        return None if nlp is not None and _ends_in_nationality(kept, nlp) else kept

    cuts: dict[int, tuple[bool, bool]] = {}
    for i in range(1, limit):
        word = words[i]
        if word[-1] in ",;:":
            closes_list = _fold(words[i - 1]) in lex.conjunctions and words[i - 2][-1] == ","
            place = words[i - 1][-1] == "," and words[i - 1][0].isupper() and word[0].isupper()
            inside_list = (
                words[i - 1][-1] in ",;:"
                or any(_fold(later) in lex.conjunctions for later in words[i + 1 : i + 3])
                or any(later[-1] in ",;:" for later in words[i + 1 : i + 4])
            )
            if closes_list or place or not inside_list:
                cuts[i + 1] = (True, closes_list or place)
        elif _fold(word) in lex.adjunct_prepositions:
            # Never leave a participle or adjective dangling ("formed by", "based in", "responsible for").
            before = _fold(words[i - 1])
            cuts.setdefault(
                i - 1 if before.endswith(lex.participle_endings) or before in lex.participles else i, (False, False)
            )
        elif _fold(word).endswith(lex.gerund_endings):
            cuts[i] = (True, False)
    for count in sorted(cuts, reverse=True):
        if (kept := prefix(count, *cuts[count])) is not None:
            return kept
    # Otherwise drop the first relative clause: "An American scientist [who ...]", "A branch of physics [that ...]".
    first = next((i for i in range(3, limit) if _fold(words[i]) in lex.relatives), None)
    if first is not None and (_fold(words[first]) in lex.person_relatives or first >= 6):
        start = first
        while start > 1 and _fold(words[start - 1]) in lex.relative_lead:
            start -= 1
        if _fold(words[start - 1].strip(",;:")) not in lex.weak_end:
            return None if nlp is not None and _ends_in_nationality(words[:start], nlp) else words[:start]
    return None


def _subject_tail(subject: str, answer: str, lex: _Words) -> str | None:
    """What follows the last word of the answer in the lead sentence subject, None when it does not occur there.

    "" for "Volume", " of a system" for "the volume of a system". The word may carry another ending or number.
    """
    last = _fold(answer.split()[-1])
    stem = last[: len(last) - lex.stem_trim] if len(last) - lex.stem_trim >= 3 else last
    subject = _fold(subject)
    match = re.search(rf"(?<![\w-]){re.escape(stem)}{lex.endings}(?![\w-])", subject)
    return None if match is None else subject[match.end() :]


def _is_noun_phrase(clue: str, nlp) -> bool:
    """A noun phrase neither starts with a verb or adverb ("Fought in 53 BC ...") nor ends in a function word or adverb."""
    tokens = [token for token in nlp(clue) if not token.is_punct]
    return bool(tokens) and tokens[0].pos_ not in _BAD_START and tokens[-1].pos_ not in _BAD_END


def definition_clue(
    lead_sentence: str, *, max_words: int = 16, answer: str | None = None, nlp=None, lang: str = "en"
) -> str | None:
    """Turn "Bebop is a style of jazz ..." into "A style of jazz ...".

    The whole predicate is used when it fits `max_words`; a longer one is cut only at a clean
    boundary, otherwise there is no definition. With `answer`, the sentence must be about it:
    its last word has to occur before the copula. With `nlp`, the result must also be a noun phrase.
    """
    lex = _WORDS[lang]
    lead = strip_parentheticals(lead_sentence).split()
    copula = next((i for i, word in enumerate(lead) if i and _fold(word) in lex.copulas), None)
    if copula is None:
        return None
    tail = None
    if answer is not None:
        tail = _subject_tail(" ".join(lead[:copula]), answer, lex)
        if tail is None:
            return None
    predicate = " ".join(lead[copula + 1 :]).rstrip(".")
    words = (lex.credited.sub(", ", predicate) if lex.credited else predicate).split()
    # A second predicate ("a mathematician and was one of ...") does not belong to the definition.
    for i in range(3, len(words) - 1):
        if _fold(words[i]) == lex.and_word and _fold(words[i + 1]) in lex.finite_verbs:
            words = words[:i]
            break
    first = _fold(words[0]) if words else ""
    if len(words) < 3 or lex.not_a_noun_phrase.match(first) or first in lex.demonstratives | lex.possessives:
        return None
    if _fold(lead[copula]) in lex.accusative_copulas and (
        first in lex.accusative_openers or (nlp is not None and _masculine_accusative(words[:6], nlp))
    ):
        return None
    if len(words) > max_words:
        words = _clean_prefix(words, max_words, lex, nlp)
        if words is None:
            return None
    lowered = [_fold(word.strip(",;:")) for word in words]
    # "... of that system": the antecedent is outside the clue.
    if any(a in lex.prepositions and b in lex.demonstratives for a, b in zip(lowered, lowered[1:])):
        return None
    # "the volume of a system is ... its state": the possessor was part of the subject, which the clue drops.
    if tail and tail.split()[0] == "of" and lex.referring & set(lowered):
        return None
    clue = " ".join(words).rstrip(" ,;:\u2013\u2014-")
    clue = clue[0].upper() + clue[1:]
    return None if nlp is not None and not _is_noun_phrase(clue, nlp) else clue


# A word is rare when at most this share of the pool's lead sentences (and never fewer than two) contain it.
_RARE_LEAD_SHARE = 1 / 30
_WORD = re.compile(r"\w+")


def lead_word_counts(leads: list[str]) -> Counter[str]:
    """For each word, the number of lead sentences it occurs in."""
    return Counter(word for lead in leads for word in set(_WORD.findall(lead.lower())))


def is_generic_definition(clue: str, counts: Counter[str], pool_size: int, nlp=None) -> bool:
    """True when every word of the clue is common among the pool's lead sentences.

    "A German physicist" or "A genre of dance music" fits dozens of leads, "The capital of Lebanon" fits one.
    With `nlp`, nationalities (named entities, or capitalised adjectives) do not count as rare: "Brazilian" is
    rare, but names no one.
    """
    nationalities = set()
    if nlp is not None:
        nationalities = {
            w
            for token in nlp(clue)
            if token.ent_type_ == "NORP" or (token.i and token.pos_ == "ADJ" and token.text[0].isupper())
            for w in _WORD.findall(token.text.lower())
        }
    limit = max(2, pool_size * _RARE_LEAD_SHARE)
    return all(counts[word] > limit for word in set(_WORD.findall(clue.lower())) - nationalities)


def has_verb(clue: str, nlp) -> bool:
    """False for titles, citations and list entries, which make poor fill-ins."""
    return any(token.pos_ in {"VERB", "AUX"} for token in nlp(clue))


def lemmatize_text(text: str, nlp) -> set[str]:
    lemmas: set[str] = set()
    if not text or nlp is None:
        return lemmas
    doc = nlp(text)
    for token in doc:
        if token.is_alpha:
            lemma = token.lemma_.lower()
            if lemma:
                lemmas.add(lemma)
    return lemmas


def is_answer_subject(sentence: str, answer: str, nlp) -> bool:
    if not sentence or not answer or nlp is None:
        return False
    answer_tokens = {token.lower() for token in answer.split()}
    answer_lemmas = lemmatize_text(answer, nlp)
    try:
        doc = nlp(sentence)
    except Exception:
        return False
    for token in doc:
        if token.dep_ in {"nsubj", "nsubjpass", "csubj", "csubjpass"}:
            token_text = token.text.lower()
            token_lemma = token.lemma_.lower() if token.lemma_ else token_text
            if token_text in answer_tokens or token_lemma in answer_lemmas:
                return True
    return False


def _content_lemmas(text: str, nlp, lang: str) -> set[str]:
    if nlp is None:
        stopwords = get_stopwords(lang)
        tokens = [(word, word.upper() in stopwords) for word in tokenize(text, lang=lang)]
    else:
        tokens = [(strip_diacritics(token.lemma_.lower()), token.is_stop) for token in nlp(text) if token.is_alpha]
    # Greek lemmas differ in gender and ending (φυσική / φυσικός): compare their stems.
    trim = _WORDS[lang].stem_trim
    return {
        lemma[: len(lemma) - trim] if len(lemma) - trim >= 5 else lemma
        for lemma, is_stop in tokens
        if not is_stop and len(lemma) >= 3
    }


def shares_root(a: str, b: str) -> bool:
    prefix = 0
    for ch_a, ch_b in zip(a, b):
        if ch_a != ch_b:
            break
        prefix += 1
    # One word is the start of the other (music/musicians) or both share a long stem (entropy/entropic).
    return prefix >= 6 or (prefix >= 4 and prefix == min(len(a), len(b)))


def has_morphological_leakage(clue: str, answer: str, nlp, *, lang: str = "en") -> bool:
    answer_lemmas = _content_lemmas(answer, nlp, lang)
    if not answer_lemmas:
        return False
    clue_lemmas = _content_lemmas(clue, nlp, lang)
    return any(shares_root(a, c) for a in answer_lemmas for c in clue_lemmas)


# A blank that is one part of a longer proper name ("the ____ Florida Institute", "the Rome-Berlin ____") clues nothing.
_CAPITAL = "A-ZΑ-ΫΆΈ-ΊΌΎΏ"  # Latin and Greek, tonos included
_PART_OF_A_NAME = re.compile(
    BLANK + rf"(?= +[{_CAPITAL}])|[{_CAPITAL}][\w']*[ \-\N{{EN DASH}}\N{{EM DASH}}](?={BLANK})"
)


def _blank_is_part_of_a_name(sentence: str) -> bool:
    """True when a blank is followed by a capitalised word, or preceded by one that does not open the sentence."""
    return any(m.start() > 0 or m.group().startswith(BLANK) for m in _PART_OF_A_NAME.finditer(sentence.strip()))


# A fill-in that opens on a word whose referent is in the previous sentence ("It is the cause of the ____.")
# stands on nothing. Folded: no case, no accents.
_BACK_REFERENCES = {
    "it", "its", "this", "that", "these", "those", "such", "he", "she", "they", "his", "her", "their", "another",
    "and", "but", "or", "so", "yet", "nor", "now", "then", "thus", "hence", "therefore", "however", "also",
    "moreover", "furthermore", "meanwhile", "instead", "consequently", "similarly", "likewise",
    "αυτος", "αυτη", "αυτο", "αυτοι", "αυτες", "αυτα", "αυτου", "αυτης", "αυτων", "εκεινος", "εκεινη", "εκεινο",
    "και", "αλλα", "ομως", "ωστοσο", "ετσι", "επισης", "επομενως", "τοτε", "επιπλεον", "ουτε",
}


def clue_pass_validate(
    clue: str,
    answer: str,
    *,
    min_words: int = 6,
    nlp=None,
    original_sentence: str | None = None,
    require_blank: bool = False,
    lang: str = "en",
) -> bool:
    if not clue:
        return False
    if len([word for word in clue.split() if word != "..."]) < min_words:
        return False
    if require_blank and BLANK not in clue:
        return False
    if "{" in clue or "\\" in clue:  # LaTeX left in the extract ("{\displaystyle F_{i}}")
        return False
    if answer_pattern(answer).search(clue):
        return False
    if has_morphological_leakage(clue, answer, nlp, lang=lang):
        return False
    if require_blank and _fold(clue.split()[0].strip("\"'(«“")) in _BACK_REFERENCES:
        return False
    if original_sentence and is_answer_subject(original_sentence, answer, nlp):
        return False
    if require_blank and original_sentence and _blank_is_part_of_a_name(mask_answer(original_sentence, answer)):
        return False
    return True


def enforce_diversity(clues: list[dict], max_per_bucket: int = 2) -> tuple[list[dict], list[str]]:
    """Return (kept_clues, rejected_answers) after diversity filtering."""
    bucket_counts: dict[str, int] = {}
    output = []
    rejected_answers: list[str] = []
    for clue in clues:
        tokens = clue["clue"].split()
        bucket = " ".join(tokens[:3]).lower()
        bucket_counts.setdefault(bucket, 0)
        if bucket_counts[bucket] >= max_per_bucket:
            rejected_answers.append(clue.get("answer", ""))
            continue
        bucket_counts[bucket] += 1
        output.append(clue)
    return output, rejected_answers
