from __future__ import annotations

import re
import unicodedata
from bisect import bisect_right
from collections import Counter
from typing import NamedTuple

from .term_extractor import get_stopwords
from .text_normalize import strip_diacritics, tokenize


def _fold(word: str) -> str:
    """Case and accents carry no meaning when words are compared with a word list."""
    return strip_diacritics(word).lower()


# No split after initials or common abbreviations ("John F. Kennedy", "Charles Mingus Jr.", "300 π.Χ.").
_SENTENCE_END = re.compile(
    r"(?<=[.!?])(?<!\b[^\W\d_]\.)(?<!\bJr\.)(?<!\bSr\.)(?<!\bSt\.)(?<!\bDr\.)(?<!\bMr\.)(?<!\bVol\.)(?<!\bvs\.)\s+"
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
    # The tagger finds a second predicate ("... and is considered ...") too; where this is set it replaces `finite_verbs`.
    tagged_second_predicate: bool
    conjunctions: frozenset[str]
    and_word: str
    between: frozenset[str]
    # A comma segment that opens with one of these ("including ...") is an optional tail.
    tails: frozenset[str]
    # Fill-in checks. Pronouns that need an antecedent in the clue, determiners that point at an earlier sentence
    # ("this pump") with the tags of the word after them that make them determiners, whether they may also follow
    # their noun ("η μέθοδος αυτή"), and phrases that point outside (matched against the folded clue).
    personal: frozenset[str]
    # The subject is left out when it was named before: a clause that opens on a finite verb with no subject of its
    # own, or a possessive clitic ("της") with no noun before it, leans on the previous sentence. Tagger only.
    pro_drop: bool
    pointing: frozenset[str]
    pointing_tags: frozenset[str]
    pointing_follows_noun: bool
    # A determiner and its article are both in the genitive or neither is: "αυτήν του φωτός" is "that of light".
    genitive_endings: tuple[str, ...]
    outside: re.Pattern[str]
    # Nouns for a word: "the term was confined to ..." leans on the sentence that named the term.
    labels: frozenset[str]
    # Time nouns that want what follows them ("in the years from the Spanish-American War to ...").
    time_heads: frozenset[str]
    articles: frozenset[str]
    # The words that join the last item of a list, as written: folded, the Greek "ή" (or) is the article "η".
    coordinators: frozenset[str]
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
    not_a_noun_phrase=re.compile(r"(?:also|best|known|born|\w+ly|\w{3,}ed)\b"),
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
    tagged_second_predicate=False,
    conjunctions=_words("and or"),
    and_word="and",
    between=_words("between among"),
    tails=_words("including such especially particularly notably mainly mostly while whereas because although"),
    personal=_words("he she they it him them"),
    pro_drop=False,
    pointing=_words("this these those"),
    pointing_tags=frozenset({"NOUN", "PROPN"}),
    pointing_follows_noun=False,
    genitive_endings=(),
    outside=re.compile(
        r"\b(?:the (?:former|latter|following)|(?:doing|did|does) so|likewise|aforementioned|above-mentioned|see (?:above|below)"
        r"|as (?:noted|mentioned|discussed|described|stated|explained|shown|seen) (?:above|below|earlier|previously|before))\b"
    ),
    labels=_words("term word name phrase"),
    time_heads=_words("years decades period time era days"),
    articles=frozenset(),
    coordinators=frozenset({"and", "or"}),
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
_EL_DEMONSTRATIVE = (
    "αυτός αυτή αυτό αυτοί αυτές αυτά αυτού αυτής αυτών αυτόν αυτήν αυτούς"
    " εκείνος εκείνη εκείνο εκείνοι εκείνες εκείνα εκείνου εκείνης εκείνων"
)
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
    tagged_second_predicate=True,
    conjunctions=_words("και"),
    and_word="και",
    between=_words("μεταξύ"),
    tails=_words("συμπεριλαμβανομένων ιδίως ειδικά ειδικότερα όπως δηλαδή ενώ επειδή καθώς αφού αν εάν"),
    personal=frozenset(),  # the subject pronoun is rarely written
    pro_drop=True,
    pointing=_words(_EL_DEMONSTRATIVE),
    pointing_tags=frozenset({"DET"}),  # "αυτή η μέθοδος": the article comes first
    pointing_follows_noun=True,
    genitive_endings=("ου", "ης", "ων"),
    outside=re.compile(
        r"\b(?:ακολουθ(?:ος|η|ο|οι|ες|α|ου|ης|ων|ους|ον)|παρακατω|ανωτερω|προαναφερθ\w+|οπως (?:αναφερθηκε|ειδαμε)"
        r"|(?:ο|η|το) πρωτ\w+\b.*\b(?:ο|η|το) δευτερ\w+)\b"
    ),
    labels=frozenset(),
    time_heads=frozenset(),
    articles=_words(_EL_ARTICLE),
    coordinators=frozenset({"και", "κι", "ή"}),
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
    """True when one of the first three nouns or adjectives is tagged masculine accusative ("ιδιαίτερο κλάδο").

    The tagger often gets the first word wrong ("ιδιαίτερο" came out as a neuter nominative), but not all three.
    """
    heads = [token for token in nlp(" ".join(words)) if token.pos_ in {"NOUN", "ADJ", "PROPN"}][:3]
    return any(head.morph.get("Case") == ["Acc"] and head.morph.get("Gender") == ["Masc"] for head in heads)


def _opens_list_item(following: list[str], lex: _Words) -> bool:
    """True when the words after a comma open another item of a list: an article, or "and" and an article.

    Not an article that begins a relative clause ("..., η οποία"). The article test is lexical on purpose: the Greek
    tagger confuses the cases of a noun and the one after it.
    """
    folded = [_fold(word) for word in following] + ["", ""]
    if folded[0] in lex.conjunctions:
        folded = folded[1:]
    return folded[0] in lex.articles and folded[1] not in lex.relatives


def _opens_tail(words: list[str], i: int, lex: _Words) -> bool:
    """True when the comma at `words[i]` is followed by "and" and a preposition, and goes on to something else.

    "... of Rhineland-Palatinate, and with around 223,000 inhabitants" does. A comma after another one closes a list
    ("from the past, through the present, and into the future"), a preposition used before continues one ("in his
    native country, and in his later years").
    """
    following = [_fold(word) for word in words[i + 1 : i + 3]]
    return (
        len(following) == 2
        and following[0] == lex.and_word
        and following[1] in lex.adjunct_prepositions
        and not any(word[-1] in ",;:" or _fold(word) == following[1] for word in words[:i])
    )


def _reaches_its_verb(kept: list[str], relative: int, nlp) -> bool:
    """True when the tagger finds a verb after the relative word at `kept[relative]`.

    "Όρος της Φυσικής, που σε πολύ απλή απόδοση," has none: the cut fell inside the clause.
    """
    start = len(" ".join(kept[: relative + 1]))
    return any(token.pos_ in {"VERB", "AUX"} and token.idx >= start for token in nlp(" ".join(kept)))


# Prepositions that take a time ("... Mark Antony during the final years"), where "in", "with" or "at" may belong to a
# noun or to the verb ("played a key role in ...", "merged X with ..."). The parser's "prep" label is English.
_TIME_ADJUNCTS = frozenset({"during", "after", "before", "since", "until", "throughout"})


def _ends_its_clause(kept: list[str], relatives: list[int], doc, nlp) -> bool:
    """True when each relative clause of `kept` has its verb and the cut falls after a noun, before a time adjunct of that verb.

    "A general who formed the Second Triumvirate alongside Octavian and Mark Antony [during ...]" is whole. `doc` is the
    tagged, parsed predicate.
    """
    offset = len(" ".join(kept)) + 1
    cut = next((token for token in doc if token.idx == offset), None)
    last = [token for token in doc if token.idx < offset and not token.is_punct]
    return (
        cut is not None
        and cut.dep_ == "prep"
        and cut.lower_ in _TIME_ADJUNCTS
        and cut.head.pos_ in {"VERB", "AUX"}
        and bool(last)
        and last[-1].pos_ in {"NOUN", "PROPN"}
        and all(_reaches_its_verb(kept, pos, nlp) for pos in relatives)
    )


def _clean_prefix(words: list[str], limit: int, lex: _Words, nlp=None) -> list[str] | None:
    """Longest prefix of at most `limit` words that stops at a clean boundary, or None."""

    doc = nlp(" ".join(words)) if nlp is not None else None

    def prefix(count: int, at_comma: bool, safe: bool) -> list[str] | None:
        kept = words[:count]
        commas = [i for i, word in enumerate(kept[:-1]) if word[-1] in ",;:"]
        # An orphan after the last comma ("..., all", "..., a state") is dropped; "..., and X" closes a list, unless
        # it is the only comma and the cut took the complement of X ("a painter, and expert [in the rendering of ...]").
        if (
            not at_comma
            and commas
            and len(kept) - commas[-1] - 1 <= 2
            and (
                _fold(kept[commas[-1] + 1]) not in lex.conjunctions
                or (len(commas) == 1 and len(kept) - commas[-1] == 3 and _fold(words[count]) in lex.adjunct_prepositions)
            )
        ):
            kept = kept[: commas[-1] + 1]
        if len(kept) < 5 or _fold(kept[-1].strip(",;:")) in lex.weak_end:
            return None
        # A bare "the years" or "the period" waits for what follows it: "... in New Orleans in the years [from the ...]".
        end = [_fold(word) for word in kept[-2:]]
        if (
            end[-1] in lex.time_heads
            and end[0] in lex.determiners
            and len(kept) < len(words)
            and _fold(words[len(kept)]) in lex.prepositions | lex.relatives
        ):
            return None
        # A name is no place to cut ("member of the Centre [for Quantum Information]").
        if doc is not None and any(ent.start_char < len(" ".join(kept)) < ent.end_char for ent in doc.ents):
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
        # ", and with around 223,000 inhabitants" is no part of the definition, even when a comma closes it.
        if opened and _opens_tail(kept, opened[0] - 1, lex):
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
            if not (has_verb and (at_comma or doc is not None and _ends_its_clause(kept, relatives, doc, nlp))):
                return None
        # A relative clause that opens on an aside ("που σε πολύ απλή απόδοση, εκφράζει ...") has its verb after it.
        # Elsewhere the word after the relative is taken for the verb: the tagger misses too many ("που μελετά").
        if nlp is not None and any(
            pos + 1 < len(lowered) and lowered[pos + 1] in lex.prepositions and not _reaches_its_verb(kept, pos, nlp)
            for pos in relatives
        ):
            return None
        return None if nlp is not None and _ends_in_nationality(kept, nlp) else kept

    cuts: dict[int, tuple[bool, bool]] = {}
    for i in range(1, limit):
        word = words[i]
        if word[-1] in ",;:":
            if _fold(word.strip(",;:")) in lex.conjunctions:  # "... jazz piano [and, according to one source, ...]"
                cuts.setdefault(i, (True, False))
            closes_list = _fold(words[i - 1]) in lex.conjunctions and words[i - 2][-1] == ","
            place = words[i - 1][-1] == "," and words[i - 1][0].isupper() and word[0].isupper()
            inside_list = (
                words[i - 1][-1] in ",;:"
                or any(_fold(later) in lex.conjunctions for later in words[i + 1 : i + 3])
                or _opens_list_item(words[i + 1 : i + 3], lex)
                or any(later[-1] in ",;:" for later in words[i + 1 : i + 4])
            )
            if closes_list or place or not inside_list or _opens_tail(words, i, lex):
                cuts[i + 1] = (True, closes_list or place)
        elif _fold(word) in lex.adjunct_prepositions:
            # Never leave a participle or adjective dangling ("formed by", "based in", "responsible for").
            before = _fold(words[i - 1])
            cuts.setdefault(
                i - 1 if before.endswith(lex.participle_endings) or before in lex.participles else i, (False, False)
            )
        elif _fold(word).endswith(lex.gerund_endings):
            cuts[i] = (True, False)
        elif _fold(word) in lex.relatives:
            # The first relative clause may start the cut-off part ("... the circle of painters in Paris [who ...]"), as in
            # the last resort below, unless a participle or a comma before it has a say ("hypotheses proposing [that ...]").
            before = _fold(words[i - 1])
            if (
                (i >= 6 or _fold(word) in lex.person_relatives)
                and words[i - 1][-1] not in ",;:"
                and not before.endswith(lex.participle_endings)
                and before not in lex.participles
            ):
                cuts[i] = (False, False)
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


def lead_subject(lead: str, *, lang: str = "en") -> str:
    """The words of a lead before its copula, parentheses and all: "Oscar-Claude Monet (1840 – 1926)"."""
    lex = _WORDS[lang]
    words = lead.split()
    copula = next((i for i, word in enumerate(words) if i and _fold(word) in lex.copulas), len(words))
    return " ".join(words[:copula])


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
    """A noun phrase neither starts with a verb or adverb ("Fought in 53 BC ...") nor ends in a function word or adverb.

    Nor does it open on an adjective that wants a complement ("Popular in the 1940s", "Distinct from previous bebop").
    """
    tokens = [token for token in nlp(clue) if not token.is_punct]
    if not tokens or tokens[0].pos_ in _BAD_START or tokens[-1].pos_ in _BAD_END:
        return False
    return not (len(tokens) > 1 and tokens[0].pos_ == "ADJ" and tokens[1].pos_ in {"ADP", "SCONJ"})


def _is_finite_verb(token) -> bool:
    return token.pos_ in {"VERB", "AUX"} and "Fin" in token.morph.get("VerbForm")


def _starts_second_predicate(words: list[str], i: int, lex: _Words, doc) -> bool:
    """True when the "and" at `words[i]` is followed by a verb of its own ("a mathematician and was one of ...").

    With a tagged `doc` any finite verb counts, but not the last of a list of verbs ("describes, analyses and predicts").
    Where `finite_verbs` stays in use, the tagger adds only verbs outside a relative clause, which they go on with
    ("that has mass and occupies space").
    """
    if _fold(words[i]) != lex.and_word:
        return False
    if _fold(words[i + 1]) in lex.finite_verbs and (doc is None or not lex.tagged_second_predicate):
        return True
    if doc is None or (not lex.tagged_second_predicate and any(_fold(word) in lex.relatives for word in words[:i])):
        return False

    def finite(index: int) -> bool:
        offset = len(" ".join(words[:index])) + (index > 0)
        token = next((token for token in doc if token.idx == offset), None)
        return token is not None and _is_finite_verb(token)

    # Nor the "και" of "μιας και" (since): cutting there would end the clue on a function word.
    return finite(i + 1) and not finite(i - 1) and _fold(words[i - 1].strip(",;:")) not in lex.weak_end


# The dependencies of a plain list of examples: items, their modifiers, commas and "and".
_LIST_PARTS = {"pobj", "conj", "cc", "punct", "compound", "det", "amod", "poss", "nummod"}


def _example_span(words: list[str], nlp) -> tuple[int, int] | None:
    """The words (start, end) of the first example phrase that more of the sentence follows, or None.

    "like Paul Gauguin, Émile Bernard and Louis Anquetin" in "artists like ... to distinguish their work";
    "such as particles, atoms, or bodies," in "blocks, such as ..., that behaves as a single object".
    The phrase is the marker and the plain list after it. Not when the next word hangs from the list
    ("instruments such as the piano [that were popular]", "including the developments [spanning ...]")
    unless a comma closes the phrase.
    """
    doc = nlp(" ".join(words))
    starts = [0]
    for word in words[:-1]:
        starts.append(starts[-1] + len(word) + 1)
    for marker in doc:
        # Not examples inside a list of their own ("..., infrastructure such as power grid or ships, complex software, ...").
        if marker.dep_ != "prep" or marker.head.conjuncts:
            continue
        if marker.lower_ in {"like", "including"}:
            opener = marker
        elif marker.lower_ == "as" and marker.i and doc[marker.i - 1].lower_ == "such":
            opener = doc[marker.i - 1]
        else:
            continue
        last = marker.i
        while last + 1 < len(doc) and doc[last + 1].dep_ in _LIST_PARTS:
            last += 1
        start, end = bisect_right(starts, opener.idx) - 1, bisect_right(starts, doc[last].idx)
        listed = words[start:end]
        # Commas make a list, and a list the parser stopped inside ("such as learning, reasoning, [problem-solving ...]") is no phrase.
        open_list = any(word.endswith(",") for word in listed[:-1]) and not any(_fold(word) in _EN.conjunctions for word in listed)
        if end == len(words) or start == 0 or last == marker.i or open_list:
            continue
        following = next(token for token in doc if token.idx >= starts[end])
        if words[end - 1].endswith(",") or not any(opener.i <= up.i <= last for up in following.ancestors):
            return start, end
    return None


def _copula_definition(
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
    # What follows a colon explains the definition ("... in Physics twice: first in 1956").
    colon = next((i for i, word in enumerate(words) if word.endswith(":")), None)
    if colon is not None:
        words = words[: colon + 1]
        if _fold(words[-1].rstrip(":")) in lex.weak_end:
            return None
    # A second predicate ("a mathematician and was one of ...") does not belong to the definition.
    doc = nlp(" ".join(words)) if nlp is not None else None
    for i in range(3, len(words) - 1):
        if _starts_second_predicate(words, i, lex, doc):
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
        cut = _clean_prefix(words, max_words, lex, nlp)
        examples = _example_span(words, nlp) if lang == "en" and nlp is not None else None
        if examples and (cut is None or examples[0] < len(cut) < examples[1]):
            # The cut stops inside the examples: "artists like A, B and C to distinguish their work" loses them instead.
            start, end = examples
            if words[end - 1].endswith(","):
                words[start - 1] = words[start - 1].rstrip(",")
            words = words[:start] + words[end:]
            cut = words if len(words) <= max_words else _clean_prefix(words, max_words, lex, nlp)
        if cut is None:
            return None
        words = cut
    lowered = [_fold(word.strip(",;:")) for word in words]
    # "... of that system": the antecedent is outside the clue.
    if any(a in lex.prepositions and b in lex.demonstratives for a, b in zip(lowered, lowered[1:])):
        return None
    # "... the following conditions": the list it promises is not there.
    if lex.outside.search(" ".join(lowered)):
        return None
    # "the volume of a system is ... its state": the possessor was part of the subject, which the clue drops.
    if tail and tail.split()[0] == "of" and lex.referring & set(lowered):
        return None
    clue = " ".join(words).rstrip(" ,;:\u2013\u2014-")
    clue = clue[0].upper() + clue[1:]
    return None if nlp is not None and not _is_noun_phrase(clue, nlp) else clue


def definition_clue(
    lead_sentence: str, *, max_words: int = 16, answer: str | None = None, nlp=None, lang: str = "en"
) -> str | None:
    """The definition a lead sentence gives: through its copula ("Bebop is a style of jazz ..." -> "A style of jazz ..."),
    else, for a lead with a main verb of the answer's own ("Temperature reflects the average kinetic energy ..."),
    "It reflects the average kinetic energy ...". The first is `_copula_definition`, which keeps every sentence it defines.
    """
    clue = _copula_definition(lead_sentence, max_words=max_words, answer=answer, nlp=nlp, lang=lang)
    if clue is None and answer is not None and nlp is not None and lang == "en":
        return _it_definition(lead_sentence, max_words, answer, nlp)
    return clue


def _names_the_answer(subject: list[str], answer: str, lex: _Words) -> bool:
    """True when the subject is the answer's name or surname and nothing else.

    It ends in the answer's last word and has no common word of its own: not "Hancock's mother", "the album by Hancock".
    """
    parts = {_fold(part) for part in answer.split()}
    return (
        bool(subject)
        and _subject_tail(" ".join(subject), answer, lex) == ""
        and all(word[0].isupper() or _fold(word) in parts for word in subject)
    )


def follow_up_definition(
    sentence: str, answer: str, *, max_words: int = 16, nlp=None, lang: str = "en"
) -> str | None:
    """Cut a definition from a later sentence of the intro: "He was one of the most influential figures in jazz piano."

    For an answer whose first sentence defines nothing. The sentence has to be about the answer and say what it is:
    its subject is a personal pronoun, the answer's name or surname, or (in Greek, which drops the subject) nothing,
    and the first verb is a copula. "He played with Miles Davis and was a member of ..." and "Hancock's mother was ..."
    are refused. The cut, the word limit and the checks are those of `definition_clue`.
    """
    lex = _WORDS[lang]
    words = strip_parentheticals(sentence).split()
    copula = next((i for i, word in enumerate(words) if _fold(word) in lex.copulas), None)
    if copula is None:
        return None
    subject = [word for word in words[:copula] if _fold(word) not in lex.articles]
    if not (
        _names_the_answer(subject, answer, lex)
        or (len(subject) == 1 and _fold(subject[0]) in lex.personal)
        or (not subject and lex.pro_drop)
    ):
        return None
    # `definition_clue` wants the answer's name before the copula, as in a first sentence.
    lead = " ".join([answer.split()[-1], *words[copula:]])
    return definition_clue(lead, max_words=max_words, answer=answer, nlp=nlp, lang=lang)


# Prepositions whose phrase may be left out of a predicate: an accompaniment or a time. "in terms of ...", "at ...",
# "into ..." and "of ..." complete what comes before; the parser hangs a "with" on the nearest noun, wrongly.
_LEFT_OUT = _TIME_ADJUNCTS | {"with"}
# Adverbs that continue an earlier event. "later" and "first" read on their own ("later joined Miles Davis's quintet").
_SEQUENCERS = frozenset("then soon subsequently afterwards afterward thereafter next meanwhile eventually".split())
# Words that point at something said before: an opening phrase that has them cannot be left out, nor can a predicate
# that has them when the opening phrase is.
_POINTS_BACK = frozenset(
    "also too other others another such same then there this these those it its they them their both former latter"
    " next following previous subsequent soon later afterwards thereafter".split()
)
# A clue may end on these: not on a verb ("to excel"), an adjective or an adverb.
_COMPLETE_END = {"NOUN", "PROPN", "NUM", "PRON"}
# A head that "who" can follow: "An American jazz pianist, composer and bandleader", no apposition or participle.
_OCCUPATION_TAGS = {"DET", "ADJ", "NOUN", "PROPN", "CCONJ", "PUNCT"}


def _is_plain_occupation(kinds) -> bool:
    return all(token.pos_ in _OCCUPATION_TAGS for token in kinds) and kinds[-1].pos_ in {"NOUN", "PROPN"}


def _own_clause(doc, verb) -> int | None:
    """Character offset of a clause with a subject of its own coordinated with `verb`, else of a semicolon or colon.

    "Armstrong received accolades[, and his influence crossed genres]" is about something else than the person.
    """
    for other in verb.conjuncts:
        if other.i > verb.i and any(child.dep_ in {"nsubj", "nsubjpass"} for child in other.children):
            start = other.left_edge.i
            while doc[start - 1].dep_ in {"cc", "punct"}:
                start -= 1
            return doc[start].idx
    return next((token.idx for token in doc[verb.i :] if token.text in {";", ":"}), None)


def _cut_predicate(words: list[str], room: int, lex: _Words, nlp) -> list[str] | None:
    """The longest clean cut of a predicate to at most `room` words that loses an adjunct, never a complement.

    The cut ends on a noun, after a comma or before a preposition that hangs from the verb ("shared the 1965 Nobel
    Prize in Physics [with ...]"). Not before "in the history of jazz" after "one of the most influential figures",
    nor before "which came to be known as the Boston School", and not inside a list of nouns or titles ("noted for
    Chemin en été[, La dispersion ...]"). Without such a cut, the part before a second predicate ("started his career
    with Art Blakey [and later moved on to play ...]") will do.
    """
    doc = nlp(" ".join(words))

    def ends_on_noun(size: int) -> bool:
        offset = len(" ".join(words[:size])) + 1
        return [token for token in doc if token.idx < offset and not token.is_punct][-1].pos_ in _COMPLETE_END

    limit = room
    while (cut := _clean_prefix(words, limit, lex, nlp)) is not None:
        offset = len(" ".join(cut)) + 1
        first_dropped = next(token for token in doc if token.idx == offset)
        dropped = _fold(words[len(cut)])
        if cut[-1].endswith(","):  # an aside, but no "of ..." that completes a noun of the cut
            noun_complement = dropped in lex.prepositions and first_dropped.head.pos_ in {"NOUN", "PROPN"}
            clean = not words[len(cut)][0].isupper() and (first_dropped.lower_ in _LEFT_OUT or not noun_complement)
        else:
            clean = first_dropped.lower_ in _LEFT_OUT
        # A list of nouns, or of three verbs, is not left open; nor is a verb left without its complement.
        split_list = any(
            other.idx >= offset
            and (other.pos_ in {"NOUN", "PROPN"} or (other.pos_ == "VERB" and any(c.idx < offset for c in token.conjuncts)))
            for token in doc
            if token.idx < offset
            for other in token.conjuncts
        )
        top = first_dropped
        while top.head.idx >= offset and top.head != top:
            top = top.head
        complement = top.head.idx < offset and (top.dep_ in {"xcomp", "ccomp"} or (top.dep_ == "acl" and not cut[-1].endswith(",")))
        if clean and not split_list and not complement and ends_on_noun(len(cut)):
            break
        limit = len(cut) - 1
    second = next((i for i in range(5, min(len(words) - 1, room + 1)) if _starts_second_predicate(words, i, lex, doc)), None)
    if second is not None and ends_on_noun(second) and (cut is None or second > len(cut)):
        return words[:second]
    return cut


def _predicate_words(doc, verb, start: int, lex: _Words, nlp) -> list[str]:
    """The words of `doc` from token `start` to the end of the clause of `verb`, as one predicate.

    "studied the nature of heat, and discovered its relationship" loses the comma before the second verb.
    """
    words = doc.text[doc[start].idx : _own_clause(doc, verb)].rstrip(".").split()
    words[-1] = words[-1].rstrip(",;:")
    rest = nlp(" ".join(words))
    for i in [i for i in range(1, len(words) - 1) if words[i - 1].endswith(",") and _starts_second_predicate(words, i, lex, rest)]:
        words[i - 1] = words[i - 1].rstrip(",")
    return words


def _it_definition(lead_sentence: str, max_words: int, answer: str, nlp) -> str | None:
    """Turn "Temperature reflects the average kinetic energy of the particles ..." into "It reflects the average kinetic energy ...".

    For a lead whose first finite verb is a main verb, not "be" or an auxiliary. After an opening phrase that ends in
    a comma ("In classical thermodynamics,") and an article, the subject is the answer's name and nothing else, word
    for word, a plural ending allowed: "Research concerning the relationship between ... entropy and ... life began",
    "The mathematical concept of a Hilbert space generalizes" and "Curia in ancient Rome referred" are refused.
    Not "have" ("It has a long tradition"), nor an object that opens on a number word or a vague quantifier
    ("It refers to two related concepts"). "It" for a singular subject, "They" for a plural one; not for a person.
    The predicate is kept whole when it fits,     else cut by `_cut_predicate`.
    """
    lex = _EN
    doc = nlp(strip_parentheticals(lead_sentence))
    verb = next((token for token in doc if _is_finite_verb(token)), None)
    if verb is None or verb.pos_ != "VERB" or verb.lemma_ in {"be", "have"} or verb.dep_ != "ROOT":
        return None
    subject_head = next((child for child in verb.children if child.dep_ == "nsubj"), None)
    if subject_head is None:
        return None
    end = verb.i
    while end and doc[end - 1].pos_ == "ADV":
        end -= 1
    start = max((token.i + 1 for token in doc[:end] if token.text == ","), default=0)
    if start and doc[0].pos_ != "ADP":
        return None
    subject = [word for word in doc[start:end].text.split()]
    if subject and subject[0].lower() in {"the", "a", "an"}:
        subject = subject[1:]
    names = answer.split()
    if len(subject) != len(names) or not all(_subject_tail(a, b, lex) == "" for a, b in zip(subject, names)):
        return None
    first = start + (doc[start].lower_ in {"the", "a", "an"})
    if any(ent.label_ == "PERSON" and ent.start <= first and ent.end >= end for ent in doc.ents):
        return None  # "Constantine IX Monomachos reigned": not "It" ("Einstein field equations" are not a person)
    # "It refers to two related concepts" says how many, not what: the object, after a preposition, opens on a number word or a vague quantifier.
    after = next((token for token in doc[verb.i + 1 :] if token.pos_ != "ADP"), None)
    if after is not None and (after.like_num and after.is_alpha or _VAGUE_OPENER.match(doc[after.i :].text)):
        return None
    words = _predicate_words(doc, verb, end, lex, nlp)
    room = max_words - 1
    cut = words if len(words) <= room else _cut_predicate(words, room, lex, nlp)
    if cut is None:
        return None
    lowered = [_fold(word.strip(",;:")) for word in cut]
    # "... that civilization": the antecedent is outside the clue, as in `_copula_definition`.
    if any(a in lex.prepositions and b in lex.demonstratives for a, b in zip(lowered, lowered[1:])) or lex.outside.search(" ".join(lowered)):
        return None
    plural = verb.tag_ == "VBP" or subject_head.tag_ in {"NNS", "NNPS"}
    clue = f"{'They' if plural else 'It'} {' '.join(cut)}".rstrip(" ,;:\u2013\u2014-")
    return clue if clue.count('"') % 2 == 0 else None


def who_definition(
    head: str, sentence: str, answer: str, *, max_words: int = MAX_CLUE_WORDS, nlp=None, lang: str = "en"
) -> str | None:
    """Join a generic head and a later sentence about the same person: "An English physicist who studied the nature of heat".

    `head` ("An English physicist, mathematician and brewer") says too little, `sentence` ("Joule studied the nature
    of heat, and discovered its relationship to mechanical work.") adds what the person did. Its subject is "He", "She",
    the name or the surname and nothing else, and its first verb is finite and follows it ("He later joined ..." is fine).
    A head that is no plain list of occupations gives its first item ("A French painter"), if that is one. One opening
    phrase that ends in a comma ("In 1864,") before the subject is left out, unless it or the predicate points back
    ("also", "other", "there", "that year"). The predicate is kept whole when it fits `max_words` with the head, the
    head being cut back to its first item if that makes it fit; otherwise it is cut by `_cut_predicate`, and without a clean cut there is no result. A sequencing
    adverb ("then", "soon") or "was born" refuses the sentence. For a person's page only: "who" is wrong for anything
    else. Needs the parser.
    """
    if nlp is None or lang != "en":
        return None
    lex = _WORDS[lang]
    if not _is_plain_occupation(nlp(head)):
        # "A French painter, printmaker and a member of the circle of painters in Paris": its first item will do.
        if "," not in head or not _is_plain_occupation(nlp(head.split(",")[0])):
            return None
        head = head.split(",")[0]
    doc = nlp(strip_parentheticals(sentence))
    verb = next((token for token in doc if _is_finite_verb(token)), None)
    if verb is None:
        return None
    # One opening phrase that ends in a comma ("In 1864,", "Born in Belfast,") stays out of the clue.
    start = max((token.i + 1 for token in doc[: verb.i] if token.text == ","), default=0)
    end = verb.i
    while end > start and doc[end - 1].pos_ == "ADV":
        end -= 1
    subject = [token.text for token in doc[start:end]]
    if not (_names_the_answer(subject, answer, lex) or [word.lower() for word in subject] in (["he"], ["she"])):
        return None
    if start and {token.lower_ for token in doc[:start]} & _POINTS_BACK:
        return None
    if _VAGUE_OPENER.match(doc[verb.i + 1 :].text):
        return None
    # "who then moved to the U.S." points at an event the clue does not have, "was born in ..." says only where.
    around = {token.lower_ for token in doc[end : verb.i + 2]}
    if around & _SEQUENCERS or "born" in {token.lower_ for token in doc[verb.i + 1 : verb.i + 3]}:
        return None
    words = _predicate_words(doc, verb, end, lex, nlp)
    short = head.split(",")[0]
    room = max_words - len(short.split()) - 1
    cut = words if len(words) <= room else _cut_predicate(words, room, lex, nlp)
    if cut is None:
        return None
    # What follows the dropped phrase must not lean on it: "also", "other", "there", "it", "that year" ...
    if start and {_fold(word.strip(_PUNCTUATION)) for word in cut} & _POINTS_BACK:
        return None
    used = head if len(cut) <= max_words - len(head.split()) - 1 else short
    clue = f"{used} who {' '.join(cut)}".rstrip(" ,;:\u2013\u2014-")
    tokens = [token for token in nlp(clue) if not token.is_punct]
    # "A Russian playwright ... who was a physician by profession": the predicate adds nothing the head had.
    nouns = [token for token in tokens if token.pos_ == "NOUN" and token.dep_ != "compound"]
    repeats = any(a.lemma_ == b.lemma_ for a in nouns if a.idx < len(used) for b in nouns if b.idx >= len(used))
    return clue if clue.count('"') % 2 == 0 and tokens[-1].pos_ in _COMPLETE_END and not repeats else None


# A word is rare when at most this share of the pool's lead sentences (and never fewer than two) contain it.
_RARE_LEAD_SHARE = 1 / 30
_WORD = re.compile(r"\w+")


def lead_word_counts(leads: list[str]) -> Counter[str]:
    """For each word, the number of lead sentences it occurs in."""
    return Counter(word for lead in leads for word in set(_WORD.findall(lead.lower())))


# A definition that opens on a nationality ("An American jazz drummer") needs this many informative words, any other one.
_MIN_INFORMATIVE_WORDS = 2
# Head nouns that name a kind of word, not a kind of thing: generic unless the definition says what the word means.
_KIND_OF_WORD = {"term", "name", "word", "phrase", "concept", "notion", "idea", "expression", "title", "label", "designation"}
_MEANING_MARKERS = {
    "for", "referring", "refers", "describing", "describes", "denoting", "denotes", "meaning", "means", "used", "associated",
    "encompassing", "that", "which",
}
# "Any of many different quantities", "One of several ...": the class is not named. "One of the earliest ..." is.
# So is "A wide range of physical processes", with nothing but a bare plural after the "of", and "A separate part of a
# larger system", with nothing but the system after it: what follows
# "A large group of Germanic peoples first reported by Julius Caesar" pins the group down.
_VAGUE_OPENER = re.compile(
    r"(?:any|various|several|many|some|numerous|multiple)\b"
    r"|one of (?:the )?(?:several|many|various|numerous|multiple|some|a number of|a few)\b"
    r"|(?:an?|the) (?:wide|broad|large|great|vast|whole|diverse) "
    r"(?:range|variety|number|array|class|set|group|family|collection) of (?-i:[a-z-]+(?: [a-z-]+)?)$"
    r"|(?:an?|the) (?:[a-z-]+ )?(?:part|piece|portion|component|element) of (?:an?|the) "
    r"(?:larger|bigger|greater|whole) (?-i:[a-z-]+(?: [a-z-]+)?)$",
    re.IGNORECASE,
)


def _is_kind_of_word(word: str) -> bool:
    return word in _KIND_OF_WORD or word[:-1] in _KIND_OF_WORD


def _nationalities(doc) -> list:
    """Tokens of a tagged clue that name a people: a NORP entity, or a capitalised adjective ("Brazilian")."""
    return [token for token in doc if token.ent_type_ == "NORP" or (token.i and token.pos_ == "ADJ" and token.text[0].isupper())]


def _informative_words(
    clue: str, counts: Counter[str], pool_size: int, nationalities: set[str], seed_title: str | None
) -> set[str]:
    """Words of the clue that tell it apart: rare among the pool's leads, not a seed word, a kind of word or a nationality."""
    seed_words = [w for w in _WORD.findall((seed_title or "").lower()) if len(w) >= 4]
    limit = max(2, pool_size * _RARE_LEAD_SHARE)
    return {
        word
        for word in set(_WORD.findall(clue.lower())) - nationalities
        if any(ch.isdigit() for ch in word)
        or (counts[word] <= limit and not _is_kind_of_word(word) and not any(shares_root(word, seed) for seed in seed_words))
    }


def _generic_reason(clue: str, counts: Counter[str], pool_size: int, nlp, seed_title: str | None) -> str | None:
    """Why a definition fits too many answers, or None."""
    if _VAGUE_OPENER.match(clue):
        return "vague quantifier"
    words = _WORD.findall(clue.lower())
    doc = nlp(clue) if nlp is not None else None
    if doc is None:
        kinds = {word for word in words[:4] if _is_kind_of_word(word)}
    else:  # "word processor": a compound modifier is not the head
        kinds = {token.lower_ for token in doc if token.i < 4 and token.dep_ != "compound" and _is_kind_of_word(token.lower_)}
    if kinds and not _MEANING_MARKERS & set(words):
        return "kind of word"
    people = _nationalities(doc) if doc is not None else []
    # The answer's own nationality comes before any preposition; "A canvas by the French painter Claude Monet" has another's.
    # Nor does "the common English title" give one: a nationality on a kind of word names a language.
    own = any(
        not any(before.pos_ == "ADP" for before in doc[: token.i])
        for token in people
        if not _is_kind_of_word(token.head.lower_)
    )
    nationalities = {word for token in people for word in _WORD.findall(token.lower_)}
    informative = _informative_words(clue, counts, pool_size, nationalities, seed_title)
    if len(informative) < (_MIN_INFORMATIVE_WORDS if own else 1):
        return "too few informative words"
    return None


def is_generic_definition(clue: str, counts: Counter[str], pool_size: int, nlp=None, *, seed_title: str | None = None) -> bool:
    """True when the definition fits too many answers of the pool.

    "A German physicist" or "A genre of dance music" use words common among the pool's lead sentences, "The capital of
    Lebanon" does not. A clue needs one informative word: rare among the leads, not a word of the seed title, not a
    kind of word, not a nationality (with `nlp`: "Brazilian" is rare, but names no one). Numbers always inform. A clue
    that gives the answer's own nationality needs `_MIN_INFORMATIVE_WORDS`: "An American jazz drummer" names a crowd,
    so one rare word does not tell its members apart, while "A brass instrument" does not need a second one. A head
    noun that names a word ("A term coined in the 1980s") or a vague quantifier ("Any of many different quantities")
    is generic as well.
    """
    return _generic_reason(clue, counts, pool_size, nlp, seed_title) is not None


def has_verb(clue: str, nlp) -> bool:
    """False for titles, citations and list entries, and for a clause cut before its main verb: poor fill-ins.

    A clue cut at a clause boundary needs a finite verb, modals included ("Apart from informing the public about ____
    history, ..." has none), or a verb with a subject of its own, because the tagger takes many a past tense for a
    participle. A whole sentence needs any verb: the stricter test would also reject the sentences where the tagger
    made that mistake and the parser found no subject ("____ evolved from historical ideas about the vacuum.").
    """
    # The tokenizer splits a blank into its underscores, and the tagger may call one of them a verb.
    verbs = [token for token in nlp(clue) if token.pos_ in {"VERB", "AUX"} and "_" not in token.text]
    if not clue.endswith("..."):
        return bool(verbs)
    return any(_is_finite_verb(token) or any(child.dep_ == "nsubj" for child in token.children) for token in verbs)


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
    for token in nlp(sentence):
        if token.dep_ in {"nsubj", "nsubjpass", "csubj", "csubjpass"}:
            token_text = token.text.lower()
            token_lemma = token.lemma_.lower() if token.lemma_ else token_text
            if token_text in answer_tokens or token_lemma in answer_lemmas:
                return True
    return False


def _bare_words(text: str) -> list[str]:
    """Words spelt with a modifier letter, an apostrophe or a combining mark, without it: "Hawaiʻi" and "Hawai'i" give "hawaii"."""
    words = []
    for raw in text.split():
        letters = [ch for ch in strip_diacritics(raw.lower()) if unicodedata.category(ch) not in {"Lm", "Sk"} and ch not in "'’‘"]
        if len(letters) < len(raw):
            words += "".join(ch if ch.isalnum() else " " for ch in letters).split()
    return words


def _content_lemmas(text: str, nlp, lang: str) -> set[str]:
    stopwords = get_stopwords(lang)
    if nlp is None:
        tokens = [(word, word.upper() in stopwords) for word in tokenize(text, lang=lang)]
    else:
        tokens = [(strip_diacritics(token.lemma_.lower()), token.is_stop) for token in nlp(text) if token.is_alpha]
    tokens += [(word, word.upper() in stopwords) for word in _bare_words(text)]
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
    # The same content word is a leak, however short ("arts" for "French art"); a root needs more letters in common.
    return any(a == c or shares_root(a, c) for a in answer_lemmas for c in clue_lemmas)


# A blank that is one part of a longer proper name ("the ____ Florida Institute", "the Rome-Berlin ____") clues nothing.
_CAPITAL = "A-ZΑ-ΫΆΈ-ΊΌΎΏ"  # Latin and Greek, tonos included
_PART_OF_A_NAME = re.compile(
    BLANK + rf"(?= +[{_CAPITAL}])|[{_CAPITAL}][\w']*[ \-\N{{EN DASH}}\N{{EM DASH}}](?={BLANK})"
)


def blank_is_part_of_a_name(sentence: str) -> bool:
    """True when a blank is followed by a capitalised word, or preceded by one that does not open the sentence."""
    return any(m.start() > 0 or m.group().startswith(BLANK) for m in _PART_OF_A_NAME.finditer(sentence.strip()))


_PUNCTUATION = ",;:.!?\"'()«»“”"


# A fill-in that opens on a word whose referent is in the previous sentence ("It is the cause of the ____.")
# stands on nothing. Folded: no case, no accents.
_BACK_REFERENCES = {
    "it", "its", "this", "that", "these", "those", "such", "he", "she", "they", "his", "her", "their", "another",
    "and", "but", "or", "so", "yet", "nor", "now", "then", "thus", "hence", "therefore", "however", "also",
    "moreover", "furthermore", "meanwhile", "instead", "consequently", "similarly", "likewise", "rather",
    "αυτος", "αυτη", "αυτο", "αυτοι", "αυτες", "αυτα", "αυτου", "αυτης", "αυτων", "εκεινος", "εκεινη", "εκεινο",
    "και", "αλλα", "ομως", "ωστοσο", "ετσι", "επισης", "επομενως", "τοτε", "επιπλεον", "ουτε", "προκειται",
}


def _outside_reference(clue: str, nlp, lang: str) -> str | None:
    """Name the rule when a fill-in leans on an earlier sentence in the middle, else None.

    A personal pronoun with no noun, name or blank before it ("In 1870 he introduced ..."), "this pump" or "της
    εμφάνισης αυτής" when no pump or εμφάνιση came before, "the former", "doing so", "as noted above", and with `nlp`
    "the term" or "the name" with a verb right after it ("the term was confined to ..."), and a "his" or "her" with no noun
    before it when a pronoun is the subject ("On his 1961 tour of ____, he was booed"). Greek, which
    drops its subject, also leans back with a finite verb that opens the clue without a subject ("Εφαρμόζεται στην
    ____") and with a possessive "της" that has no noun before it. With `nlp` the nouns are the tagged ones; without
    it every word that is not a stopword or a number counts as one, so "Because of this, it ..." is not caught, and
    neither are the two Greek cases.
    """
    lex = _WORDS[lang]
    if lex.outside.search(" ".join(_fold(word) for word in clue.split())):
        return "phrase"
    blanks = {i for m in re.finditer(re.escape(BLANK), clue) for i in range(*m.span())}
    clitics: set[int] = set()
    if nlp is not None:
        # The tokenizer splits a blank into its underscores.
        tokens = [token for token in nlp(clue) if not (token.idx in blanks and token.idx - 1 in blanks)]
        items = [(token.text, token.idx in blanks or token.pos_ in {"NOUN", "PROPN"}, token.pos_) for token in tokens]
        # "the term was confined to ...": which term, an earlier sentence says. Not "the term "jazz" was ..." or "the name of ...".
        if any(
            the.lower_ == "the" and label.lower_ in lex.labels and verb.pos_ == "AUX"
            for the, label, verb in zip(tokens, tokens[1:], tokens[2:])
        ):
            return "label"
        # "On his 1961 tour of ____, he was booed": the "his" is the subject pronoun's, and nobody is named. "In his 1905
        # paper, Einstein proposed ..." and "In her memoir, ____ describes ..." name the person after it.
        if any(
            token.tag_ == "PRP$" and token.lower_ in {"his", "her"} and not any(noun for _, noun, _ in items[:i])
            for i, token in enumerate(tokens)
        ) and any(
            token.dep_ in {"nsubj", "nsubjpass"} and token.head.dep_ == "ROOT" and token.lower_ in lex.personal
            for token in tokens
        ):
            return "possessive"
        if lex.pro_drop:
            first = tokens[0]  # "Μπορούμε" (we can) names its subject: only a third person has it in an earlier sentence
            if _is_finite_verb(first) and first.morph.get("Person") == ["3"] and not any(
                child.dep_.startswith(("nsubj", "csubj")) for child in first.children
            ):
                return "verb without subject"
            clitics = {i for i, token in enumerate(tokens) if token.pos_ == "PRON" and "Yes" in token.morph.get("Poss")}
    else:
        ignored = get_stopwords(lang) | {word.upper() for word in _BACK_REFERENCES}
        items = []
        for word in clue.split():
            core = _fold(word.strip(_PUNCTUATION))
            article = core in lex.articles
            noun = BLANK in word or (core.replace("-", "").isalpha() and core.upper() not in ignored and not article)
            items.append((word, noun, "DET" if article else "NOUN" if noun else ""))
    cores = [_fold(text.strip(_PUNCTUATION)) for text, _, _ in items] + ["", ""]
    for i, word in enumerate(cores[:-2]):
        if (word in lex.personal or i in clitics) and not any(noun for _, noun, _ in items[:i]):
            return "pronoun"
        # "αυτό που ...", "αυτός ο οποίος ..." (the one that) point at nothing.
        if word not in lex.pointing or cores[i + 2 if cores[i + 1] in lex.articles else i + 1] in lex.relatives:
            continue
        # The noun a demonstrative goes with comes after it ("this pump", "αυτή τη μέθοδο") or, in Greek, right
        # before it ("της εμφάνισης αυτής").
        if i + 1 < len(items) and items[i + 1][2] in lex.pointing_tags:
            if word.endswith(lex.genitive_endings) != cores[i + 1].endswith(lex.genitive_endings):
                continue  # a pronoun with its referent in the clause, not a determiner
            at = next((k for k in range(i + 1, len(items)) if items[k][1]), None)
        elif lex.pointing_follows_noun and i and items[i - 1][1] and items[i - 1][0][-1] not in _PUNCTUATION:
            at = i - 1
        else:
            continue
        if at is not None and not any(cores[at] == seen or shares_root(cores[at], seen) for seen in cores[: min(at, i)]):
            return "demonstrative"
    return None


# "A, B, C etc.": the abbreviation closes a list as "and others" does. Folded, without the full stops of "κ.ά.".
_ET_CETERA = {"etc", "κλπ", "κτλ", "κα"}
# A list item between two commas has at most this many words: a longer stretch is a clause or an apposition.
_MAX_LIST_ITEM_WORDS = 3
# A blank with one of these relations is part of an item ("____ drumming"), and an item with such a child says more
# than the blank does ("integrated ____"): not every item of the list fits it.
_MODIFIES_A_NOUN = {"amod", "compound", "poss", "nummod"}
_ITEM_MODIFIERS = _MODIFIES_A_NOUN | {"nmod", "acl", "relcl", "acl:relcl"}


def _list_stretches(words: list[str], lex: _Words) -> tuple[list[tuple[int, int]], list[str]]:
    """Split a sentence at its commas and coordinators.

    Returns the stretches of words between them as (start, end) and what follows each: "," for a comma, "&" for a
    coordinator, ",&" for both ("A, and B"), "" where a list cannot go on (other punctuation, the end of the
    sentence). "etc." is a coordinator and a last item in one.
    """
    stretches: list[tuple[int, int]] = []
    after: list[str] = []
    start = 0
    for i, word in enumerate(words):
        bare = word.strip(_PUNCTUATION)
        et_cetera = "." in word and _fold(bare).replace(".", "") in _ET_CETERA or _fold(bare) == "etc"
        last = word.rstrip("\"')»”")[-1:]
        if bare.lower() in lex.coordinators or et_cetera:
            if i > start:
                stretches.append((start, i))
                after.append("&")
            elif after and after[-1] == ",":
                after[-1] = ",&"
            start = i + 1
            if et_cetera:
                stretches.append((i, start))
                after.append("")
        elif (last and last in ",;:.!?") or not bare.strip("-–—"):  # a dash on its own ends a stretch as well
            stretches.append((start, i + 1))
            after.append("," if last == "," else "")
            start = i + 1
    if start < len(words):
        stretches.append((start, len(words)))
        after.append("")
    if after:
        after[-1] = ""
    return stretches, after


def _list_around_blank(words: list[str], lex: _Words) -> int | None:
    """The number of commas in the list of three or more items that every blank is an item of, or None.

    The blank has to be a whole item: "Egypt, ____ and Tunisia", "led by ____, Dizzy Gillespie, Thelonious Monk and
    others", "pressure, temperature, and ____", but not "____ oil and wine" or "Sissle and ____'s Shuffle Along".
    Items between two commas are short, a coordinator closes the list, and a stretch that opens on a preposition
    the blank does not have ("In 1860, Monet and ____") or is one linking word ("however") is no item.
    """
    stretches, after = _list_stretches(words, lex)
    function = lex.determiners | lex.prepositions

    def size(j: int) -> int:
        return stretches[j][1] - stretches[j][0]

    def opener(j: int) -> str:
        return _fold(words[stretches[j][0]].strip(_PUNCTUATION)) if 0 <= j < len(stretches) else ""

    def aside(j: int, preposition: str | None) -> bool:
        short_adverbial = size(j) <= _MAX_LIST_ITEM_WORDS and opener(j) in lex.prepositions
        return (size(j) == 1 and opener(j) in _BACK_REFERENCES) or (short_adverbial and opener(j) != preposition)

    def items_before(i: int, preposition: str | None) -> int:
        """Items of the list before stretch `i`, each of them followed by a comma."""
        count = 0
        while i > 0 and after[i - 1] == "," and not aside(i - 1, preposition):
            short = size(i - 1) <= _MAX_LIST_ITEM_WORDS
            # A short stretch after a coordinator closes another list, a long one between commas is a clause or an
            # apposition. Any other long stretch ends in the first item of this list.
            if i > 1 and ("&" in after[i - 2] if short else after[i - 2] == ","):
                break
            count += 1
            i -= 1
            if not short:
                break
        return count

    found = None
    for k, (start, end) in enumerate(stretches):
        for b in range(start, end):
            if BLANK not in words[b]:
                continue
            if words[b].strip(_PUNCTUATION) != BLANK:
                return None
            lead = [_fold(word.strip(_PUNCTUATION)) for word in words[start:b]]
            own = len(lead) - next((n for n, word in enumerate(reversed(lead)) if word not in function), len(lead))
            whole = own == 0  # nothing but "the" or "με την" before the blank
            preposition = next((word for word in lead[own:] if word in lex.prepositions), None)
            # "Most often, in ____, mass and energy ...": with a preposition all to itself the blank is an adverbial.
            if whole and preposition and preposition not in (opener(k - 1), opener(k + 1)):
                return None
            items = commas = 0
            if k and "&" in after[k - 1] and whole:  # the last item: "A, B and ____"
                more = items_before(k - 1, preposition) if size(k - 1) <= _MAX_LIST_ITEM_WORDS else 0
                items, commas = 2 + more, int("," in after[k - 1]) + more
            elif b == end - 1 and after[k]:  # the first item or one in the middle: "____, B and C", "A, ____, C and D"
                j, items, closed = k, 1, False
                while after[j] and not closed and not aside(j + 1, preposition):
                    items += 1
                    commas += int("," in after[j])
                    closed = "&" in after[j]
                    j += 1
                    if size(j) > _MAX_LIST_ITEM_WORDS:
                        break
                if not closed:
                    return None
                more = items_before(k, preposition) if whole else 0
                items, commas = items + more, commas + more
            if items < 3:
                return None
            found = commas if found is None else min(found, commas)
    return found


def _blank_is_one_of_a_list(sentence: str, answer: str, nlp, lang: str) -> bool:
    """True when the answer is one item of a list of three or more in its sentence: any of the others would fit.

    Two commas and a coordinator ("R&B, ____, and pop music", "τη Φυσική, ____, Μεταλλουργία, Βιολογία κλπ.") make a
    list. One comma ("Egypt, ____ and Tunisia") reads the same as "two concepts, ____ and entropy": that is a list
    only when the parser coordinates the answer with two other words, so without `nlp` it is not counted. The parser
    also shows an apposition ("His son, ____, conquered Jerusalem and Masada.") and an item that says more than the
    blank ("integrated ____, machine vision, and ..."), which stay clues.
    """
    commas = _list_around_blank(mask_answer(sentence, answer).split(), _WORDS[lang])
    if commas is None:
        return False
    if nlp is None:
        return commas >= 2
    doc = nlp(sentence)
    for match in answer_pattern(answer).finditer(sentence):
        span = [token for token in doc if match.start() <= token.idx < match.end()]
        head = next((token for token in span if token.head == token or token.head not in span), None)
        if head is None:
            continue
        if head.dep_ in _MODIFIES_A_NOUN or any(child.dep_ in _ITEM_MODIFIERS for child in head.children if child not in span):
            return False
        if (head.dep_ == "appos" and not head.conjuncts) or (commas < 2 and len(head.conjuncts) < 2):
            return False
    return True


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
    if require_blank and _fold(clue.split()[0].strip(_PUNCTUATION)) in _BACK_REFERENCES:
        return False
    if require_blank and _outside_reference(clue, nlp, lang):
        return False
    if original_sentence and is_answer_subject(original_sentence, answer, nlp):
        return False
    if require_blank and original_sentence and blank_is_part_of_a_name(mask_answer(original_sentence, answer)):
        return False
    if require_blank and original_sentence and _blank_is_one_of_a_list(original_sentence, answer, nlp, lang):
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
