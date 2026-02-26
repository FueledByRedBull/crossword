from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    return [sentence.strip() for sentence in _SENTENCE_END.split(text) if sentence.strip()]


def mask_answer(sentence: str, answer: str) -> str:
    if not sentence or not answer:
        return sentence
    pattern = re.compile(re.escape(answer), re.IGNORECASE)
    return pattern.sub("____", sentence)


def select_sentence(sentences: list[str], answer: str) -> str | None:
    for sentence in sentences:
        if re.search(re.escape(answer), sentence, re.IGNORECASE):
            return sentence
    return None


def trim_clause(sentence: str, max_words: int = 12) -> str:
    words = sentence.split()
    if len(words) <= max_words:
        return sentence
    return " ".join(words[:max_words])


def clue_pass_extract(text: str, answer: str) -> str | None:
    sentence, _ = clue_pass_extract_with_offset(text, answer)
    return sentence


def clue_pass_extract_with_offset(text: str, answer: str) -> tuple[str | None, int | None]:
    sentences = split_sentences(text)
    for idx, sentence in enumerate(sentences):
        if re.search(re.escape(answer), sentence, re.IGNORECASE):
            return sentence, idx
    return None, None


def clue_pass_mask_trim(sentence: str, answer: str, max_words: int = 12) -> str:
    masked = mask_answer(sentence, answer)
    return trim_clause(masked, max_words=max_words)


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


def has_morphological_leakage(clue: str, answer: str, nlp) -> bool:
    if nlp is None:
        return False
    answer_lemmas = lemmatize_text(answer, nlp)
    if not answer_lemmas:
        return False
    clue_lemmas = lemmatize_text(clue, nlp)
    if any(lemma in clue_lemmas for lemma in answer_lemmas):
        return True
    # Fallback: check for substring/prefix cousins (e.g., entropy vs entropic).
    def _common_prefix_len(a: str, b: str) -> int:
        count = 0
        for ch_a, ch_b in zip(a, b):
            if ch_a != ch_b:
                break
            count += 1
        return count

    for lemma in answer_lemmas:
        for clue_lemma in clue_lemmas:
            if lemma in clue_lemma or clue_lemma in lemma:
                return True
            if _common_prefix_len(lemma, clue_lemma) >= 4:
                return True
    return False


def clue_pass_validate(
    clue: str,
    answer: str,
    *,
    min_words: int = 6,
    nlp=None,
    original_sentence: str | None = None,
) -> bool:
    if not clue:
        return False
    if len(clue.split()) < min_words:
        return False
    if re.search(re.escape(answer), clue, re.IGNORECASE):
        return False
    if has_morphological_leakage(clue, answer, nlp):
        return False
    if original_sentence and is_answer_subject(original_sentence, answer, nlp):
        return False
    return True


def _is_definitional(clue: str) -> bool:
    lowered = clue.lower()
    return lowered.startswith(("the ", "a ", "an ")) or " of " in lowered


def enforce_diversity(
    clues: list[dict],
    max_per_bucket: int = 2,
    definitional_cap: int = 3,
) -> list[dict]:
    bucket_counts: dict[str, int] = {}
    definitional_count = 0
    output = []
    for clue in clues:
        tokens = clue["clue"].split()
        bucket = " ".join(tokens[:3]).lower()
        bucket_counts.setdefault(bucket, 0)
        if bucket_counts[bucket] >= max_per_bucket:
            continue
        if _is_definitional(clue["clue"]):
            if definitional_count >= definitional_cap:
                continue
            definitional_count += 1
        bucket_counts[bucket] += 1
        output.append(clue)
    return output
