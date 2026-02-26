from __future__ import annotations

from pathlib import Path
import re


_LEXICON_SPLIT_RE = re.compile(r"[\s,\t]+")


def normalize_lexicon_token(token: str) -> str:
    return "".join(ch for ch in token.upper() if ch.isalpha())


def load_lexicon_scores(path: str | Path | None) -> dict[str, float]:
    if path is None:
        return {}
    lexicon_path = Path(path)
    if not lexicon_path.exists() or not lexicon_path.is_file():
        return {}

    raw_scores: dict[str, float] = {}
    numeric_score_seen = False

    with lexicon_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            parts = [part for part in _LEXICON_SPLIT_RE.split(text) if part]
            if not parts:
                continue
            token = normalize_lexicon_token(parts[0])
            if not token:
                continue

            score = 1.0
            if len(parts) > 1:
                try:
                    score = float(parts[1])
                    numeric_score_seen = True
                except ValueError:
                    score = 1.0
            current = raw_scores.get(token)
            if current is None or score > current:
                raw_scores[token] = score

    if not raw_scores:
        return {}

    if not numeric_score_seen:
        return {token: 1.0 for token in raw_scores}

    values = list(raw_scores.values())
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        return {token: 1.0 for token in raw_scores}
    scale = hi - lo
    return {token: (score - lo) / scale for token, score in raw_scores.items()}


def lexicon_score_for_token(token: str, lexicon: dict[str, float]) -> float:
    if not token or not lexicon:
        return 0.0
    normalized = normalize_lexicon_token(token)
    if not normalized:
        return 0.0
    return float(lexicon.get(normalized, 0.0))


def lexicon_score_for_tokens(tokens: list[str], lexicon: dict[str, float]) -> float:
    if not tokens or not lexicon:
        return 0.0
    scores = [lexicon_score_for_token(token, lexicon) for token in tokens]
    scored = [score for score in scores if score > 0.0]
    if not scored:
        return 0.0
    return sum(scored) / len(scored)


def load_word_list(
    path: str | Path | None,
    *,
    min_len: int = 3,
    max_len: int = 12,
    max_per_length: int | None = None,
) -> list[str]:
    if path is None:
        return []
    word_path = Path(path)
    if not word_path.exists() or not word_path.is_file():
        return []

    buckets: dict[int, set[str]] = {}
    with word_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            token = text.split()[0]
            if not token:
                continue
            normalized = normalize_lexicon_token(token)
            if not normalized:
                continue
            length = len(normalized)
            if length < min_len or length > max_len:
                continue
            bucket = buckets.setdefault(length, set())
            bucket.add(normalized)

    words: list[str] = []
    for length in sorted(buckets.keys()):
        candidates = sorted(buckets[length])
        if max_per_length is not None:
            candidates = candidates[:max_per_length]
        words.extend(candidates)
    return words
