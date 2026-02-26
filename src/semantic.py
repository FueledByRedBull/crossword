from __future__ import annotations

import math
from typing import Iterable

from .text_normalize import tokenize


def build_tfidf_vectors(texts: list[str], *, lang: str = "en") -> list[dict[str, float]]:
    tokenized = [tokenize(text, lang=lang) for text in texts]
    doc_count = len(tokenized)
    df: dict[str, int] = {}
    for tokens in tokenized:
        seen = set(tokens)
        for term in seen:
            df[term] = df.get(term, 0) + 1

    idf: dict[str, float] = {}
    for term, freq in df.items():
        idf[term] = math.log((doc_count + 1) / (freq + 1)) + 1.0

    vectors: list[dict[str, float]] = []
    for tokens in tokenized:
        if not tokens:
            vectors.append({})
            continue
        tf: dict[str, int] = {}
        for term in tokens:
            tf[term] = tf.get(term, 0) + 1
        length = len(tokens)
        vec: dict[str, float] = {}
        for term, count in tf.items():
            vec[term] = (count / length) * idf.get(term, 0.0)
        vectors.append(vec)
    return vectors


def cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    dot = 0.0
    for term, weight in vec_a.items():
        dot += weight * vec_b.get(term, 0.0)
    norm_a = math.sqrt(sum(value * value for value in vec_a.values()))
    norm_b = math.sqrt(sum(value * value for value in vec_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def mmr_rank(
    *,
    rel_scores: list[float],
    pairwise: list[list[float]],
    depth_penalties: list[float],
    backlink_bonus: list[float],
    a: float,
    b: float,
    c: float,
    d: float,
) -> list[dict[str, float]]:
    count = len(rel_scores)
    selected: list[int] = []
    remaining = set(range(count))
    results: list[dict[str, float]] = []

    while remaining:
        best_idx = None
        best_score = -1e9
        best_red = 0.0
        for idx in remaining:
            if not selected:
                red = 0.0
            else:
                red = max(pairwise[idx][sel] for sel in selected)
            score = (
                a * rel_scores[idx]
                - b * red
                - c * depth_penalties[idx]
                + d * backlink_bonus[idx]
            )
            if score > best_score:
                best_score = score
                best_idx = idx
                best_red = red
        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
        results.append(
            {
                "index": best_idx,
                "selection_score": best_score,
                "red_score": best_red,
            }
        )
    return results


def build_pairwise_matrix(vectors: list[dict[str, float]]) -> list[list[float]]:
    count = len(vectors)
    matrix = [[0.0 for _ in range(count)] for _ in range(count)]
    for i in range(count):
        matrix[i][i] = 1.0
        for j in range(i + 1, count):
            sim = cosine_similarity(vectors[i], vectors[j])
            matrix[i][j] = sim
            matrix[j][i] = sim
    return matrix
