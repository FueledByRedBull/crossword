from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KSelectionResult:
    selected_indices: list[int]
    selected_k: int
    trace: list[dict]


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def coherence_score(rel_scores: list[float], selected: list[int]) -> float:
    return _mean([rel_scores[idx] for idx in selected])


def diversity_score(pairwise: list[list[float]], selected: list[int]) -> float:
    if len(selected) < 2:
        return 0.0
    sims = []
    for i, idx in enumerate(selected):
        for j in range(i + 1, len(selected)):
            sims.append(pairwise[idx][selected[j]])
    if not sims:
        return 0.0
    mean_sim = _mean(sims)
    return max(0.0, 1.0 - mean_sim)


def term_quality_score(term_scores: list[float] | None, selected: list[int]) -> float:
    if not term_scores:
        return 0.0
    return _mean([term_scores[idx] for idx in selected])


def compute_jk(
    *,
    rel_scores: list[float],
    pairwise: list[list[float]],
    selected: list[int],
    weights: dict[str, float],
    term_scores: list[float] | None = None,
    template_fit: float = 0.0,
    fill_conflict: float = 0.0,
) -> dict[str, float]:
    coherence = coherence_score(rel_scores, selected)
    diversity = diversity_score(pairwise, selected)
    term_quality = term_quality_score(term_scores, selected)
    w1 = weights.get("w1", 1.0)
    w2 = weights.get("w2", 1.0)
    w3 = weights.get("w3", 1.0)
    w4 = weights.get("w4", 0.5)
    w5 = weights.get("w5", 0.5)
    jk = (w1 * coherence) + (w2 * diversity) + (w3 * term_quality) + (w4 * template_fit) - (
        w5 * fill_conflict
    )
    return {
        "jk": jk,
        "coherence": coherence,
        "diversity": diversity,
        "term_quality": term_quality,
        "template_fit": template_fit,
        "fill_conflict": fill_conflict,
    }


def select_k(
    *,
    ranked_indices: list[int],
    rel_scores: list[float],
    pairwise: list[list[float]],
    weights: dict[str, float] | None = None,
    min_k: int = 5,
    max_k: int | None = None,
    epsilon: float = 0.01,
    m: int = 2,
    term_scores: list[float] | None = None,
    template_fit_by_k: list[float] | None = None,
    fill_conflict_by_k: list[float] | None = None,
) -> KSelectionResult:
    if weights is None:
        weights = {"w1": 1.0, "w2": 1.0, "w3": 1.0, "w4": 0.5, "w5": 0.5}
    if not ranked_indices:
        return KSelectionResult(selected_indices=[], selected_k=0, trace=[])

    max_k = min(max_k or len(ranked_indices), len(ranked_indices))
    min_k = min(min_k, max_k)
    trace: list[dict] = []
    prev_jk = None
    below_eps = 0
    stop_k = max_k

    for k in range(1, max_k + 1):
        selected = ranked_indices[:k]
        template_fit = template_fit_by_k[k - 1] if template_fit_by_k and k - 1 < len(template_fit_by_k) else 0.0
        fill_conflict = (
            fill_conflict_by_k[k - 1] if fill_conflict_by_k and k - 1 < len(fill_conflict_by_k) else 0.0
        )
        metrics = compute_jk(
            rel_scores=rel_scores,
            pairwise=pairwise,
            selected=selected,
            weights=weights,
            term_scores=term_scores,
            template_fit=template_fit,
            fill_conflict=fill_conflict,
        )
        jk = metrics["jk"]
        delta = None if prev_jk is None else jk - prev_jk
        if k >= min_k and delta is not None:
            if delta < epsilon:
                below_eps += 1
            else:
                below_eps = 0
            if below_eps >= m:
                stop_k = k
                trace.append(
                    {
                        "k": k,
                        "jk": jk,
                        "delta": delta,
                        **metrics,
                        "stop": True,
                    }
                )
                break
        trace.append(
            {
                "k": k,
                "jk": jk,
                "delta": delta,
                **metrics,
                "stop": False,
            }
        )
        prev_jk = jk

    selected_indices = ranked_indices[:stop_k]
    return KSelectionResult(selected_indices=selected_indices, selected_k=len(selected_indices), trace=trace)
