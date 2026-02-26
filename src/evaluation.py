from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    seed: str
    pipeline_errors: list[str]
    candidate_count: int
    selected_k: int
    term_count: int
    clue_count: int
    leakage_count: int
    leakage_rate: float
    fill_status: str
    fill_percent: float
    provenance_missing_count: int


def summarize_benchmark(payload: dict) -> BenchmarkResult:
    return BenchmarkResult(
        seed=payload.get("seed", ""),
        pipeline_errors=payload.get("pipeline_errors", []),
        candidate_count=payload.get("candidate_count", 0),
        selected_k=payload.get("selected_k", 0),
        term_count=payload.get("term_count", 0),
        clue_count=payload.get("clue_count", 0),
        leakage_count=payload.get("leakage_count", 0),
        leakage_rate=payload.get("leakage_rate", 0.0),
        fill_status=payload.get("fill_status", "failed"),
        fill_percent=payload.get("fill_percent", 0.0),
        provenance_missing_count=payload.get("provenance_missing_count", 0),
    )

