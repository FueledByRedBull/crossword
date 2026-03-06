from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass


@dataclass
class BenchmarkResult:
    seed: str
    solver_backend: str
    puzzle_status: str
    pipeline_errors: list[str]
    candidate_count: int
    selected_k: int
    term_count: int
    clue_count: int
    leakage_count: int
    leakage_rate: float
    fill_status: str
    fill_percent: float
    filler_used_ratio: float
    clued_entry_ratio: float
    source_backed_entry_ratio: float
    used_source_backed_entry_ratio: float
    fallback_only_entry_count: int
    fallback_only_entry_ratio: float
    used_template_fallback_entry_ratio: float
    long_slot_theme_ratio: float
    quality_objective: float
    provenance_missing_count: int
    used_clue_provenance_missing_count: int
    synthetic_filler_clue_count: int
    packaged_synthetic_filler_count: int
    preferred_fill_target: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class BenchmarkAggregate:
    seed_count: int
    backend_counts: dict[str, int]
    fill_status_counts: dict[str, int]
    puzzle_status_counts: dict[str, int]
    average_fill_percent: float
    average_filler_used_ratio: float
    average_long_slot_theme_ratio: float
    average_clued_entry_ratio: float
    average_source_backed_entry_ratio: float
    average_fallback_only_entry_ratio: float
    average_used_source_backed_entry_ratio: float
    average_used_template_fallback_entry_ratio: float
    average_synthetic_filler_clue_count: float
    average_packaged_synthetic_filler_count: float
    average_used_clue_provenance_missing_count: float
    average_leakage_rate: float
    fill_pass_rate: float
    puzzle_ok_rate: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def summarize_benchmark(payload: dict) -> BenchmarkResult:
    return BenchmarkResult(
        seed=payload.get("seed", ""),
        solver_backend=payload.get("solver_backend", "python"),
        puzzle_status=payload.get("puzzle_status", "unknown"),
        pipeline_errors=payload.get("pipeline_errors", []),
        candidate_count=payload.get("candidate_count", 0),
        selected_k=payload.get("selected_k", 0),
        term_count=payload.get("term_count", 0),
        clue_count=payload.get("clue_count", 0),
        leakage_count=payload.get("leakage_count", 0),
        leakage_rate=payload.get("leakage_rate", 0.0),
        fill_status=payload.get("fill_status", "failed"),
        fill_percent=payload.get("fill_percent", 0.0),
        filler_used_ratio=payload.get("filler_used_ratio", 0.0),
        clued_entry_ratio=payload.get("clued_entry_ratio", 0.0),
        source_backed_entry_ratio=payload.get("source_backed_entry_ratio", 0.0),
        used_source_backed_entry_ratio=payload.get(
            "used_source_backed_entry_ratio",
            payload.get("source_backed_entry_ratio", 0.0),
        ),
        fallback_only_entry_count=payload.get("fallback_only_entry_count", 0),
        fallback_only_entry_ratio=payload.get("fallback_only_entry_ratio", 0.0),
        used_template_fallback_entry_ratio=payload.get(
            "used_template_fallback_entry_ratio",
            payload.get("fallback_only_entry_ratio", 0.0),
        ),
        long_slot_theme_ratio=payload.get("long_slot_theme_ratio", 0.0),
        quality_objective=payload.get("quality_objective", 0.0),
        provenance_missing_count=payload.get("provenance_missing_count", 0),
        used_clue_provenance_missing_count=payload.get(
            "used_clue_provenance_missing_count",
            payload.get("provenance_missing_count", 0),
        ),
        synthetic_filler_clue_count=payload.get("synthetic_filler_clue_count", 0),
        packaged_synthetic_filler_count=payload.get(
            "packaged_synthetic_filler_count",
            payload.get("synthetic_filler_clue_count", 0),
        ),
        preferred_fill_target=payload.get("preferred_fill_target", 0.85),
    )


def summarize_benchmark_collection(results: list[dict] | list[BenchmarkResult]) -> BenchmarkAggregate:
    normalized = [
        result if isinstance(result, BenchmarkResult) else summarize_benchmark(result)
        for result in results
    ]
    if not normalized:
        return BenchmarkAggregate(
            seed_count=0,
            backend_counts={},
            fill_status_counts={},
            puzzle_status_counts={},
            average_fill_percent=0.0,
            average_filler_used_ratio=0.0,
            average_long_slot_theme_ratio=0.0,
            average_clued_entry_ratio=0.0,
            average_source_backed_entry_ratio=0.0,
            average_fallback_only_entry_ratio=0.0,
            average_used_source_backed_entry_ratio=0.0,
            average_used_template_fallback_entry_ratio=0.0,
            average_synthetic_filler_clue_count=0.0,
            average_packaged_synthetic_filler_count=0.0,
            average_used_clue_provenance_missing_count=0.0,
            average_leakage_rate=0.0,
            fill_pass_rate=0.0,
            puzzle_ok_rate=0.0,
        )

    count = len(normalized)
    backend_counts = Counter(result.solver_backend for result in normalized)
    fill_status_counts = Counter(result.fill_status for result in normalized)
    puzzle_status_counts = Counter(result.puzzle_status for result in normalized)
    fill_pass_count = sum(1 for result in normalized if result.fill_status in {"partial", "complete"})
    puzzle_ok_count = sum(1 for result in normalized if result.puzzle_status == "ok")

    return BenchmarkAggregate(
        seed_count=count,
        backend_counts=dict(backend_counts),
        fill_status_counts=dict(fill_status_counts),
        puzzle_status_counts=dict(puzzle_status_counts),
        average_fill_percent=sum(result.fill_percent for result in normalized) / count,
        average_filler_used_ratio=sum(result.filler_used_ratio for result in normalized) / count,
        average_long_slot_theme_ratio=sum(result.long_slot_theme_ratio for result in normalized) / count,
        average_clued_entry_ratio=sum(result.clued_entry_ratio for result in normalized) / count,
        average_source_backed_entry_ratio=sum(result.source_backed_entry_ratio for result in normalized) / count,
        average_fallback_only_entry_ratio=sum(result.fallback_only_entry_ratio for result in normalized) / count,
        average_used_source_backed_entry_ratio=sum(
            result.used_source_backed_entry_ratio for result in normalized
        ) / count,
        average_used_template_fallback_entry_ratio=sum(
            result.used_template_fallback_entry_ratio for result in normalized
        ) / count,
        average_synthetic_filler_clue_count=sum(result.synthetic_filler_clue_count for result in normalized) / count,
        average_packaged_synthetic_filler_count=sum(
            result.packaged_synthetic_filler_count for result in normalized
        ) / count,
        average_used_clue_provenance_missing_count=sum(
            result.used_clue_provenance_missing_count for result in normalized
        ) / count,
        average_leakage_rate=sum(result.leakage_rate for result in normalized) / count,
        fill_pass_rate=fill_pass_count / count,
        puzzle_ok_rate=puzzle_ok_count / count,
    )
