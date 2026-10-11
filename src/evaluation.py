from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

RELEASE_MIN_SEED_PASS_RATE = 0.80
MAX_AVERAGE_LEAKAGE_RATE = 0.05
MIN_CENTRAL_RECALL = 0.75


@dataclass
class BenchmarkResult:
    seed: str
    puzzle_status: str
    pipeline_errors: list[str]
    candidate_count: int
    selected_k: int
    term_count: int
    clue_count: int
    fill_status: str
    valid: bool
    word_count: int
    crossing_count: int
    mean_crossings_per_word: float
    two_plus_crossings_ratio: float
    checked_letter_ratio: float
    definition_clue_ratio: float
    blank_missing_count: int
    duplicate_clue_count: int
    leakage_count: int
    leakage_rate: float
    used_clue_provenance_missing_count: int
    gate_a_graded: bool = False
    central_recall: float = 0.0
    off_theme_count: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class BenchmarkAggregate:
    seed_count: int
    fill_status_counts: dict[str, int]
    puzzle_status_counts: dict[str, int]
    average_word_count: float
    average_crossing_count: float
    average_mean_crossings_per_word: float
    average_two_plus_crossings_ratio: float
    average_checked_letter_ratio: float
    average_definition_clue_ratio: float
    average_leakage_rate: float
    average_used_clue_provenance_missing_count: float
    average_central_recall: float
    off_theme_total: int
    blank_missing_count: int
    duplicate_clue_count: int
    valid_rate: float
    fill_pass_rate: float
    puzzle_ok_rate: float
    gate_a_pass: bool
    gate_b_pass: bool
    gate_c_pass: bool
    gate_d_pass: bool
    release_candidate_pass: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def summarize_benchmark(payload: dict) -> BenchmarkResult:
    return BenchmarkResult(
        seed=payload.get("seed", ""),
        puzzle_status=payload.get("puzzle_status", "unknown"),
        pipeline_errors=payload.get("pipeline_errors", []),
        candidate_count=payload.get("candidate_count", 0),
        selected_k=payload.get("selected_k", 0),
        term_count=payload.get("term_count", 0),
        clue_count=payload.get("clue_count", 0),
        fill_status=payload.get("fill_status", "failed"),
        valid=payload.get("valid", False),
        word_count=payload.get("word_count", 0),
        crossing_count=payload.get("crossing_count", 0),
        mean_crossings_per_word=payload.get("mean_crossings_per_word", 0.0),
        two_plus_crossings_ratio=payload.get("two_plus_crossings_ratio", 0.0),
        checked_letter_ratio=payload.get("checked_letter_ratio", 0.0),
        definition_clue_ratio=payload.get("definition_clue_ratio", 0.0),
        blank_missing_count=payload.get("blank_missing_count", 0),
        duplicate_clue_count=payload.get("duplicate_clue_count", 0),
        leakage_count=payload.get("leakage_count", 0),
        leakage_rate=payload.get("leakage_rate", 0.0),
        used_clue_provenance_missing_count=payload.get("used_clue_provenance_missing_count", 0),
        gate_a_graded=payload.get("gate_a_graded", False),
        central_recall=payload.get("central_recall", 0.0),
        off_theme_count=payload.get("off_theme_count", 0),
    )


def summarize_benchmark_collection(results: list[dict] | list[BenchmarkResult]) -> BenchmarkAggregate:
    normalized = [
        result if isinstance(result, BenchmarkResult) else summarize_benchmark(result)
        for result in results
    ]
    count = len(normalized)

    def average(field: str) -> float:
        return sum(getattr(result, field) for result in normalized) / count if count else 0.0

    def rate(flags: list[bool]) -> float:
        return sum(flags) / len(flags) if flags else 0.0

    laid_out = [result for result in normalized if result.word_count > 0]
    valid_rate = rate([result.valid for result in laid_out])
    fill_pass_rate = rate([result.fill_status == "complete" for result in normalized])
    puzzle_ok_rate = rate([result.puzzle_status == "ok" for result in normalized])
    blank_missing_count = sum(result.blank_missing_count for result in normalized)
    duplicate_clue_count = sum(result.duplicate_clue_count for result in normalized)
    average_leakage_rate = average("leakage_rate")
    provenance_missing = sum(result.used_clue_provenance_missing_count for result in normalized)
    off_theme_total = sum(result.off_theme_count for result in normalized)
    # Gates fail closed: a seed with no layout or no packaging diagnostics has nothing to grade.
    graded = count > 0 and all(
        result.word_count > 0 and result.puzzle_status != "unknown" for result in normalized
    )
    # An invalid layout must never pass, so B needs every laid-out seed valid.
    gate_b_pass = fill_pass_rate >= RELEASE_MIN_SEED_PASS_RATE and bool(laid_out) and valid_rate == 1.0
    gate_c_pass = (
        graded
        and average_leakage_rate <= MAX_AVERAGE_LEAKAGE_RATE
        and blank_missing_count == 0
        and duplicate_clue_count == 0
    )
    gate_d_pass = graded and provenance_missing == 0
    # A needs a fixture and a pool for every seed, each seed above the recall bar, and no off-theme page.
    gate_a_pass = (
        count > 0
        and all(result.gate_a_graded and result.central_recall >= MIN_CENTRAL_RECALL for result in normalized)
        and off_theme_total == 0
    )

    return BenchmarkAggregate(
        seed_count=count,
        fill_status_counts=dict(Counter(result.fill_status for result in normalized)),
        puzzle_status_counts=dict(Counter(result.puzzle_status for result in normalized)),
        average_word_count=average("word_count"),
        average_crossing_count=average("crossing_count"),
        average_mean_crossings_per_word=average("mean_crossings_per_word"),
        average_two_plus_crossings_ratio=average("two_plus_crossings_ratio"),
        average_checked_letter_ratio=average("checked_letter_ratio"),
        average_definition_clue_ratio=average("definition_clue_ratio"),
        average_leakage_rate=average_leakage_rate,
        average_used_clue_provenance_missing_count=average("used_clue_provenance_missing_count"),
        average_central_recall=average("central_recall"),
        off_theme_total=off_theme_total,
        blank_missing_count=blank_missing_count,
        duplicate_clue_count=duplicate_clue_count,
        valid_rate=valid_rate,
        fill_pass_rate=fill_pass_rate,
        puzzle_ok_rate=puzzle_ok_rate,
        gate_a_pass=gate_a_pass,
        gate_b_pass=gate_b_pass,
        gate_c_pass=gate_c_pass,
        gate_d_pass=gate_d_pass,
        release_candidate_pass=(
            gate_a_pass
            and gate_b_pass
            and gate_c_pass
            and gate_d_pass
            and puzzle_ok_rate >= RELEASE_MIN_SEED_PASS_RATE
        ),
    )
