from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
from typing import Any, Callable

from .lexicon import load_lexicon_scores, load_word_list
from .provenance import has_packageable_clue_provenance, infer_clue_class

QUALITY_GATE_MIN_FILL_PERCENT = 0.70
QUALITY_GATE_MAX_FILLER_RATIO = 0.25
QUALITY_GATE_MIN_CLUED_RATIO = 0.90


@dataclass(slots=True)
class SolverVocabulary:
    word_scores: dict[str, float]
    words: list[str]
    themed_words: list[str]
    themed_set: set[str]
    clue_answers: set[str]
    source_backed_answers: set[str]
    template_fallback_answers: set[str]
    fallback_only_answers: set[str]
    unsupported_answers: set[str]
    clue_answers_available: bool
    filler_words: list[str]
    filler_raw_count: int
    filler_added: int
    filler_limit_per_length: int
    filler_weight: float
    long_filler_weight: float
    effective_filler_max_len: int
    min_word_len: int | None
    max_word_len: int | None


def safe_float(value: Any, *, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_solver_token(token: str, *, lang: str) -> str:
    text = (token or "").strip().upper()
    if not text:
        return ""
    if lang == "en":
        return "".join(ch for ch in text if "A" <= ch <= "Z")
    return "".join(ch for ch in text if ch.isalpha())


def build_solver_vocabulary(
    *,
    terms_path: str | Path,
    lang: str,
    filler_path: str | Path | None,
    filler_min_len: int,
    filler_max_len: int,
    filler_max_per_length: int,
    filler_weight: float,
    common_lexicon_path: str | Path | None = "data/lexicon/combined_wordfreq.txt",
) -> SolverVocabulary:
    word_scores: dict[str, float] = {}
    with Path(terms_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            normalized = normalize_solver_token(row.get("normalized_answer") or "", lang=lang)
            if not normalized:
                continue
            answer_score = safe_float(row.get("answer_score"), default=0.0)
            lexicon_score = safe_float(row.get("lexicon_score"), default=0.0)
            crossword_score = safe_float(row.get("crosswordability_score"), default=0.0)
            composite = (0.6 * answer_score) + (0.25 * lexicon_score) + (0.15 * crossword_score)
            previous = word_scores.get(normalized)
            if previous is None or composite > previous:
                word_scores[normalized] = composite

    clue_answers_path = Path(terms_path).with_name("clues.csv")
    clue_answers: set[str] = set()
    source_backed_answers: set[str] = set()
    template_fallback_answers: set[str] = set()
    fallback_only_answers: set[str] = set()
    if clue_answers_path.exists():
        with clue_answers_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                answer = normalize_solver_token(row.get("normalized_answer") or "", lang=lang)
                if answer:
                    clue_class = infer_clue_class(row)
                    if not has_packageable_clue_provenance(row):
                        continue
                    clue_answers.add(answer)
                    if clue_class == "source_backed":
                        source_backed_answers.add(answer)
                    elif clue_class == "template_fallback":
                        template_fallback_answers.add(answer)
                        fallback_only_answers.add(answer)
    clue_answers &= set(word_scores)
    source_backed_answers &= clue_answers
    template_fallback_answers &= clue_answers
    fallback_only_answers &= clue_answers
    fallback_only_answers -= source_backed_answers
    template_fallback_answers -= source_backed_answers
    clue_answers_available = bool(clue_answers)
    if clue_answers_available:
        clue_answer_lengths = {len(answer) for answer in clue_answers}
        word_scores = {
            word: score
            for word, score in word_scores.items()
            if word in clue_answers or len(word) not in clue_answer_lengths
        }

    themed_words = sorted(word_scores, key=lambda word: (word_scores[word], word), reverse=True)
    themed_set = set(word_scores.keys())
    unsupported_answers = themed_set - clue_answers

    common_lexicon = load_lexicon_scores(common_lexicon_path)
    enforce_lexicon_cutoff = True
    if filler_path is not None:
        try:
            default_filler_path = Path("data/lexicon/filler_words.txt").resolve()
            enforce_lexicon_cutoff = Path(filler_path).resolve() == default_filler_path
        except Exception:
            enforce_lexicon_cutoff = True

    def is_strict_filler_candidate(word: str) -> bool:
        if not word:
            return False
        if lang == "en":
            if not word.isascii():
                return False
            if any(ch < "A" or ch > "Z" for ch in word):
                return False
            if len(set(word)) <= 1:
                return False
            vowel_count = sum(ch in "AEIOUY" for ch in word)
            if vowel_count == 0:
                return False
            if len(word) <= 4 and vowel_count <= 1:
                return False
        if enforce_lexicon_cutoff and common_lexicon and common_lexicon.get(word, 0.0) < 0.12:
            return False
        return True

    effective_filler_max_len = filler_max_len
    if clue_answers_available:
        effective_filler_max_len = min(effective_filler_max_len, 5)

    filler_raw_words = load_word_list(
        filler_path,
        min_len=filler_min_len,
        max_len=effective_filler_max_len,
        max_per_length=filler_max_per_length,
    )
    filler_raw_count = len(filler_raw_words)
    filler_limit_per_length = max(0, int(filler_max_per_length))
    filler_buckets: dict[int, list[str]] = {}
    for word in filler_raw_words:
        if word in word_scores or not is_strict_filler_candidate(word):
            continue
        filler_buckets.setdefault(len(word), []).append(word)
    filler_words: list[str] = []
    for length in sorted(filler_buckets):
        unique = sorted(
            set(filler_buckets[length]),
            key=lambda token: (common_lexicon.get(token, 0.0), token),
            reverse=True,
        )
        filler_words.extend(unique[:filler_limit_per_length])

    normalized_filler_weight = max(0.0, float(filler_weight))
    long_filler_weight = normalized_filler_weight * 0.25
    filler_added = 0
    for word in filler_words:
        if word in word_scores:
            continue
        word_scores[word] = long_filler_weight if len(word) >= 6 else normalized_filler_weight
        filler_added += 1

    words = sorted(word_scores, key=lambda word: (word_scores[word], word), reverse=True)
    min_word_len = min((len(word) for word in words), default=None)
    max_word_len = max((len(word) for word in words), default=None)

    return SolverVocabulary(
        word_scores=word_scores,
        words=words,
        themed_words=themed_words,
        themed_set=themed_set,
        clue_answers=clue_answers,
        source_backed_answers=source_backed_answers,
        template_fallback_answers=template_fallback_answers,
        fallback_only_answers=fallback_only_answers,
        unsupported_answers=unsupported_answers,
        clue_answers_available=clue_answers_available,
        filler_words=filler_words,
        filler_raw_count=filler_raw_count,
        filler_added=filler_added,
        filler_limit_per_length=filler_limit_per_length,
        filler_weight=normalized_filler_weight,
        long_filler_weight=long_filler_weight,
        effective_filler_max_len=effective_filler_max_len,
        min_word_len=min_word_len,
        max_word_len=max_word_len,
    )


def imply_assignments(
    *,
    rendered_grid: list[list[str]],
    slots_to_check: list[Any],
    assignments_in: dict[int, str],
    word_scores: dict[str, float],
) -> tuple[dict[int, str], list[dict[str, Any]]]:
    implied = dict(assignments_in)
    used_words = set(implied.values())
    invalid_slots: list[dict[str, Any]] = []
    for slot in slots_to_check:
        if slot.id in implied:
            continue
        letters: list[str] = []
        for r, c in slot.cells:
            cell = rendered_grid[r][c]
            if cell in (".", "#"):
                letters = []
                break
            letters.append(cell)
        if not letters:
            continue
        word = "".join(letters)
        if word in word_scores and word not in used_words:
            implied[slot.id] = word
            used_words.add(word)
        else:
            invalid_slots.append(
                {
                    "slot_id": slot.id,
                    "direction": slot.direction,
                    "length": slot.length,
                    "position": slot.cells[0],
                    "word": word,
                    "reason": "not_in_pool" if word not in word_scores else "duplicate",
                }
            )
    return implied, invalid_slots


def prune_conflicting_assignments(
    *,
    grid: list[list[str]],
    slots: list[Any],
    assignments_in: dict[int, str],
    word_scores: dict[str, float],
    themed_set: set[str],
    clue_answers: set[str],
    clue_answers_available: bool,
    fallback_only_answers: set[str] | None = None,
    render_grid_fn: Callable[[list[list[str]], list[Any], dict[int, str]], list[list[str]]],
) -> tuple[dict[int, str], list[dict[str, Any]], list[list[str]], list[dict[str, Any]]]:
    current_assignments = dict(assignments_in)
    slot_by_id = {slot.id: slot for slot in slots}
    fallback_only_answers = fallback_only_answers or set()
    removed_assignments: list[dict[str, Any]] = []
    max_iterations = max(1, len(slots))

    for _ in range(max_iterations):
        rendered = render_grid_fn(grid, slots, current_assignments)
        implied, invalid_slots = imply_assignments(
            rendered_grid=rendered,
            slots_to_check=slots,
            assignments_in=current_assignments,
            word_scores=word_scores,
        )
        current_assignments = implied
        if not invalid_slots:
            return current_assignments, invalid_slots, rendered, removed_assignments

        participation: dict[int, int] = {}
        for invalid_slot in invalid_slots:
            slot = slot_by_id.get(invalid_slot["slot_id"])
            if slot is None:
                continue
            invalid_cells = set(slot.cells)
            for assigned_slot_id, word in current_assignments.items():
                assigned_slot = slot_by_id.get(assigned_slot_id)
                if assigned_slot is None:
                    continue
                if any(cell in invalid_cells for cell in assigned_slot.cells):
                    participation[assigned_slot_id] = participation.get(assigned_slot_id, 0) + 1

        if not participation:
            return current_assignments, invalid_slots, rendered, removed_assignments

        def removal_rank(slot_id: int) -> tuple[int, int, int, int, float, int]:
            word = current_assignments[slot_id]
            slot = slot_by_id[slot_id]
            long_non_theme = 1 if slot.length >= 6 and word not in themed_set else 0
            unclued = 1 if clue_answers_available and word not in clue_answers else 0
            fallback_only = 1 if word in fallback_only_answers else 0
            filler = 1 if word not in themed_set else 0
            involvement = participation.get(slot_id, 0)
            return (
                long_non_theme,
                unclued,
                fallback_only,
                filler,
                involvement,
                -word_scores.get(word, 0.0),
                slot.length,
            )

        slot_to_remove = max(participation, key=removal_rank)
        removed_assignments.append(
            {
                "slot_id": slot_to_remove,
                "word": current_assignments[slot_to_remove],
                "reasons": {
                    "involved_invalid_slots": participation[slot_to_remove],
                    "is_filler": current_assignments[slot_to_remove] not in themed_set,
                    "is_clued": current_assignments[slot_to_remove] in clue_answers,
                    "is_fallback_only": current_assignments[slot_to_remove] in fallback_only_answers,
                },
            }
        )
        current_assignments.pop(slot_to_remove, None)

    rendered = render_grid_fn(grid, slots, current_assignments)
    final_assignments, invalid_slots = imply_assignments(
        rendered_grid=rendered,
        slots_to_check=slots,
        assignments_in=current_assignments,
        word_scores=word_scores,
    )
    return final_assignments, invalid_slots, rendered, removed_assignments


def run_template_trial(
    *,
    template: Any,
    trial_seed: int,
    solver: Callable[..., dict[str, Any]],
    build_grid_fn: Callable[[Any], list[list[str]]],
    auto_block_long_slots_fn: Callable[..., dict[str, Any]],
    build_slots_fn: Callable[..., list[Any]],
    render_grid_fn: Callable[[list[list[str]], list[Any], dict[int, str]], list[list[str]]],
    words: list[str],
    themed_words: list[str],
    themed_set: set[str],
    clue_answers: set[str],
    source_backed_answers: set[str],
    fallback_only_answers: set[str],
    clue_answers_available: bool,
    word_scores: dict[str, float],
    long_filler_weight: float,
    max_word_len: int | None,
    effective_min_slot_len: int,
    min_domain: int,
    max_steps: int,
    max_restarts: int,
    use_ac3: bool,
    beam_width: int,
    enable_local_repair: bool,
    repair_steps: int,
) -> dict[str, Any]:
    grid = build_grid_fn(template)
    auto_block = auto_block_long_slots_fn(
        grid,
        max_slot_len=max_word_len,
        min_slot_len=effective_min_slot_len,
        symmetric=True,
    )
    grid = auto_block["grid"]
    slots = build_slots_fn(grid, min_len=effective_min_slot_len)
    domains = {slot.id: [word for word in words if len(word) == slot.length] for slot in slots}
    active_slots = [slot for slot in slots if len(domains[slot.id]) >= min_domain]
    slot_by_id = {slot.id: slot for slot in slots}
    long_slot_ids = {slot.id for slot in slots if slot.length >= 6}
    trial_errors: list[str] = []
    phase_a_result: dict[str, Any] = {"solved": False, "assignments": {}, "steps": 0, "restarts": 0}
    preferred_long_words: set[str] = set()
    source_backed_length_counts: dict[int, int] = {}
    active_slot_length_counts: dict[int, int] = {}
    for word in source_backed_answers:
        source_backed_length_counts[len(word)] = source_backed_length_counts.get(len(word), 0) + 1
    for slot in active_slots:
        active_slot_length_counts[slot.length] = active_slot_length_counts.get(slot.length, 0) + 1

    if not active_slots:
        trial_errors.append("no_active_slots")
        result: dict[str, Any] = {"solved": False, "assignments": {}, "steps": 0, "restarts": 0}
    else:
        phase_a_steps = max(2000, max_steps // 4)
        phase_a_restarts = max(1, max_restarts // 3)
        phase_a_words = sorted(
            themed_words,
            key=lambda word: (
                1 if word in source_backed_answers else 0,
                word_scores.get(word, 0.0),
                word,
            ),
            reverse=True,
        )
        phase_a_result = solver(
            grid,
            active_slots,
            phase_a_words,
            min_len=effective_min_slot_len,
            max_steps=phase_a_steps,
            max_restarts=phase_a_restarts,
            random_seed=trial_seed + 37,
            use_ac3=use_ac3,
            word_scores=word_scores,
            beam_width=max(16, beam_width // 2),
            enable_local_repair=False,
            repair_steps=0,
        )
        for slot_id, word in phase_a_result.get("assignments", {}).items():
            slot = slot_by_id.get(slot_id)
            if slot is not None and slot.length >= 6 and word in themed_set:
                preferred_long_words.add(word)

        phase_b_scores = dict(word_scores)
        for word in words:
            if word not in themed_set and len(word) >= 6:
                phase_b_scores[word] = min(phase_b_scores[word], long_filler_weight)
            if word in source_backed_answers:
                phase_b_scores[word] = phase_b_scores.get(word, 0.0) + (0.12 if len(word) >= 6 else 0.05)
            elif word in clue_answers:
                phase_b_scores[word] = phase_b_scores.get(word, 0.0) + 0.02
            if (
                word in fallback_only_answers
                and source_backed_length_counts.get(len(word), 0) >= active_slot_length_counts.get(len(word), 0)
            ):
                penalty_factor = 0.90 if len(word) >= 6 else 0.96
                phase_b_scores[word] = phase_b_scores.get(word, 0.0) * penalty_factor
        for word in preferred_long_words:
            phase_b_scores[word] = phase_b_scores.get(word, 0.0) + 0.35

        result = solver(
            grid,
            active_slots,
            words,
            min_len=effective_min_slot_len,
            max_steps=max_steps,
            max_restarts=max_restarts,
            random_seed=trial_seed,
            use_ac3=use_ac3,
            word_scores=phase_b_scores,
            beam_width=beam_width,
            enable_local_repair=enable_local_repair,
            repair_steps=repair_steps,
        )

    raw_assignments = result.get("assignments", {})
    solver_assignments = {
        int(slot_id): str(word)
        for slot_id, word in raw_assignments.items()
    }
    final_assignments, invalid_slots, rendered, removed_assignments = prune_conflicting_assignments(
        grid=grid,
        slots=slots,
        assignments_in=solver_assignments,
        word_scores=word_scores,
        themed_set=themed_set,
        clue_answers=clue_answers,
        clue_answers_available=clue_answers_available,
        fallback_only_answers=fallback_only_answers,
        render_grid_fn=render_grid_fn,
    )
    unclued_removed_count = 0
    if clue_answers_available:
        unclued_removed_count = sum(
            1
            for word in final_assignments.values()
            if word in themed_set and word not in clue_answers
        )
    fill_count = sum(1 for row in rendered for cell in row if cell not in (".", "#"))
    total_cells = sum(1 for row in rendered for cell in row if cell != "#")
    fill_percent = 0.0 if total_cells == 0 else fill_count / total_cells
    unfilled_slots: list[dict[str, Any]] = []
    assigned_ids = set(final_assignments.keys())
    for slot in slots:
        if slot.id in assigned_ids:
            continue
        unfilled_slots.append(
            {
                "slot_id": slot.id,
                "direction": slot.direction,
                "length": slot.length,
                "position": slot.cells[0],
            }
        )
    unfilled_short_slot_count = sum(1 for slot in slots if slot.id not in assigned_ids and slot.length <= 5)

    assigned_count = len(final_assignments)
    themed_assigned_count = sum(1 for word in final_assignments.values() if word in themed_set)
    if clue_answers_available:
        clued_assigned_count = assigned_count - unclued_removed_count
    else:
        clued_assigned_count = assigned_count
    filler_used_count = max(0, assigned_count - themed_assigned_count)
    filler_used_ratio = 0.0 if assigned_count == 0 else filler_used_count / assigned_count
    clued_entry_ratio = 0.0 if assigned_count == 0 else clued_assigned_count / assigned_count
    source_backed_entry_count = sum(
        1 for word in final_assignments.values() if word in source_backed_answers
    )
    source_backed_entry_ratio = (
        0.0 if assigned_count == 0 else source_backed_entry_count / assigned_count
    )
    fallback_only_assigned_count = sum(
        1 for word in final_assignments.values() if word in fallback_only_answers
    )
    fallback_only_entry_ratio = (
        0.0 if assigned_count == 0 else fallback_only_assigned_count / assigned_count
    )
    long_slot_assigned_count = sum(1 for slot_id in final_assignments if slot_id in long_slot_ids)
    long_slot_non_theme_count = sum(
        1
        for slot_id, word in final_assignments.items()
        if slot_id in long_slot_ids and word not in themed_set
    )
    fallback_only_long_count = sum(
        1
        for slot_id, word in final_assignments.items()
        if slot_id in long_slot_ids and word in fallback_only_answers
    )
    fallback_only_long_ratio = (
        0.0
        if long_slot_assigned_count == 0
        else fallback_only_long_count / long_slot_assigned_count
    )
    long_slot_theme_ratio = (
        1.0
        if long_slot_assigned_count == 0
        else (long_slot_assigned_count - long_slot_non_theme_count) / long_slot_assigned_count
    )

    quality_objective = (
        (1.35 * fill_percent)
        + (0.75 * (0.0 if assigned_count == 0 else themed_assigned_count / assigned_count))
        + (0.90 * source_backed_entry_ratio)
        + (0.25 * clued_entry_ratio)
        + (0.35 * long_slot_theme_ratio)
        - (1.10 * filler_used_ratio)
        - (0.30 * len(invalid_slots))
        - (0.70 * long_slot_non_theme_count)
        - (0.75 * fallback_only_entry_ratio)
        - (0.90 * fallback_only_long_ratio)
        - (0.10 * unfilled_short_slot_count)
    )

    solved_flag = bool(result.get("solved", False))
    solved_final = solved_flag or (len(final_assignments) == len(slots) and not invalid_slots)
    if solved_final and active_slots and len(final_assignments) == len(slots):
        fill_status = "complete"
    elif final_assignments:
        fill_status = "partial"
    else:
        fill_status = "failed"

    return {
        "template": template,
        "rendered": rendered,
        "result": {
            **result,
            "assignments": final_assignments,
            "solved": solved_flag,
        },
        "slots": slots,
        "active_slots": active_slots,
        "auto_block": auto_block,
        "fill_percent": fill_percent,
        "fill_status": fill_status,
        "fill_count": fill_count,
        "total_cells": total_cells,
        "unfilled_slots": unfilled_slots,
        "trial_errors": trial_errors,
        "invalid_slots": invalid_slots,
        "removed_assignments": removed_assignments,
        "implicit_added_count": max(0, len(final_assignments) - len(solver_assignments)),
        "solved_final": solved_final,
        "quality_objective": quality_objective,
        "themed_assigned_count": themed_assigned_count,
        "clued_assigned_count": clued_assigned_count,
        "assigned_count": assigned_count,
        "filler_used_count": filler_used_count,
        "filler_used_ratio": filler_used_ratio,
        "clued_entry_ratio": clued_entry_ratio,
        "source_backed_entry_count": source_backed_entry_count,
        "source_backed_entry_ratio": source_backed_entry_ratio,
        "fallback_only_assigned_count": fallback_only_assigned_count,
        "fallback_only_entry_ratio": fallback_only_entry_ratio,
        "long_slot_assigned_count": long_slot_assigned_count,
        "long_slot_non_theme_count": long_slot_non_theme_count,
        "fallback_only_long_count": fallback_only_long_count,
        "fallback_only_long_ratio": fallback_only_long_ratio,
        "long_slot_theme_ratio": long_slot_theme_ratio,
        "unfilled_short_slot_count": unfilled_short_slot_count,
        "unclued_removed_count": unclued_removed_count,
        "phase_a": {
            "steps": phase_a_result.get("steps", 0),
            "restarts": phase_a_result.get("restarts", 0),
            "assigned_count": len(phase_a_result.get("assignments", {})),
            "preferred_long_words_count": len(preferred_long_words),
        },
    }


def evaluate_quality_gate(
    *,
    fill_percent: float,
    invalid_slots: list[dict[str, Any]],
    filler_used_ratio: float,
    clued_entry_ratio: float,
    clue_answers_available: bool,
    long_slot_non_theme_count: int,
) -> tuple[bool, list[str]]:
    quality_gate_reasons: list[str] = []
    if fill_percent < QUALITY_GATE_MIN_FILL_PERCENT:
        quality_gate_reasons.append(
            f"fill_percent_below_min:{QUALITY_GATE_MIN_FILL_PERCENT:.2f}"
        )
    if invalid_slots:
        quality_gate_reasons.append("invalid_slots_present")
    if filler_used_ratio > QUALITY_GATE_MAX_FILLER_RATIO:
        quality_gate_reasons.append(
            f"filler_ratio_above_max:{QUALITY_GATE_MAX_FILLER_RATIO:.2f}"
        )
    if clue_answers_available and clued_entry_ratio < QUALITY_GATE_MIN_CLUED_RATIO:
        quality_gate_reasons.append(
            f"clued_ratio_below_min:{QUALITY_GATE_MIN_CLUED_RATIO:.2f}"
        )
    if long_slot_non_theme_count > 0:
        quality_gate_reasons.append("long_slots_using_filler")
    return len(quality_gate_reasons) == 0, quality_gate_reasons
