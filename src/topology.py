from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class GridTemplate:
    name: str
    size: int
    blocks: set[tuple[int, int]]


def make_symmetric_blocks(size: int, blocks: Iterable[tuple[int, int]]) -> set[tuple[int, int]]:
    all_blocks: set[tuple[int, int]] = set()
    for r, c in blocks:
        all_blocks.add((r, c))
        all_blocks.add((size - 1 - r, size - 1 - c))
    return all_blocks


def _grid_blocks(size: int, gap: int = 4) -> set[tuple[int, int]]:
    """Place blacks on every gap-th row AND every gap-th column.

    This hard-caps slot length at (gap - 1) in both directions.
    With gap=5 on a 13-grid:
      Black rows: 0, 5, 10  → across runs of length 4 (rows 1-4, 6-9, 11-12→pruned)
      Black cols: 0, 5, 10  → down runs of length 4 (same)
    No slot is ever longer than gap - 1 = 4 letters, which stays within
    the min_len=4 / max_len=12 vocabulary window.
    """
    blocks: set[tuple[int, int]] = set()
    for i in range(0, size, gap):
        for j in range(size):
            blocks.add((i, j))  # full black row
            blocks.add((j, i))  # full black col
    return blocks


def get_templates(size: int) -> list[GridTemplate]:
    if size == 15:
        base_blocks = {
            (0, 3),
            (0, 11),
            (1, 5),
            (1, 9),
            (2, 7),
            (3, 1),
            (3, 13),
            (4, 4),
            (4, 10),
            (5, 7),
        }
        return [
            GridTemplate(name="open", size=size, blocks=set()),
            GridTemplate(name="symmetric_sparse", size=size, blocks=make_symmetric_blocks(size, base_blocks)),
            GridTemplate(name="dense", size=size, blocks=_grid_blocks(size, gap=5)),
        ]
    if size == 13:
        base_blocks = {
            (0, 3),
            (0, 9),
            (1, 5),
            (1, 7),
            (2, 6),
            (3, 1),
            (3, 11),
            (4, 4),
            (4, 8),
        }
        return [
            GridTemplate(name="open", size=size, blocks=set()),
            GridTemplate(name="symmetric_sparse", size=size, blocks=make_symmetric_blocks(size, base_blocks)),
            GridTemplate(name="dense", size=size, blocks=_grid_blocks(size, gap=8)),
        ]
    return [GridTemplate(name="open", size=size, blocks=set())]


def build_grid(template: GridTemplate) -> list[list[str]]:
    grid = [["." for _ in range(template.size)] for _ in range(template.size)]
    for r, c in template.blocks:
        grid[r][c] = "#"
    return grid


def extract_slots(grid: list[list[str]], min_len: int = 3) -> list[dict]:
    slots = []
    size = len(grid)
    slot_id = 0

    for r in range(size):
        c = 0
        while c < size:
            if grid[r][c] == "#":
                c += 1
                continue
            start = c
            while c < size and grid[r][c] != "#":
                c += 1
            length = c - start
            if length >= min_len:
                slots.append(
                    {
                        "id": slot_id,
                        "direction": "across",
                        "length": length,
                        "cells": [(r, cc) for cc in range(start, c)],
                    }
                )
                slot_id += 1
    for c in range(size):
        r = 0
        while r < size:
            if grid[r][c] == "#":
                r += 1
                continue
            start = r
            while r < size and grid[r][c] != "#":
                r += 1
            length = r - start
            if length >= min_len:
                slots.append(
                    {
                        "id": slot_id,
                        "direction": "down",
                        "length": length,
                        "cells": [(rr, c) for rr in range(start, r)],
                    }
                )
                slot_id += 1
    return slots


def auto_block_long_slots(
    grid: list[list[str]],
    *,
    max_slot_len: int | None,
    symmetric: bool = True,
    max_added_blocks: int = 64,
) -> dict:
    adjusted = [row[:] for row in grid]
    size = len(adjusted)

    if max_slot_len is None or max_slot_len <= 0:
        return {
            "grid": adjusted,
            "added_blocks": [],
            "long_slots_before": [],
            "long_slots_after": [],
            "iterations": 0,
        }

    long_slots_before = [
        slot for slot in extract_slots(adjusted, min_len=1) if slot["length"] > max_slot_len
    ]
    added_blocks: set[tuple[int, int]] = set()
    iterations = 0

    def try_place(cell: tuple[int, int]) -> bool:
        placements = [cell]
        if symmetric:
            mirror = (size - 1 - cell[0], size - 1 - cell[1])
            if mirror != cell:
                placements.append(mirror)
        changed = False
        for r, c in placements:
            if adjusted[r][c] == "#":
                continue
            adjusted[r][c] = "#"
            added_blocks.add((r, c))
            changed = True
        return changed

    while len(added_blocks) < max_added_blocks:
        long_slots = [
            slot for slot in extract_slots(adjusted, min_len=1) if slot["length"] > max_slot_len
        ]
        if not long_slots:
            break
        target = max(long_slots, key=lambda slot: slot["length"])
        if target["length"] <= 2:
            break
        midpoint = target["length"] // 2
        midpoint = max(1, min(target["length"] - 2, midpoint))
        if try_place(target["cells"][midpoint]):
            iterations += 1
            continue

        placed = False
        for idx in range(1, target["length"] - 1):
            if try_place(target["cells"][idx]):
                iterations += 1
                placed = True
                break
        if not placed:
            break

    long_slots_after = [
        slot for slot in extract_slots(adjusted, min_len=1) if slot["length"] > max_slot_len
    ]
    return {
        "grid": adjusted,
        "added_blocks": sorted(added_blocks),
        "long_slots_before": long_slots_before,
        "long_slots_after": long_slots_after,
        "iterations": iterations,
    }


TOPOLOGY_SCORE_WEIGHTS_DEFAULT: dict[str, float] = {
    "length_fit": 0.4,
    "anchor": 0.2,
    "short_fill": 0.2,
    "crossing": 0.2,
}


def score_template(
    words: list[str],
    template: GridTemplate,
    min_len: int = 3,
    *,
    weights: dict[str, float] | None = None,
) -> dict:
    w = {**TOPOLOGY_SCORE_WEIGHTS_DEFAULT, **(weights or {})}
    grid = build_grid(template)
    slots = extract_slots(grid, min_len=min_len)

    slot_lengths: dict[int, int] = {}
    for slot in slots:
        slot_lengths[slot["length"]] = slot_lengths.get(slot["length"], 0) + 1

    word_lengths: dict[int, int] = {}
    for word in words:
        word_lengths[len(word)] = word_lengths.get(len(word), 0) + 1

    total_slots = sum(slot_lengths.values()) or 1
    length_fit = sum(
        min(word_lengths.get(length, 0), count) for length, count in slot_lengths.items()
    ) / total_slots

    long_slots = sum(count for length, count in slot_lengths.items() if length >= 9)
    long_words = sum(count for length, count in word_lengths.items() if length >= 9)
    anchor_score = min(1.0, long_words / long_slots) if long_slots else 1.0

    short_slots = sum(count for length, count in slot_lengths.items() if 4 <= length <= 5)
    short_words = sum(count for length, count in word_lengths.items() if 4 <= length <= 5)
    short_score = min(1.0, short_words / short_slots) if short_slots else 1.0

    # Crossing potential: intersections per slot normalized by slot count.
    cell_map: dict[tuple[int, int], int] = {}
    for slot in slots:
        for cell in slot["cells"]:
            cell_map[cell] = cell_map.get(cell, 0) + 1
    intersections = sum(1 for count in cell_map.values() if count > 1)
    crossing_score = min(1.0, intersections / max(1, len(slots)))

    score = (
        w["length_fit"] * length_fit
        + w["anchor"] * anchor_score
        + w["short_fill"] * short_score
        + w["crossing"] * crossing_score
    )

    return {
        "template": template.name,
        "score": score,
        "length_fit": length_fit,
        "anchor_score": anchor_score,
        "short_score": short_score,
        "crossing_score": crossing_score,
        "slot_count": len(slots),
        "weights": w,
    }


def score_template_from_length_hist(
    length_hist: dict[int, int],
    template: GridTemplate,
    min_len: int = 3,
    *,
    weights: dict[str, float] | None = None,
    max_word_len: int | None = None,
    auto_block_long_slots_enabled: bool = True,
) -> dict:
    w = {**TOPOLOGY_SCORE_WEIGHTS_DEFAULT, **(weights or {})}
    base_grid = build_grid(template)
    auto_block = {
        "grid": base_grid,
        "added_blocks": [],
        "long_slots_before": [],
        "long_slots_after": [],
        "iterations": 0,
    }
    if max_word_len is not None and auto_block_long_slots_enabled:
        auto_block = auto_block_long_slots(base_grid, max_slot_len=max_word_len, symmetric=True)
    grid = auto_block["grid"]
    slots = extract_slots(grid, min_len=min_len)

    slot_lengths: dict[int, int] = {}
    for slot in slots:
        slot_lengths[slot["length"]] = slot_lengths.get(slot["length"], 0) + 1

    total_slots = sum(slot_lengths.values()) or 1
    length_fit = sum(
        min(length_hist.get(length, 0), count) for length, count in slot_lengths.items()
    ) / total_slots

    long_slots = sum(count for length, count in slot_lengths.items() if length >= 9)
    long_words = sum(count for length, count in length_hist.items() if length >= 9)
    anchor_score = min(1.0, long_words / long_slots) if long_slots else 1.0

    short_slots = sum(count for length, count in slot_lengths.items() if 4 <= length <= 5)
    short_words = sum(count for length, count in length_hist.items() if 4 <= length <= 5)
    short_score = min(1.0, short_words / short_slots) if short_slots else 1.0

    cell_map: dict[tuple[int, int], int] = {}
    for slot in slots:
        for cell in slot["cells"]:
            cell_map[cell] = cell_map.get(cell, 0) + 1
    intersections = sum(1 for count in cell_map.values() if count > 1)
    crossing_score = min(1.0, intersections / max(1, len(slots)))

    # Conflict is the fraction of slots that cannot be uniquely supplied by
    # current length inventory (shortage, not just complete absence).
    shortage_slots = sum(
        max(0, count - length_hist.get(length, 0)) for length, count in slot_lengths.items()
    )
    fill_conflict = 0.0 if total_slots == 0 else shortage_slots / total_slots

    long_slot_penalty = 0.0
    long_slot_count_before = 0
    long_slot_count_after = 0
    if max_word_len is not None:
        if auto_block_long_slots_enabled:
            long_slot_count_before = len(auto_block["long_slots_before"])
            long_slot_count_after = len(auto_block["long_slots_after"])
            overflow_letters = sum(
                max(0, slot["length"] - max_word_len) for slot in auto_block["long_slots_before"]
            )
            total_letters = sum(
                slot["length"] for slot in extract_slots(base_grid, min_len=1)
            ) or 1
        else:
            long_slots = [slot for slot in slots if slot["length"] > max_word_len]
            long_slot_count_before = len(long_slots)
            long_slot_count_after = len(long_slots)
            overflow_letters = sum(max(0, slot["length"] - max_word_len) for slot in long_slots)
            total_letters = sum(slot["length"] for slot in slots) or 1
        long_slot_penalty = overflow_letters / total_letters

    auto_block_density = 0.0
    if auto_block["added_blocks"]:
        auto_block_density = len(auto_block["added_blocks"]) / max(1, template.size * template.size)

    score = (
        w["length_fit"] * length_fit
        + w["anchor"] * anchor_score
        + w["short_fill"] * short_score
        + w["crossing"] * crossing_score
    )

    return {
        "template": template.name,
        "score": score,
        "length_fit": length_fit,
        "anchor_score": anchor_score,
        "short_score": short_score,
        "crossing_score": crossing_score,
        "slot_count": len(slots),
        "fill_conflict": fill_conflict,
        "long_slot_penalty": long_slot_penalty,
        "long_slot_count_before": long_slot_count_before,
        "long_slot_count_after": long_slot_count_after,
        "max_word_len": max_word_len,
        "auto_block_added_count": len(auto_block["added_blocks"]),
        "auto_block_density": auto_block_density,
        "auto_block_iterations": auto_block["iterations"],
        "auto_block_added_blocks": auto_block["added_blocks"],
        "weights": w,
    }


def select_best_template(
    words: list[str],
    size: int,
    min_len: int = 3,
    *,
    weights: dict[str, float] | None = None,
    fill_conflict_weight: float = 0.5,
    long_slot_penalty_weight: float = 2.0,
    auto_block_penalty_weight: float = 0.2,
    max_word_len: int | None = None,
    auto_block_long_slots_enabled: bool = True,
) -> dict:
    templates = get_templates(size)
    length_hist: dict[int, int] = {}
    for word in words:
        length = len(word)
        length_hist[length] = length_hist.get(length, 0) + 1
    inferred_max_word_len = max(length_hist.keys(), default=None)
    if max_word_len is None:
        max_word_len = inferred_max_word_len

    scored = [
        score_template_from_length_hist(
            length_hist,
            template,
            min_len=min_len,
            weights=weights,
            max_word_len=max_word_len,
            auto_block_long_slots_enabled=auto_block_long_slots_enabled,
        )
        for template in templates
    ]
    for row in scored:
        row["selection_score"] = (
            row["score"]
            - (fill_conflict_weight * row["fill_conflict"])
            - (long_slot_penalty_weight * row["long_slot_penalty"])
            - (auto_block_penalty_weight * row["auto_block_density"])
        )
    scored.sort(key=lambda row: (row["selection_score"], row["score"]), reverse=True)
    return {"selected": scored[0]["template"], "scored": scored}
