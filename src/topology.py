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
) -> dict:
    w = {**TOPOLOGY_SCORE_WEIGHTS_DEFAULT, **(weights or {})}
    grid = build_grid(template)
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
        "weights": w,
    }


def select_best_template(
    words: list[str],
    size: int,
    min_len: int = 3,
    *,
    weights: dict[str, float] | None = None,
    fill_conflict_weight: float = 0.5,
) -> dict:
    templates = get_templates(size)
    length_hist: dict[int, int] = {}
    for word in words:
        length = len(word)
        length_hist[length] = length_hist.get(length, 0) + 1

    scored = [
        score_template_from_length_hist(length_hist, template, min_len=min_len, weights=weights)
        for template in templates
    ]
    for row in scored:
        row["selection_score"] = row["score"] - (fill_conflict_weight * row["fill_conflict"])
    scored.sort(key=lambda row: (row["selection_score"], row["score"]), reverse=True)
    return {"selected": scored[0]["template"], "scored": scored}
