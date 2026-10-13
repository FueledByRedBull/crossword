from __future__ import annotations

import os
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

EMPTY = "#"
ACROSS = "across"
DOWN = "down"

_BLANKS = (EMPTY, ".")
_MIN_WORD_LEN = 3

# The search is the layout-rs program: simulated annealing over ruin-and-recreate steps, `restarts`
# times this many of them (see layout-rs/src/main.rs). It runs about 15,000 steps a second, so the
# default 96 restarts (72,000 steps) take about 4 s on an 80-candidate list. More steps keep paying,
# slowly: mean over 21 candidate lists and 3 seeds at 96 restarts, value = words + 1.5 * crossings
# + 3 * the squared normalised scores of the placed answers,
#   steps per restart   25     250    750    2,500
#   words               21.5   22.2   22.4   22.5
#   crossings           26.3   28.3   29.3   29.7
#   value               78.7   82.8   84.1   85.3
# and on 8 of the lists going from 2,500 to 25,000 added about 0.4 words, 0.5 crossings and 1.8 value.
_STEPS_PER_RESTART = 750

_CRATE = Path(__file__).resolve().parents[1] / "layout-rs"
_BINARY = _CRATE / "target" / "release" / ("crossword-layout.exe" if os.name == "nt" else "crossword-layout")


@dataclass(frozen=True)
class Placement:
    answer: str
    row: int
    col: int
    direction: str


@dataclass
class Layout:
    grid: list[list[str]]
    placements: list[Placement]
    metrics: dict[str, float | int]


def _cells(placement: Placement) -> list[tuple[int, int]]:
    dr, dc = (0, 1) if placement.direction == ACROSS else (1, 0)
    return [
        (placement.row + dr * i, placement.col + dc * i)
        for i in range(len(placement.answer))
    ]


def _is_letter(grid: Sequence[Sequence[str]], r: int, c: int) -> bool:
    return (
        0 <= r < len(grid)
        and 0 <= c < len(grid[r])
        and grid[r][c] not in _BLANKS
    )


def _runs(grid: Sequence[Sequence[str]]) -> dict[tuple[int, int, int, str], str]:
    """Maximal runs of 2+ letter cells, keyed by (row, col, length, direction)."""
    runs: dict[tuple[int, int, int, str], str] = {}
    for r in range(len(grid)):
        for c in range(len(grid[r])):
            if not _is_letter(grid, r, c):
                continue
            for direction, dr, dc in ((ACROSS, 0, 1), (DOWN, 1, 0)):
                if _is_letter(grid, r - dr, c - dc):
                    continue
                n = 1
                while _is_letter(grid, r + dr * n, c + dc * n):
                    n += 1
                if n >= 2:
                    runs[(r, c, n, direction)] = "".join(
                        grid[r + dr * i][c + dc * i] for i in range(n)
                    )
    return runs


def validate_layout(
    grid: Sequence[Sequence[str]], placements: Sequence[Placement]
) -> list[str]:
    problems: list[str] = []
    covered: dict[tuple[int, int], list[int]] = {}
    sound: list[int] = []  # placements whose grid letters spell their answer
    for idx, p in enumerate(placements):
        label = f"{p.direction} '{p.answer}' at ({p.row},{p.col})"
        if p.direction not in (ACROSS, DOWN):
            problems.append(f"{label}: direction must be across or down")
            continue
        cells = _cells(p)
        if not all(0 <= r < len(grid) and 0 <= c < len(grid[r]) for r, c in cells):
            problems.append(f"{label}: out of bounds")
            continue
        spelled = "".join(grid[r][c] for r, c in cells)
        if spelled != p.answer:
            problems.append(f"{label}: grid spells '{spelled}'")
        else:
            sound.append(idx)
        for cell in cells:
            covered.setdefault(cell, []).append(idx)

    for answer, count in Counter(p.answer for p in placements).items():
        if count > 1:
            problems.append(f"answer '{answer}' appears {count} times")

    runs = _runs(grid)
    known = {(p.row, p.col, len(p.answer), p.direction) for p in placements}
    for key, text in runs.items():
        if key not in known:
            r, c, n, direction = key
            problems.append(
                f"letters '{text}' at ({r},{c}) form a non-word {direction} run of {n}"
            )
    for idx in sound:
        p = placements[idx]
        if (p.row, p.col, len(p.answer), p.direction) not in runs:
            problems.append(
                f"{p.direction} '{p.answer}' at ({p.row},{p.col}) touches other letters at an end"
            )

    for r in range(len(grid)):
        for c in range(len(grid[r])):
            if _is_letter(grid, r, c) and (r, c) not in covered:
                problems.append(f"letter '{grid[r][c]}' at ({r},{c}) belongs to no word")

    if len(placements) < 2:
        problems.append("layout needs at least 2 words")
        return problems

    neighbours: dict[int, set[int]] = {i: set() for i in range(len(placements))}
    for owners in covered.values():
        for a in owners:
            neighbours[a].update(o for o in owners if o != a)
    for idx, others in neighbours.items():
        if not others:
            p = placements[idx]
            problems.append(f"{p.direction} '{p.answer}' at ({p.row},{p.col}) crosses no other word")
    linked = [i for i, others in neighbours.items() if others]
    seen: set[int] = set()
    groups = 0
    for start in linked:
        if start in seen:
            continue
        groups += 1
        stack = [start]
        while stack:
            node = stack.pop()
            if node not in seen:
                seen.add(node)
                stack.extend(neighbours[node])
    if groups > 1:
        problems.append(f"words form {groups} disconnected groups")
    return problems


def validate_puzzle(puzzle: dict) -> list[str]:
    placements = [
        Placement(str(entry["answer"]), int(entry["row"]), int(entry["col"]), direction)
        for direction, key in ((ACROSS, "across_entries"), (DOWN, "down_entries"))
        for entry in puzzle.get(key, [])
    ]
    return validate_layout(puzzle["grid_cells"], placements)


def number_placements(placements: Sequence[Placement]) -> dict[tuple[int, int], int]:
    starts = sorted({(p.row, p.col) for p in placements})
    return {start: number for number, start in enumerate(starts, start=1)}


def layout_metrics(
    grid: Sequence[Sequence[str]], placements: Sequence[Placement]
) -> dict[str, float | int]:
    owners: dict[tuple[int, int], set[str]] = {}
    for p in placements:
        for cell in _cells(p):
            owners.setdefault(cell, set()).add(p.direction)
    crossing_cells = {cell for cell, dirs in owners.items() if len(dirs) == 2}
    per_word = [
        sum(cell in crossing_cells for cell in _cells(p)) for p in placements
    ]
    letters = [
        (r, c)
        for r in range(len(grid))
        for c in range(len(grid[r]))
        if _is_letter(grid, r, c)
    ]
    rows = [r for r, _ in letters]
    cols = [c for _, c in letters]
    bbox_rows = max(rows) - min(rows) + 1 if letters else 0
    bbox_cols = max(cols) - min(cols) + 1 if letters else 0
    word_count = len(placements)
    two_plus = sum(n >= 2 for n in per_word)
    return {
        "word_count": word_count,
        "across_count": sum(p.direction == ACROSS for p in placements),
        "down_count": sum(p.direction == DOWN for p in placements),
        "crossing_count": len(crossing_cells),
        "mean_crossings_per_word": round(sum(per_word) / word_count, 4) if word_count else 0.0,
        "words_with_two_plus_crossings": two_plus,
        "two_plus_crossings_ratio": round(two_plus / word_count, 4) if word_count else 0.0,
        "letter_count": len(letters),
        "checked_letter_ratio": round(len(crossing_cells) / len(letters), 4) if letters else 0.0,
        "bbox_rows": bbox_rows,
        "bbox_cols": bbox_cols,
        "density": round(len(letters) / (bbox_rows * bbox_cols), 4) if letters else 0.0,
    }


def _binary() -> Path:
    """The layout-rs program, built first when it is missing or older than its sources."""
    sources = [_CRATE / "Cargo.toml", *(_CRATE / "src").rglob("*.rs")]
    if _BINARY.exists() and _BINARY.stat().st_mtime >= max(path.stat().st_mtime for path in sources):
        return _BINARY
    command = ["cargo", "build", "--release", "--offline", "--manifest-path", str(_CRATE / "Cargo.toml")]
    try:
        build = subprocess.run(command, capture_output=True, encoding="utf-8", errors="replace")
    except FileNotFoundError:
        raise RuntimeError(
            "the layout search is a Rust program and needs cargo to build it: install a Rust toolchain from https://rustup.rs"
        ) from None
    if build.returncode:
        tail = "\n".join(build.stderr.strip().splitlines()[-15:])
        raise RuntimeError(f"building layout-rs failed ({' '.join(command)}):\n{tail}")
    return _BINARY


def _search(words: Sequence[str], scores: Sequence[float], size: int, seed: int, steps: int, limit: int) -> list[Placement]:
    """Run the layout-rs search on words already ordered best first; limit 0 means no cap."""
    lines = [f"{size} {seed % 2**64} {steps} {limit}"]
    for word, score in zip(words, scores):
        if "\t" in word or "\n" in word:
            raise ValueError(f"answer {word!r} contains a tab or newline")
        lines.append(f"{score!r}\t{word}")
    done = subprocess.run([str(_binary())], input="\n".join(lines).encode("utf-8") + b"\n", capture_output=True)
    if done.returncode:
        raise RuntimeError(f"layout-rs exited with status {done.returncode}: {done.stderr.decode('utf-8', 'replace').strip()}")
    placements = []
    for line in filter(None, done.stdout.decode("utf-8").split("\n")):
        answer, row, col, direction = line.rsplit("\t", 3)
        placements.append(Placement(answer, int(row), int(col), ACROSS if direction == "A" else DOWN))
    return placements


def _to_layout(raw: Sequence[Placement], size: int) -> Layout:
    cells = [cell for p in raw for cell in _cells(p)]
    row_off = col_off = 0
    if cells:
        min_r, max_r = min(r for r, _ in cells), max(r for r, _ in cells)
        min_c, max_c = min(c for _, c in cells), max(c for _, c in cells)
        row_off = (size - (max_r - min_r + 1)) // 2 - min_r
        col_off = (size - (max_c - min_c + 1)) // 2 - min_c
    grid = [[EMPTY] * size for _ in range(size)]
    placements = [Placement(p.answer, p.row + row_off, p.col + col_off, p.direction) for p in raw]
    for p in placements:
        for (r, c), ch in zip(_cells(p), p.answer):
            grid[r][c] = ch
    return Layout(grid, placements, layout_metrics(grid, placements))


def build_layout(
    candidates: Sequence[tuple[str, float]],
    *,
    size: int = 15,
    random_seed: int = 13,
    restarts: int = 96,
    max_words: int | None = None,
) -> Layout:
    best_score: dict[str, float] = {}
    for answer, score in candidates:
        if _MIN_WORD_LEN <= len(answer) <= size:
            best_score[answer] = max(score, best_score.get(answer, score))
    ordered = sorted(best_score.items(), key=lambda item: (-item[1], item[0]))
    if not ordered:
        return _to_layout([], size)

    low, high = ordered[-1][1], ordered[0][1]
    spread = high - low
    unit = [(s - low) / spread if spread else 1.0 for _, s in ordered]
    scores = [u * u for u in unit]  # not `** 2`: that calls the platform's pow
    limit = max(max_words, 1) if max_words is not None else 0  # the first word is always placed
    steps = max(restarts, 1) * _STEPS_PER_RESTART
    return _to_layout(_search([w for w, _ in ordered], scores, size, random_seed, steps, limit), size)
