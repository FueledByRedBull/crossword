from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

EMPTY = "#"
ACROSS = "across"
DOWN = "down"

_BLANKS = (EMPTY, ".")
_ACROSS_BIT = 1
_DOWN_BIT = 2
_MIN_WORD_LEN = 3
_START_WORDS = 4

# Placement value = crossings dominate; bounding-box growth and candidate score break ties.
_CROSSING_WEIGHT = 3.0
_GROWTH_WEIGHT = 0.04
_SCORE_WEIGHT = 1.0
_NOISE_WEIGHT = 0.3

# Polish: the best _POLISH_TOP grown boards each get _POLISH_ROUNDS remove-and-refill tries.
# That cost is fixed per call (each try rescans every unused candidate), whatever `restarts` is.
_POLISH_TOP = 4
_POLISH_ROUNDS = 120
_POLISH_REMOVE = 3
_POLISH_NOISE = 1.0


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


class _Board:
    """Sparse, unbounded-coordinate layout under construction.

    Invariant: the board is always a valid layout, so a move only needs local checks.
    """

    def __init__(self, span: int) -> None:
        self.span = span
        self.letters: dict[tuple[int, int], str] = {}
        self.mask: dict[tuple[int, int], int] = {}
        self.by_letter: dict[str, list[tuple[int, int]]] = {}
        self.placements: list[Placement] = []
        self.bounds = (0, 0, 0, 0)  # min_row, max_row, min_col, max_col

    @property
    def area(self) -> int:
        min_r, max_r, min_c, max_c = self.bounds
        return (max_r - min_r + 1) * (max_c - min_c + 1) if self.letters else 0

    def moves(self, word: str) -> dict[tuple[int, int, bool], None]:
        """Starts that cross an existing word; a dict keeps order deterministic and unique."""
        found: dict[tuple[int, int, bool], None] = {}
        for i, ch in enumerate(word):
            for r, c in self.by_letter.get(ch, ()):
                mask = self.mask[(r, c)]
                if mask == _ACROSS_BIT:
                    found[(r - i, c, False)] = None
                elif mask == _DOWN_BIT:
                    found[(r, c - i, True)] = None
        return found

    def fit(self, word: str, r: int, c: int, across: bool) -> tuple[int, int] | None:
        """Return (crossings, resulting bounding-box area) or None if the move is illegal."""
        dr, dc = (0, 1) if across else (1, 0)
        n = len(word)
        if (r - dr, c - dc) in self.letters or (r + dr * n, c + dc * n) in self.letters:
            return None
        min_r, max_r, min_c, max_c = self.bounds
        min_r, max_r = min(min_r, r), max(max_r, r + dr * (n - 1))
        min_c, max_c = min(min_c, c), max(max_c, c + dc * (n - 1))
        if max_r - min_r >= self.span or max_c - min_c >= self.span:
            return None
        own = _ACROSS_BIT if across else _DOWN_BIT
        crossings = 0
        for i, ch in enumerate(word):
            pos = (r + dr * i, c + dc * i)
            have = self.letters.get(pos)
            if have is None:
                if (pos[0] + dc, pos[1] + dr) in self.letters or (pos[0] - dc, pos[1] - dr) in self.letters:
                    return None
            elif have != ch or self.mask[pos] & own:
                return None
            else:
                crossings += 1
        if not crossings:
            return None
        return crossings, (max_r - min_r + 1) * (max_c - min_c + 1)

    def place(self, word: str, r: int, c: int, across: bool) -> None:
        dr, dc = (0, 1) if across else (1, 0)
        own = _ACROSS_BIT if across else _DOWN_BIT
        for i, ch in enumerate(word):
            pos = (r + dr * i, c + dc * i)
            if pos not in self.letters:
                self.letters[pos] = ch
                self.by_letter.setdefault(ch, []).append(pos)
            self.mask[pos] = self.mask.get(pos, 0) | own
        end_r, end_c = r + dr * (len(word) - 1), c + dc * (len(word) - 1)
        if self.placements:
            min_r, max_r, min_c, max_c = self.bounds
            self.bounds = (min(min_r, r), max(max_r, end_r), min(min_c, c), max(max_c, end_c))
        else:
            self.bounds = (r, end_r, c, end_c)
        self.placements.append(Placement(word, r, c, ACROSS if across else DOWN))


def _extend(
    board: _Board,
    remaining: list[tuple[str, float]],
    limit: int,
    rng: random.Random,
    noise: float,
) -> None:
    while remaining and len(board.placements) < limit:
        base_area = board.area
        best_value = None
        best_move = None
        for idx, (word, score) in enumerate(remaining):
            for r, c, across in board.moves(word):
                fitted = board.fit(word, r, c, across)
                if fitted is None:
                    continue
                crossings, area = fitted
                value = (
                    _CROSSING_WEIGHT * crossings
                    - _GROWTH_WEIGHT * (area - base_area)
                    + _SCORE_WEIGHT * score
                    + noise * rng.random()
                )
                if best_value is None or value > best_value:
                    best_value, best_move = value, (idx, r, c, across)
        if best_move is None:
            break
        idx, r, c, across = best_move
        board.place(remaining.pop(idx)[0], r, c, across)


def _grow(
    start: str,
    words: Sequence[tuple[str, float]],
    size: int,
    limit: int,
    rng: random.Random,
    noise: float,
) -> _Board:
    board = _Board(size)
    board.place(start, 0, 0, True)
    _extend(board, [(w, s) for w, s in words if w != start], limit, rng, noise)
    return board


def _connected(placements: Sequence[Placement]) -> bool:
    owners: dict[tuple[int, int], list[int]] = {}
    for idx, p in enumerate(placements):
        for cell in _cells(p):
            owners.setdefault(cell, []).append(idx)
    links: dict[int, set[int]] = {idx: set() for idx in range(len(placements))}
    for shared in owners.values():
        for idx in shared:
            links[idx].update(shared)
    seen = {0}
    stack = [0]
    while stack:
        stack.extend(links[stack.pop()] - seen)
        seen.update(stack)
    return len(seen) == len(placements)


def _key(board: _Board, size: int, scores: dict[str, float]) -> tuple:
    layout = _to_layout(board, size)
    m = layout.metrics
    return (
        m["word_count"],
        m["words_with_two_plus_crossings"],
        m["crossing_count"],
        sum(scores[p.answer] for p in layout.placements),
        m["density"],
    )


def _polish(
    board: _Board,
    words: Sequence[tuple[str, float]],
    size: int,
    limit: int,
    rng: random.Random,
    scores: dict[str, float],
) -> tuple[tuple, _Board]:
    """Drop a few words (weakly crossed ones most often), refill from the rest, keep if no worse."""
    best, best_key = board, _key(board, size, scores)
    for _ in range(_POLISH_ROUNDS):
        crossings = [
            sum(best.mask[cell] == _ACROSS_BIT | _DOWN_BIT for cell in _cells(p))
            for p in best.placements
        ]
        order = sorted(range(len(crossings)), key=lambda i: rng.random() * crossings[i] ** 2)
        gone = set(order[: rng.randint(1, _POLISH_REMOVE)])
        kept = [p for i, p in enumerate(best.placements) if i not in gone]
        if len(kept) < 2 or not _connected(kept):
            continue
        trial = _Board(size)
        for p in kept:
            trial.place(p.answer, p.row, p.col, p.direction == ACROSS)
        used = {p.answer for p in kept}
        _extend(trial, [(w, s) for w, s in words if w not in used], limit, rng, _POLISH_NOISE)
        key = _key(trial, size, scores)
        if key >= best_key:
            best, best_key = trial, key
    return best_key, best


def _to_layout(board: _Board, size: int) -> Layout:
    min_r, max_r, min_c, max_c = board.bounds
    row_off = (size - (max_r - min_r + 1)) // 2 - min_r
    col_off = (size - (max_c - min_c + 1)) // 2 - min_c
    if not board.letters:
        row_off = col_off = 0
    grid = [[EMPTY] * size for _ in range(size)]
    for (r, c), ch in board.letters.items():
        grid[r + row_off][c + col_off] = ch
    placements = [
        Placement(p.answer, p.row + row_off, p.col + col_off, p.direction)
        for p in board.placements
    ]
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
        return _to_layout(_Board(size), size)

    low, high = ordered[-1][1], ordered[0][1]
    spread = high - low
    scaled = [(w, (s - low) / spread if spread else 1.0) for w, s in ordered]
    starts = sorted(scaled, key=lambda item: (-item[1] * len(item[0]), item[0]))[:_START_WORDS]
    limit = max_words if max_words is not None else len(ordered)
    rng = random.Random(random_seed)

    grown = []
    for attempt in range(max(restarts, 1)):
        start = starts[attempt % len(starts)][0]
        board = _grow(start, scaled, size, limit, rng, noise=_NOISE_WEIGHT if attempt else 0.0)
        grown.append((_key(board, size, best_score), -attempt, board))
    grown.sort(key=lambda item: item[:2], reverse=True)
    polished = [
        _polish(board, scaled, size, limit, rng, best_score) for _, _, board in grown[:_POLISH_TOP]
    ]
    return _to_layout(max(polished, key=lambda item: item[0])[1], size)
