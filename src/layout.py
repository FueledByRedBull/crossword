from __future__ import annotations

import itertools
import math
import random
from collections import Counter
from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Iterable, Sequence

EMPTY = "#"
ACROSS = "across"
DOWN = "down"

_BLANKS = (EMPTY, ".")
_MIN_WORD_LEN = 3

# A cell is the int row * _ROW + col, so its neighbours are +-1 and +-_ROW. The row is so wide that
# a column cannot wrap into the next row however far the board drifts while the search moves it.
_ROW_BITS = 32
_ROW = 1 << _ROW_BITS
_COL_MASK = _ROW - 1
_ORIGIN = (_ROW // 2) * (_ROW + 1)

# Search value of a layout: 1 per word, a bonus per crossing, and a bonus per answer score squared.
# Scores are scaled to 0..1, so only the top answers earn much: one outweighs a word or two.
_CROSSING_VALUE = 1.5
_SCORE_VALUE = 3.0

# Simulated annealing over ruin-and-recreate steps: each step drops 1.._RUIN_MAX random words,
# refills greedily, and keeps the result unless it is worse than the current board by more than a
# random allowance that cools linearly from _START_TEMPERATURE to zero. A step takes about 2 ms and
# ten times the default step count gains only a fraction of a word, so `restarts` scales the length
# of this one run instead of starting independent ones.
_STEPS_PER_RESTART = 25
_RUIN_MAX = 3
_START_TEMPERATURE = 3.0

# Refill takes the legal placement with the best value per newly used cell (square-rooted, so long
# answers are not ruled out), raised by a random factor of up to 1 + _PICK_NOISE so refills differ.
_PICK_NOISE = 1.0


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
    """Sparse board with unbounded coordinates, built from and cut back by whole words.

    Invariant: the board is always a valid layout, so a move only needs local checks.
    """

    def __init__(self, span: int) -> None:
        self.span = span
        self.letters: dict[int, str] = {}
        self.owner: dict[bool, dict[int, str]] = {True: {}, False: {}}  # keyed by across
        self.placed: dict[str, tuple[int, bool]] = {}  # answer -> (start cell, across)
        self.crossings = 0
        self.bounds = (0, 0, 0, 0)  # min_row, max_row, min_col, max_col

    def cells(self, word: str) -> list[int]:
        pos, across = self.placed[word]
        step = 1 if across else _ROW
        return [pos + step * i for i in range(len(word))]

    def fit(self, word: str, pos: int, across: bool) -> int | None:
        """Return the crossings `word` would make from `pos`, or None if the move is illegal."""
        n = len(word)
        lo_r, hi_r, lo_c, hi_c = self.bounds
        reach = self.span - 1
        r, c = pos >> _ROW_BITS, pos & _COL_MASK
        if across:
            step, side = 1, _ROW
            if r < hi_r - reach or r > lo_r + reach or c < hi_c - reach or c + n - 1 > lo_c + reach:
                return None
        else:
            step, side = _ROW, 1
            if c < hi_c - reach or c > lo_c + reach or r < hi_r - reach or r + n - 1 > lo_r + reach:
                return None
        letters, mine = self.letters, self.owner[across]
        if pos - step in letters or pos + step * n in letters:
            return None
        crossings = 0
        for ch in word:
            have = letters.get(pos)
            if have is None:
                if pos + side in letters or pos - side in letters:
                    return None
            elif have != ch or pos in mine:
                return None
            else:
                crossings += 1
            pos += step
        return crossings or None

    def anchors(self, cells: Iterable[int]) -> dict[str, dict[int, bool]]:
        """Single-owner cells by letter; the value says whether a crossing word runs across."""
        down = self.owner[False]
        found: dict[str, dict[int, bool]] = {}
        for pos in cells:
            if (pos in self.owner[True]) != (pos in down):
                found.setdefault(self.letters[pos], {})[pos] = pos in down
        return found

    def moves(self, word: str, anchors: dict[str, dict[int, bool]]) -> list[tuple[int, int, bool]]:
        """Legal (crossings, start cell, across) placements of `word` through the given anchors."""
        found = []
        for i, ch in enumerate(word):
            for anchor, across in anchors.get(ch, {}).items():
                pos = anchor - i if across else anchor - i * _ROW
                crossings = self.fit(word, pos, across)
                if crossings:
                    found.append((crossings, pos, across))
        return found

    def place(self, word: str, pos: int, across: bool) -> None:
        step = 1 if across else _ROW
        mine = self.owner[across]
        self.placed[word] = (pos, across)
        for ch in word:
            if pos in self.letters:
                self.crossings += 1
            else:
                self.letters[pos] = ch
            mine[pos] = word
            pos += step
        self._rebound()

    def remove(self, word: str) -> None:
        pos, across = self.placed.pop(word)
        step = 1 if across else _ROW
        mine, theirs = self.owner[across], self.owner[not across]
        for _ in word:
            del mine[pos]
            if pos in theirs:
                self.crossings -= 1
            else:
                del self.letters[pos]
            pos += step
        if self.placed:
            self._rebound()

    def _rebound(self) -> None:
        spans = []
        for word, (pos, across) in self.placed.items():
            r, c = pos >> _ROW_BITS, pos & _COL_MASK
            end = len(word) - 1
            spans.append((r, r if across else r + end, c, c + end if across else c))
        lo_r, hi_r, lo_c, hi_c = zip(*spans)
        self.bounds = (min(lo_r), max(hi_r), min(lo_c), max(hi_c))

    def components(self) -> list[list[str]]:
        """Groups of words joined by crossings."""
        seen: set[str] = set()
        groups = []
        for first in self.placed:
            if first in seen:
                continue
            seen.add(first)
            group = [first]
            for word in group:  # the loop also visits words appended meanwhile
                theirs = self.owner[not self.placed[word][1]]
                for cell in self.cells(word):
                    other = theirs.get(cell)
                    if other is not None and other not in seen:
                        seen.add(other)
                        group.append(other)
            groups.append(group)
        return groups

    def value(self, scores: dict[str, float]) -> float:
        return (
            len(self.placed)
            + _CROSSING_VALUE * self.crossings
            + _SCORE_VALUE * sum(scores[word] for word in self.placed)
        )


def _refill(
    board: _Board, words: Sequence[str], scores: dict[str, float], rng: random.Random, limit: int
) -> list[str]:
    """Place unused words best-first until none fits; return them in placement order.

    One scan finds every legal move; after that a placement only adds the moves that cross its own
    new letters, and stale moves are re-checked when they come off the heap.
    """
    heap: list[tuple[float, int, str, int, bool]] = []
    ticket = itertools.count()  # unique, so equal gains never compare words
    placed = board.placed
    unused = [word for word in words if word not in placed]

    def offer(word: str, anchors: dict[str, dict[int, bool]]) -> None:
        for crossings, pos, across in board.moves(word, anchors):
            gain = 1 + _CROSSING_VALUE * crossings + _SCORE_VALUE * scores[word]
            gain /= math.sqrt(max(len(word) - crossings, 1))
            gain *= 1 + _PICK_NOISE * rng.random()
            heappush(heap, (-gain, next(ticket), word, pos, across))

    everywhere = board.anchors(board.letters)
    for word in unused:
        offer(word, everywhere)
    added = []
    while heap and len(placed) < limit:
        _, _, word, pos, across = heappop(heap)
        if word in placed or board.fit(word, pos, across) is None:
            continue
        board.place(word, pos, across)
        added.append(word)
        fresh = board.anchors(board.cells(word))
        for other in unused:
            if other not in placed:
                offer(other, fresh)
    return added


def _search(
    words: Sequence[str],
    scores: dict[str, float],
    size: int,
    rng: random.Random,
    steps: int,
    limit: int,
) -> dict[str, tuple[int, bool]]:
    board = _Board(size)
    board.place(words[0], _ORIGIN, True)
    _refill(board, words, scores, rng, limit)
    value = board.value(scores)
    best_value, best = value, dict(board.placed)
    for step in range(steps):
        if len(board.placed) < 2:
            break
        temperature = _START_TEMPERATURE * (1 - step / steps)
        dropped = {}
        ruined = rng.randint(1, min(_RUIN_MAX, len(board.placed) - 1))
        for word in rng.sample(list(board.placed), ruined):
            dropped[word] = board.placed[word]
            board.remove(word)
        groups = sorted(board.components(), key=len, reverse=True)
        for word in (word for group in groups[1:] for word in group):  # cut off from the main group
            dropped[word] = board.placed[word]
            board.remove(word)
        added = _refill(board, words, scores, rng, limit)
        trial = board.value(scores)
        if trial >= value + temperature * math.log(1 - rng.random()):
            value = trial
            if value > best_value:
                best_value, best = value, dict(board.placed)
        else:
            for word in reversed(added):
                board.remove(word)
            for word, (pos, across) in dropped.items():
                board.place(word, pos, across)
    return best


def _to_layout(placed: dict[str, tuple[int, bool]], size: int) -> Layout:
    raw = [
        Placement(word, pos >> _ROW_BITS, pos & _COL_MASK, ACROSS if across else DOWN)
        for word, (pos, across) in placed.items()
    ]
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
        return _to_layout({}, size)

    low, high = ordered[-1][1], ordered[0][1]
    spread = high - low
    scores = {w: ((s - low) / spread if spread else 1.0) ** 2 for w, s in ordered}
    limit = max_words if max_words is not None else len(ordered)
    steps = max(restarts, 1) * _STEPS_PER_RESTART
    rng = random.Random(random_seed)
    return _to_layout(_search([w for w, _ in ordered], scores, size, rng, steps, limit), size)
