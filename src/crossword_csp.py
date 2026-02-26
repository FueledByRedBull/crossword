from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass
class Slot:
    id: int
    direction: str
    length: int
    cells: list[tuple[int, int]]


def build_slots(grid: list[list[str]], min_len: int = 3) -> list[Slot]:
    slots: list[Slot] = []
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
                    Slot(
                        id=slot_id,
                        direction="across",
                        length=length,
                        cells=[(r, cc) for cc in range(start, c)],
                    )
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
                    Slot(
                        id=slot_id,
                        direction="down",
                        length=length,
                        cells=[(rr, c) for rr in range(start, r)],
                    )
                )
                slot_id += 1
    return slots


def build_intersections(slots: list[Slot]) -> dict[int, dict[int, tuple[int, int]]]:
    cell_to_slots: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for slot in slots:
        for idx, cell in enumerate(slot.cells):
            cell_to_slots.setdefault(cell, []).append((slot.id, idx))

    intersections: dict[int, dict[int, tuple[int, int]]] = {slot.id: {} for slot in slots}
    for cell, positions in cell_to_slots.items():
        if len(positions) < 2:
            continue
        for a_id, a_idx in positions:
            for b_id, b_idx in positions:
                if a_id == b_id:
                    continue
                intersections[a_id][b_id] = (a_idx, b_idx)
    return intersections


def solve_crossword(
    grid: list[list[str]],
    slots: list[Slot],
    words: list[str],
    *,
    min_len: int = 3,
    max_steps: int = 20000,
    max_restarts: int = 2,
    random_seed: int = 13,
    use_ac3: bool = True,
) -> dict:
    slot_by_id = {slot.id: slot for slot in slots}
    intersections = build_intersections(slots)

    base_domains: dict[int, list[str]] = {}
    for slot in slots:
        base_domains[slot.id] = [word for word in words if len(word) == slot.length]

    def revise(local_domains: dict[int, list[str]], xi: int, xj: int) -> bool:
        revised = False
        a_idx, b_idx = intersections[xi][xj]
        to_remove = []
        for word in local_domains[xi]:
            if not any(word[a_idx] == other[b_idx] for other in local_domains[xj]):
                to_remove.append(word)
        if to_remove:
            revised = True
            local_domains[xi] = [word for word in local_domains[xi] if word not in to_remove]
        return revised

    def ac3(local_domains: dict[int, list[str]]) -> bool:
        queue = [(xi, xj) for xi in local_domains for xj in intersections.get(xi, {})]
        while queue:
            xi, xj = queue.pop(0)
            if revise(local_domains, xi, xj):
                if not local_domains[xi]:
                    return False
                for xk in intersections.get(xi, {}):
                    if xk != xj:
                        queue.append((xk, xi))
        return True

    rng = random.Random(random_seed)
    best_overall: dict[int, str] = {}
    total_steps = 0
    solved = False
    restarts_used = 0

    for _ in range(max_restarts + 1):
        restarts_used += 1
        local_domains = {slot_id: list(words) for slot_id, words in base_domains.items()}
        if use_ac3 and not ac3(local_domains):
            # AC-3 can prove inconsistency for full completion very early.
            # Fall back to unpruned domains so the search can still recover a
            # best-effort partial assignment.
            local_domains = {slot_id: list(words) for slot_id, words in base_domains.items()}

        assignments: dict[int, str] = {}
        used_words: set[str] = set()
        steps = 0
        best_assignments: dict[int, str] = {}

        def is_consistent(slot_id: int, word: str) -> bool:
            # Uniqueness: reject if this word is already placed elsewhere.
            if word in used_words:
                return False
            for neighbor_id, (a_idx, b_idx) in intersections.get(slot_id, {}).items():
                if neighbor_id not in assignments:
                    continue
                neighbor_word = assignments[neighbor_id]
                if word[a_idx] != neighbor_word[b_idx]:
                    return False
            return True

        def forward_check(slot_id: int, word: str, domains: dict[int, list[str]]) -> bool:
            for neighbor_id, (a_idx, b_idx) in intersections.get(slot_id, {}).items():
                if neighbor_id in assignments:
                    continue
                allowed = [
                    candidate for candidate in domains[neighbor_id]
                    if candidate[b_idx] == word[a_idx]
                ]
                if not allowed:
                    return False
                domains[neighbor_id] = allowed
            return True

        def choose_slot(domains: dict[int, list[str]]) -> int | None:
            unassigned = [slot.id for slot in slots if slot.id not in assignments]
            if not unassigned:
                return None

            def key(slot_id: int) -> tuple[int, int]:
                domain_size = len(domains[slot_id])
                degree = len(intersections.get(slot_id, {}))
                return (domain_size, -degree)

            return min(unassigned, key=key)

        def order_values(slot_id: int, domains: dict[int, list[str]]) -> list[str]:
            candidates = list(domains[slot_id])
            if not candidates:
                return candidates
            neighbor_freqs: list[tuple[int, dict[str, int]]] = []
            for neighbor_id, (a_idx, b_idx) in intersections.get(slot_id, {}).items():
                if neighbor_id in assignments:
                    continue
                freq: dict[str, int] = {}
                for candidate in domains[neighbor_id]:
                    letter = candidate[b_idx]
                    freq[letter] = freq.get(letter, 0) + 1
                neighbor_freqs.append((a_idx, freq))
            rng.shuffle(candidates)

            def score(word: str) -> int:
                total = 0
                for a_idx, freq in neighbor_freqs:
                    total += freq.get(word[a_idx], 0)
                return total

            candidates.sort(key=score, reverse=True)
            return candidates

        def backtrack(domains: dict[int, list[str]]) -> bool:
            nonlocal steps, best_assignments
            if len(assignments) > len(best_assignments):
                best_assignments = dict(assignments)
            if steps > max_steps:
                return False
            slot_id = choose_slot(domains)
            if slot_id is None:
                return True
            for word in order_values(slot_id, domains):
                steps += 1
                if not is_consistent(slot_id, word):
                    continue
                assignments[slot_id] = word
                used_words.add(word)
                if len(assignments) > len(best_assignments):
                    best_assignments = dict(assignments)
                next_domains = {k: list(v) for k, v in domains.items()}
                if forward_check(slot_id, word, next_domains):
                    if backtrack(next_domains):
                        return True
                assignments.pop(slot_id, None)
                used_words.discard(word)
            return False

        solved_local = backtrack(local_domains)
        total_steps += steps
        if solved_local:
            solved = True
            best_overall = dict(assignments)
            break
        if len(best_assignments) > len(best_overall):
            best_overall = dict(best_assignments)

    return {
        "solved": solved,
        "assignments": best_overall,
        "steps": total_steps,
        "restarts": restarts_used,
    }


def render_grid(grid: list[list[str]], slots: list[Slot], assignments: dict[int, str]) -> list[list[str]]:
    rendered = [row[:] for row in grid]
    slot_by_id = {slot.id: slot for slot in slots}
    for slot_id, word in assignments.items():
        slot = slot_by_id[slot_id]
        for (r, c), letter in zip(slot.cells, word):
            rendered[r][c] = letter
    return rendered
