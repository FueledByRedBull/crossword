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
    word_scores: dict[str, float] | None = None,
    beam_width: int = 32,
    enable_local_repair: bool = True,
    repair_steps: int = 300,
) -> dict:
    intersections = build_intersections(slots)
    score_lookup = word_scores or {}
    beam_width = max(1, beam_width)
    repair_steps = max(0, repair_steps)

    base_domains: dict[int, list[str]] = {}
    for slot in slots:
        # Keep insertion order stable while removing duplicates.
        base_domains[slot.id] = list(dict.fromkeys(word for word in words if len(word) == slot.length))

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

    def is_consistent(assignments: dict[int, str], used_words: set[str], slot_id: int, word: str) -> bool:
        if word in used_words:
            return False
        for neighbor_id, (a_idx, b_idx) in intersections.get(slot_id, {}).items():
            neighbor_word = assignments.get(neighbor_id)
            if neighbor_word is None:
                continue
            if word[a_idx] != neighbor_word[b_idx]:
                return False
        return True

    def forward_check(
        assignments: dict[int, str],
        used_words: set[str],
        slot_id: int,
        word: str,
        domains: dict[int, list[str]],
    ) -> bool:
        for neighbor_id, (a_idx, b_idx) in intersections.get(slot_id, {}).items():
            if neighbor_id in assignments:
                continue
            allowed = [
                candidate
                for candidate in domains[neighbor_id]
                if candidate not in used_words and candidate[b_idx] == word[a_idx]
            ]
            if not allowed:
                return False
            domains[neighbor_id] = allowed
        return True

    def choose_slot(assignments: dict[int, str], domains: dict[int, list[str]]) -> int | None:
        unassigned = [slot.id for slot in slots if slot.id not in assignments]
        if not unassigned:
            return None

        def key(slot_id: int) -> tuple[int, int]:
            domain_size = len(domains[slot_id])
            degree = len(intersections.get(slot_id, {}))
            return (domain_size, -degree)

        return min(unassigned, key=key)

    def value_score(
        slot_id: int,
        word: str,
        assignments: dict[int, str],
        domains: dict[int, list[str]],
    ) -> float:
        support = 0
        for neighbor_id, (a_idx, b_idx) in intersections.get(slot_id, {}).items():
            if neighbor_id in assignments:
                continue
            support += sum(1 for candidate in domains[neighbor_id] if candidate[b_idx] == word[a_idx])
        return float(support) + (2.0 * score_lookup.get(word, 0.0))

    def state_rank(state: dict) -> tuple[int, float, int]:
        assignments = state["assignments"]
        domains = state["domains"]
        unassigned = [slot.id for slot in slots if slot.id not in assignments]
        domain_pressure = sum(len(domains[slot_id]) for slot_id in unassigned)
        return (len(assignments), state["quality"], -domain_pressure)

    def better_candidate(candidate: dict[int, str], quality: float, incumbent: dict[int, str], incumbent_quality: float) -> bool:
        if len(candidate) != len(incumbent):
            return len(candidate) > len(incumbent)
        return quality > incumbent_quality

    best_overall: dict[int, str] = {}
    best_overall_quality = -1e9
    total_steps = 0
    solved = False
    restarts_used = 0
    local_repair_applied = False

    for restart_idx in range(max_restarts + 1):
        restarts_used += 1
        rng = random.Random(random_seed + (restart_idx * 7919))
        local_domains = {slot_id: list(domain) for slot_id, domain in base_domains.items()}
        if use_ac3 and not ac3(local_domains):
            # AC-3 can prove inconsistency for full completion very early.
            # Fall back to unpruned domains so the search can still recover a
            # best-effort partial assignment.
            local_domains = {slot_id: list(domain) for slot_id, domain in base_domains.items()}

        states = [
            {
                "assignments": {},
                "used_words": set(),
                "domains": local_domains,
                "quality": 0.0,
            }
        ]

        while states and total_steps < max_steps:
            next_states: list[dict] = []
            for state in states:
                assignments = state["assignments"]
                used_words = state["used_words"]
                domains = state["domains"]

                if len(assignments) == len(slots):
                    solved = True
                    if better_candidate(assignments, state["quality"], best_overall, best_overall_quality):
                        best_overall = dict(assignments)
                        best_overall_quality = state["quality"]
                    break

                slot_id = choose_slot(assignments, domains)
                if slot_id is None:
                    solved = True
                    if better_candidate(assignments, state["quality"], best_overall, best_overall_quality):
                        best_overall = dict(assignments)
                        best_overall_quality = state["quality"]
                    break

                candidates = list(domains[slot_id])
                if not candidates:
                    continue
                rng.shuffle(candidates)
                candidates.sort(
                    key=lambda word: value_score(slot_id, word, assignments, domains),
                    reverse=True,
                )
                branch_limit = max(8, min(len(candidates), beam_width * 2))
                for word in candidates[:branch_limit]:
                    total_steps += 1
                    if total_steps > max_steps:
                        break
                    if not is_consistent(assignments, used_words, slot_id, word):
                        continue
                    next_assignments = dict(assignments)
                    next_assignments[slot_id] = word
                    next_used_words = set(used_words)
                    next_used_words.add(word)
                    next_domains = {k: list(v) for k, v in domains.items()}
                    if not forward_check(
                        next_assignments,
                        next_used_words,
                        slot_id,
                        word,
                        next_domains,
                    ):
                        continue
                    quality = state["quality"] + score_lookup.get(word, 0.0)
                    if better_candidate(next_assignments, quality, best_overall, best_overall_quality):
                        best_overall = dict(next_assignments)
                        best_overall_quality = quality
                    next_states.append(
                        {
                            "assignments": next_assignments,
                            "used_words": next_used_words,
                            "domains": next_domains,
                            "quality": quality,
                        }
                    )

            if solved:
                break
            if not next_states:
                break

            ranked = sorted(next_states, key=state_rank, reverse=True)
            deduped: list[dict] = []
            seen_signatures: set[tuple[tuple[int, str], ...]] = set()
            for state in ranked:
                signature = tuple(sorted(state["assignments"].items()))
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                deduped.append(state)
                if len(deduped) >= beam_width:
                    break
            states = deduped

        if solved:
            break

    if not solved and enable_local_repair and repair_steps > 0:
        repaired = dict(best_overall)
        used_words = set(repaired.values())
        repair_budget = repair_steps
        while repair_budget > 0:
            unassigned = [slot.id for slot in slots if slot.id not in repaired]
            if not unassigned:
                break
            progress = False
            unassigned.sort(key=lambda slot_id: (len(base_domains[slot_id]), -len(intersections.get(slot_id, {}))))
            for slot_id in unassigned:
                candidates = [
                    word
                    for word in base_domains[slot_id]
                    if word not in used_words and is_consistent(repaired, used_words, slot_id, word)
                ]
                if not candidates:
                    continue
                candidates.sort(
                    key=lambda word: (
                        score_lookup.get(word, 0.0),
                        len(intersections.get(slot_id, {})),
                    ),
                    reverse=True,
                )
                chosen = candidates[0]
                repaired[slot_id] = chosen
                used_words.add(chosen)
                repair_budget -= 1
                progress = True
                if repair_budget <= 0:
                    break
            if not progress:
                break

        repaired_quality = sum(score_lookup.get(word, 0.0) for word in repaired.values())
        if better_candidate(repaired, repaired_quality, best_overall, best_overall_quality):
            best_overall = repaired
            best_overall_quality = repaired_quality
        local_repair_applied = True

    return {
        "solved": solved,
        "assignments": best_overall,
        "steps": total_steps,
        "restarts": restarts_used,
        "local_repair_applied": local_repair_applied,
    }


def render_grid(grid: list[list[str]], slots: list[Slot], assignments: dict[int, str]) -> list[list[str]]:
    rendered = [row[:] for row in grid]
    slot_by_id = {slot.id: slot for slot in slots}
    for slot_id, word in assignments.items():
        slot = slot_by_id[slot_id]
        for (r, c), letter in zip(slot.cells, word):
            rendered[r][c] = letter
    return rendered
