import unittest
from collections import deque

from src.topology import (
    auto_block_long_slots,
    build_grid,
    extract_slots,
    get_templates,
    make_symmetric_blocks,
    score_template_from_length_hist,
    select_best_template,
)


def _is_symmetric(grid: list[list[str]]) -> bool:
    """Check 180° rotational symmetry."""
    size = len(grid)
    for r in range(size):
        for c in range(size):
            if (grid[r][c] == "#") != (grid[size - 1 - r][size - 1 - c] == "#"):
                return False
    return True


def _is_connected(grid: list[list[str]]) -> bool:
    """Check that all white cells form a single connected component."""
    size = len(grid)
    whites = {(r, c) for r in range(size) for c in range(size) if grid[r][c] != "#"}
    if not whites:
        return True
    start = next(iter(whites))
    visited: set[tuple[int, int]] = set()
    queue: deque[tuple[int, int]] = deque([start])
    while queue:
        cell = queue.popleft()
        if cell in visited:
            continue
        visited.add(cell)
        r, c = cell
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if (nr, nc) in whites and (nr, nc) not in visited:
                queue.append((nr, nc))
    return len(visited) == len(whites)


class TopologyTests(unittest.TestCase):
    def test_extract_slots_open(self) -> None:
        template = get_templates(5)[0]
        grid = build_grid(template)
        slots = extract_slots(grid, min_len=3)
        # 5x5 open grid has 5 across and 5 down slots
        self.assertEqual(len(slots), 10)

    def test_select_best_template(self) -> None:
        words = ["CAT", "DOG", "BIRD", "FISH"]
        selection = select_best_template(words, size=5, min_len=3)
        self.assertIn("selected", selection)

    def test_fill_conflict_penalizes_slot_shortage(self) -> None:
        dense = next(template for template in get_templates(15) if template.name == "dense")
        # Far fewer 4-letter words than dense template's 4-letter slot demand.
        score = score_template_from_length_hist({4: 3}, dense, min_len=4)
        self.assertGreater(score["fill_conflict"], 0.5)

    def test_select_best_template_avoids_dense_when_inventory_is_too_small(self) -> None:
        words = (
            ["A" * 7 for _ in range(20)]
            + ["B" * 5 for _ in range(10)]
            + ["C" * 11 for _ in range(5)]
            + ["D" * 4 for _ in range(2)]
        )
        selection = select_best_template(words, size=15, min_len=4)
        self.assertEqual(selection["selected"], "symmetric_sparse")

    def test_auto_block_reduces_overlong_slots(self) -> None:
        template = next(template for template in get_templates(15) if template.name == "open")
        grid = build_grid(template)
        blocked = auto_block_long_slots(grid, max_slot_len=12, symmetric=True)
        slots = extract_slots(blocked["grid"], min_len=1)
        self.assertTrue(blocked["added_blocks"])
        self.assertTrue(all(slot["length"] <= 12 for slot in slots))

    # --- New template validation tests ---

    def test_all_15x15_templates_have_rotational_symmetry(self) -> None:
        # dense uses _grid_blocks which doesn't guarantee symmetry
        for template in get_templates(15):
            if template.name == "dense":
                continue
            grid = build_grid(template)
            self.assertTrue(
                _is_symmetric(grid),
                f"Template {template.name!r} is not rotationally symmetric",
            )

    def test_all_13x13_templates_have_rotational_symmetry(self) -> None:
        for template in get_templates(13):
            if template.name == "dense":
                continue
            grid = build_grid(template)
            self.assertTrue(
                _is_symmetric(grid),
                f"Template {template.name!r} is not rotationally symmetric",
            )

    def test_all_15x15_templates_have_connected_white_cells(self) -> None:
        for template in get_templates(15):
            if template.name == "dense":
                continue
            grid = build_grid(template)
            self.assertTrue(
                _is_connected(grid),
                f"Template {template.name!r} has disconnected white cells",
            )

    def test_all_13x13_templates_have_connected_white_cells(self) -> None:
        for template in get_templates(13):
            if template.name == "dense":
                continue
            grid = build_grid(template)
            self.assertTrue(
                _is_connected(grid),
                f"Template {template.name!r} has disconnected white cells",
            )

    def test_all_15x15_templates_slot_bounds(self) -> None:
        for template in get_templates(15):
            grid = build_grid(template)
            slots = extract_slots(grid, min_len=3)
            slot_count = len(slots)
            self.assertGreaterEqual(
                slot_count, 5,
                f"Template {template.name!r} has only {slot_count} slots",
            )
            # After auto-blocking, no slot should exceed 12
            blocked = auto_block_long_slots(grid, max_slot_len=12, symmetric=True)
            blocked_slots = extract_slots(blocked["grid"], min_len=1)
            for slot in blocked_slots:
                self.assertLessEqual(
                    slot["length"], 12,
                    f"Template {template.name!r} has slot len={slot['length']} after auto-blocking",
                )

    def test_all_13x13_templates_slot_bounds(self) -> None:
        for template in get_templates(13):
            grid = build_grid(template)
            slots = extract_slots(grid, min_len=3)
            slot_count = len(slots)
            self.assertGreaterEqual(
                slot_count, 5,
                f"Template {template.name!r} has only {slot_count} slots",
            )

    def test_nyt_classic_slot_count_reasonable(self) -> None:
        nyt = next(t for t in get_templates(15) if t.name == "nyt_classic")
        grid = build_grid(nyt)
        slots = extract_slots(grid, min_len=3)
        self.assertGreaterEqual(len(slots), 20)
        self.assertLessEqual(len(slots), 70)

    def test_medium_open_slot_count_reasonable(self) -> None:
        medium = next(t for t in get_templates(15) if t.name == "medium_open")
        grid = build_grid(medium)
        slots = extract_slots(grid, min_len=3)
        self.assertGreaterEqual(len(slots), 20)
        self.assertLessEqual(len(slots), 70)


if __name__ == "__main__":
    unittest.main()
