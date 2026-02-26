import unittest

from src.topology import (
    build_grid,
    extract_slots,
    get_templates,
    score_template_from_length_hist,
    select_best_template,
)


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


if __name__ == "__main__":
    unittest.main()
