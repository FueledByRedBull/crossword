import unittest

from src.render import render_puzzle_text


class RenderPuzzleTests(unittest.TestCase):
    def test_valid_puzzle_renders_numbers_clues_and_ok(self) -> None:
        puzzle = {
            "seed_title": "Cats",
            "grid_cells": [list("###"), list("###"), list("CAT"), list("A##"), list("R##")],
            "across_entries": [
                {"answer": "CAT", "clue": "Pet", "row": 2, "col": 0, "length": 3},
            ],
            "down_entries": [
                {"answer": "CAR", "clue": "Vehicle", "row": 2, "col": 0, "length": 3, "enumeration": "3"},
            ],
        }
        lines = render_puzzle_text(puzzle).splitlines()
        self.assertEqual(lines[0], "Cats")
        self.assertEqual(lines[2], " 1 _ _")
        self.assertIn("1. Pet (3)", lines)
        self.assertIn("1. Vehicle (3)", lines)
        self.assertIn(" C A T", lines)
        self.assertEqual(lines[-1], "Validity: OK")

    def test_legacy_puzzle_renders_and_reports_problems(self) -> None:
        puzzle = {
            "seed_title": "Legacy",
            "grid_cells": [list("AB."), list("CD.")],
            "across_entries": [],
            "down_entries": [
                {"answer": "AC", "clue": "", "row": 0, "col": 0, "length": 2},
                {"answer": "BD", "clue": "", "row": 0, "col": 1, "length": 2},
            ],
        }
        text = render_puzzle_text(puzzle)
        self.assertIn("2.  (2)", text)
        self.assertNotIn("Validity: OK", text)
        self.assertIn("non-word across run", text)


if __name__ == "__main__":
    unittest.main()
