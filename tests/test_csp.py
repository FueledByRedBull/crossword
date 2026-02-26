import unittest

from src.crossword_csp import build_slots, render_grid, solve_crossword


class CspTests(unittest.TestCase):
    def test_solve_small_grid(self) -> None:
        # 5×5 grid with solid black rows at rows 2 — creates 4 across slots of
        # length 5 (rows 0,1,3,4) and 2 down slots of length 2 (pruned) + 5
        # down slots of length 5. Provide a pool of 5-letter words.
        # Simpler: use a 5×5 with central black row/col cross so only short slots remain.
        # Easiest: single across slot only — 1×5 would need size=5 square grid.
        # Use a 5×5 grid with all-black rows 1 and 3, giving 3 across-only zones.
        grid = [
            [".", ".", ".", ".", "."],
            ["#", "#", "#", "#", "#"],
            [".", ".", ".", ".", "."],
            ["#", "#", "#", "#", "#"],
            [".", ".", ".", ".", "."],
        ]
        slots = build_slots(grid, min_len=3)
        # 3 across slots of length 5; down slots all get chopped to length 1 → pruned.
        words = ["PIANO", "ORGAN", "FLUTE", "VIOLA", "CELLO", "DRUMS"]
        result = solve_crossword(
            grid, slots, words, min_len=3, max_steps=1000, use_ac3=False
        )
        self.assertTrue(result["assignments"])
        rendered = render_grid(grid, slots, result["assignments"])
        self.assertEqual(len(rendered), 5)

    def test_no_repeated_answers(self) -> None:
        """Each slot must receive a unique word (uniqueness constraint)."""
        grid = [
            [".", ".", ".", ".", "."],
            ["#", "#", "#", "#", "#"],
            [".", ".", ".", ".", "."],
            ["#", "#", "#", "#", "#"],
            [".", ".", ".", ".", "."],
        ]
        slots = build_slots(grid, min_len=3)
        words = ["PIANO", "ORGAN", "FLUTE", "VIOLA", "CELLO", "DRUMS"]
        result = solve_crossword(
            grid, slots, words, min_len=3, max_steps=1000, use_ac3=False
        )
        assigned = list(result["assignments"].values())
        self.assertEqual(
            len(assigned), len(set(assigned)), "Repeated answers found in grid"
        )

    def test_word_scores_drive_value_ordering(self) -> None:
        grid = [
            [".", ".", "."],
            ["#", "#", "#"],
            ["#", "#", "#"],
        ]
        slots = build_slots(grid, min_len=3)
        words = ["CAT", "DOG"]
        result = solve_crossword(
            grid,
            slots,
            words,
            min_len=3,
            max_steps=100,
            use_ac3=False,
            beam_width=2,
            word_scores={"CAT": 0.1, "DOG": 0.9},
        )
        assigned_words = list(result["assignments"].values())
        self.assertTrue(assigned_words)
        self.assertEqual(assigned_words[0], "DOG")


if __name__ == "__main__":
    unittest.main()
