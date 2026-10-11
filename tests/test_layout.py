import copy
import os
import random
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

from src import layout as layout_module
from src.layout import (
    Placement,
    _Board,
    build_layout,
    number_placements,
    validate_layout,
    validate_puzzle,
)

WORDS = [
    "HEAT", "ENERGY", "ENTROPY", "WORK", "TEMPERATURE", "PRESSURE", "VOLUME", "GAS",
    "ENGINE", "CARNOT", "KELVIN", "JOULE", "WATT", "FLUID", "PHASE", "STATE", "CYCLE",
    "LAW", "ZERO", "BOILING", "FREEZING", "MELTING", "RADIATION", "CONVECTION",
    "CONDUCTION", "MOLECULE", "ATOM", "PARTICLE", "SYSTEM", "EQUILIBRIUM", "THEORY",
    "HEATER", "STEAM", "ICE", "WATER", "OXYGEN", "NITROGEN", "ARGON", "NEON", "RADON",
    "ENTHALPY", "GIBBS", "FREE", "BOLTZMANN", "CLAUSIUS", "RANKINE", "OTTO", "DIESEL",
]

# The 50 candidates of a real "Thermodynamics" puzzle (score = centrality), most central first. Many
# are long multi-word titles, which is what makes dense layouts hard.
THERMODYNAMICS = [
    ("THERMODYNAMICS", 3.5), ("ENTROPY", 2.97), ("INTERNALENERGY", 2.86), ("HEAT", 2.81),
    ("ENGINE", 2.78), ("TEMPERATURE", 2.78), ("VOLUME", 2.69), ("STATE", 2.68), ("BOUNDARY", 2.48),
    ("RUDOLFCLAUSIUS", 2.29), ("ENGINEERING", 2.27), ("MATTER", 2.22), ("ABSOLUTEZERO", 2.2),
    ("PRESSURE", 2.19), ("PHYSICS", 2.07), ("WILLARDGIBBS", 2.03), ("MECHANICS", 2.02),
    ("SCIENCE", 1.98), ("CARNOTCYCLE", 1.95), ("LUDWIGBOLTZMANN", 1.92), ("POWER", 1.91),
    ("ISOLATEDSYSTEM", 1.89), ("MERLERANDALL", 1.85), ("ENTHALPY", 1.85), ("JULIUSVONMAYER", 1.82),
    ("GILBERTNLEWIS", 1.81), ("PERPETUALMOTION", 1.8), ("PHASE", 1.8), ("ASTROPHYSICS", 1.78),
    ("MAXWELLSDEMON", 1.76), ("MICROSCOPIC", 1.75), ("JOULEEXPANSION", 1.75),
    ("BROWNIANRATCHET", 1.75), ("PIERREDUHEM", 1.74), ("EXERGY", 1.73), ("WALTHERNERNST", 1.71),
    ("PETERTAIT", 1.68), ("COMPRESSIBILITY", 1.66), ("FRANCOISMASSIEU", 1.65),
    ("SYNERGETICS", 1.64), ("AXIOM", 1.64), ("EMILECLAPEYRON", 1.64), ("PSYCHROMETRICS", 1.64),
    ("NUCLEATION", 1.61), ("VACUUM", 1.54), ("IDEALGAS", 1.53), ("CHEMISTRY", 1.5),
    ("PISTON", 1.46), ("RADIATION", 1.42), ("THERMOMETER", 1.34),
]


def _grid(*rows: str) -> list[list[str]]:
    return [list(row) for row in rows]


class BuildLayoutTests(unittest.TestCase):
    def test_random_samples_are_valid(self) -> None:
        for seed in range(24):
            rng = random.Random(seed)
            sample = [(w, rng.random()) for w in rng.sample(WORDS, 25)]
            layout = build_layout(sample, size=13, random_seed=seed, restarts=4)
            with self.subTest(seed=seed):
                self.assertEqual(len(layout.grid), 13)
                self.assertTrue(all(len(row) == 13 for row in layout.grid))
                self.assertGreaterEqual(len(layout.placements), 2)
                self.assertEqual(validate_layout(layout.grid, layout.placements), [])
                answers = [p.answer for p in layout.placements]
                self.assertEqual(len(answers), len(set(answers)))
                for p in layout.placements:
                    end = (p.row, p.col + len(p.answer) - 1) if p.direction == "across" else (
                        p.row + len(p.answer) - 1, p.col)
                    self.assertTrue(0 <= p.row and 0 <= p.col and end[0] < 13 and end[1] < 13)

    def test_same_inputs_give_identical_layout(self) -> None:
        candidates = [(w, 1.0 - i / 100) for i, w in enumerate(WORDS)]
        first = build_layout(candidates, restarts=6)
        second = build_layout(candidates, restarts=6)
        self.assertEqual(first, second)

    def test_layout_is_stable_across_hash_seeds(self) -> None:
        code = (
            "from tests.test_layout import WORDS\n"
            "from src.layout import build_layout\n"
            "layout = build_layout([(w, 1.0) for w in WORDS], restarts=4)\n"
            "print(layout.placements)\n"
            "print(layout.grid)\n"
        )
        root = Path(__file__).resolve().parent.parent
        outputs = []
        for hash_seed in ("1", "2"):
            env = {**os.environ, "PYTHONHASHSEED": hash_seed}
            result = subprocess.run(
                [sys.executable, "-c", code], cwd=root, env=env,
                capture_output=True, text=True, check=True,
            )
            outputs.append(result.stdout)
        self.assertEqual(outputs[0], outputs[1])

    def test_real_sized_list_is_deterministic(self) -> None:
        self.assertEqual(
            build_layout(THERMODYNAMICS, restarts=8), build_layout(THERMODYNAMICS, restarts=8)
        )

    def test_real_sized_list_is_valid_and_denser_than_the_old_search(self) -> None:
        layout = build_layout(THERMODYNAMICS)
        self.assertEqual(validate_layout(layout.grid, layout.placements), [])
        metrics = layout.metrics
        # The greedy-and-polish search this replaced placed 18 words with 21 crossings on this list.
        self.assertGreaterEqual(metrics["word_count"], 19)
        self.assertGreaterEqual(metrics["crossing_count"], 21)
        placed = {p.answer for p in layout.placements}
        top_ten = [answer for answer, _ in sorted(THERMODYNAMICS, key=lambda c: -c[1])[:10]]
        self.assertGreaterEqual(sum(answer in placed for answer in top_ten), 6)

    def test_search_improves_on_its_first_greedy_fill(self) -> None:
        for seed in range(3):
            with self.subTest(seed=seed):
                with mock.patch.object(layout_module, "_STEPS_PER_RESTART", 0):
                    plain = build_layout(THERMODYNAMICS, random_seed=seed)
                searched = build_layout(THERMODYNAMICS, random_seed=seed, restarts=24)
                self.assertGreater(searched.metrics["word_count"], plain.metrics["word_count"])

    def test_search_work_is_a_fixed_number_of_steps(self) -> None:
        with mock.patch.object(layout_module, "_refill", wraps=layout_module._refill) as refill:
            build_layout(THERMODYNAMICS, restarts=2)
        self.assertEqual(refill.call_count, 1 + 2 * layout_module._STEPS_PER_RESTART)

    def test_bad_candidates_are_ignored(self) -> None:
        layout = build_layout(
            [("CAT", 0.5), ("CAT", 0.9), ("TO", 1.0), ("ACTIVITIES", 1.0), ("TAR", 0.4)],
            size=5,
        )
        self.assertEqual(sorted(p.answer for p in layout.placements), ["CAT", "TAR"])
        self.assertEqual(validate_layout(layout.grid, layout.placements), [])

    def test_max_words_is_respected(self) -> None:
        layout = build_layout([(w, 1.0) for w in WORDS], max_words=5, restarts=4)
        self.assertEqual(len(layout.placements), 5)
        self.assertEqual(validate_layout(layout.grid, layout.placements), [])

    def test_unplaceable_input_returns_what_it_has(self) -> None:
        self.assertEqual(build_layout([]).placements, [])
        single = build_layout([("CAT", 1.0)], size=5)
        self.assertEqual([p.answer for p in single.placements], ["CAT"])
        self.assertTrue(validate_layout(single.grid, single.placements))


class BoardTests(unittest.TestCase):
    def test_remove_undoes_place(self) -> None:
        board = _Board(15)
        board.place("CAT", layout_module._ORIGIN, True)
        board.place("CAR", layout_module._ORIGIN, False)
        before = copy.deepcopy(vars(board))
        board.place("TOE", layout_module._ORIGIN + 2, False)
        self.assertEqual(board.crossings, 2)
        board.remove("TOE")
        self.assertEqual(vars(board), before)

    def test_fit_refuses_words_running_alongside_or_past_the_span(self) -> None:
        board = _Board(3)
        board.place("CAT", layout_module._ORIGIN, True)
        board.place("CAR", layout_module._ORIGIN, False)
        self.assertIsNone(board.fit("ARE", layout_module._ORIGIN + 1, False))
        self.assertEqual(board.fit("TOE", layout_module._ORIGIN + 2, False), 1)
        self.assertIsNone(board.fit("TOES", layout_module._ORIGIN + 2, False))

    def test_components_split_when_a_bridge_word_goes(self) -> None:
        board = _Board(15)
        board.place("CAT", layout_module._ORIGIN, True)
        board.place("TOE", layout_module._ORIGIN + 2, False)
        board.place("EAR", layout_module._ORIGIN + 2 * layout_module._ROW, True)
        self.assertEqual(len(board.components()), 1)
        board.remove("TOE")
        self.assertEqual(sorted(sorted(group) for group in board.components()), [["CAT"], ["EAR"]])


class ValidateLayoutTests(unittest.TestCase):
    def test_accepts_connected_crossing(self) -> None:
        grid = _grid("CAT", "A##", "R##")
        placements = [Placement("CAT", 0, 0, "across"), Placement("CAR", 0, 0, "down")]
        self.assertEqual(validate_layout(grid, placements), [])

    def test_rejects_side_by_side_words(self) -> None:
        grid = _grid("CAT", "ARE", "R##")
        placements = [
            Placement("CAT", 0, 0, "across"),
            Placement("CAR", 0, 0, "down"),
            Placement("ARE", 1, 0, "across"),
        ]
        problems = validate_layout(grid, placements)
        self.assertTrue(any("non-word down run" in p and "'AR'" in p for p in problems))

    def test_rejects_letter_mismatch(self) -> None:
        grid = _grid("CAT", "A##", "R##")
        placements = [Placement("COT", 0, 0, "across"), Placement("CAR", 0, 0, "down")]
        problems = validate_layout(grid, placements)
        self.assertTrue(any("grid spells 'CAT'" in p for p in problems))

    def test_rejects_disconnected_word(self) -> None:
        grid = _grid("CAT#DOG", "A######", "R######")
        placements = [
            Placement("CAT", 0, 0, "across"),
            Placement("CAR", 0, 0, "down"),
            Placement("DOG", 0, 4, "across"),
        ]
        problems = validate_layout(grid, placements)
        self.assertTrue(any("'DOG'" in p and "crosses no other word" in p for p in problems))

    def test_rejects_two_separate_groups(self) -> None:
        grid = _grid("CAT#DOG", "A###U##", "R###G##")
        placements = [
            Placement("CAT", 0, 0, "across"),
            Placement("CAR", 0, 0, "down"),
            Placement("DOG", 0, 4, "across"),
            Placement("DUG", 0, 4, "down"),
        ]
        problems = validate_layout(grid, placements)
        self.assertTrue(any("2 disconnected groups" in p for p in problems))

    def test_rejects_duplicate_answer(self) -> None:
        grid = _grid("CAT", "A#A", "T#T")
        placements = [
            Placement("CAT", 0, 0, "across"),
            Placement("CAT", 0, 0, "down"),
            Placement("CAT", 0, 2, "down"),
        ]
        problems = validate_layout(grid, placements)
        self.assertTrue(any("'CAT' appears 3 times" in p for p in problems))

    def test_rejects_single_word(self) -> None:
        grid = _grid("CAT", "###", "###")
        problems = validate_layout(grid, [Placement("CAT", 0, 0, "across")])
        self.assertTrue(any("at least 2 words" in p for p in problems))

    def test_rejects_word_extending_past_its_run(self) -> None:
        grid = _grid("CATS", "A###", "R###")
        placements = [Placement("CAT", 0, 0, "across"), Placement("CAR", 0, 0, "down")]
        problems = validate_layout(grid, placements)
        self.assertTrue(any("touches other letters" in p for p in problems))

    def test_rejects_letter_in_no_word(self) -> None:
        grid = _grid("CAT#", "A###", "R##X")
        placements = [Placement("CAT", 0, 0, "across"), Placement("CAR", 0, 0, "down")]
        problems = validate_layout(grid, placements)
        self.assertTrue(any("'X'" in p and "belongs to no word" in p for p in problems))

    def test_dot_cells_count_as_empty(self) -> None:
        grid = _grid("CAT", "A..", "R..")
        placements = [Placement("CAT", 0, 0, "across"), Placement("CAR", 0, 0, "down")]
        self.assertEqual(validate_layout(grid, placements), [])


class ValidatePuzzleTests(unittest.TestCase):
    def test_rejects_legacy_parallel_down_words(self) -> None:
        puzzle = {
            "grid_cells": _grid("ABC", "DEF", "GHI"),
            "across_entries": [],
            "down_entries": [
                {"answer": "ADG", "row": 0, "col": 0},
                {"answer": "BEH", "row": 0, "col": 1},
                {"answer": "CFI", "row": 0, "col": 2},
            ],
        }
        problems = validate_puzzle(puzzle)
        self.assertTrue(any("non-word across run" in p for p in problems))

    def test_accepts_valid_puzzle(self) -> None:
        puzzle = {
            "grid_cells": _grid("CAT", "A##", "R##"),
            "across_entries": [{"answer": "CAT", "row": 0, "col": 0}],
            "down_entries": [{"answer": "CAR", "row": 0, "col": 0}],
        }
        self.assertEqual(validate_puzzle(puzzle), [])


class NumberPlacementsTests(unittest.TestCase):
    def test_numbers_follow_row_major_starts(self) -> None:
        placements = [
            Placement("CAT", 0, 0, "across"),
            Placement("CAR", 0, 0, "down"),
            Placement("TOE", 0, 2, "down"),
            Placement("RUG", 2, 0, "across"),
        ]
        self.assertEqual(
            number_placements(placements),
            {(0, 0): 1, (0, 2): 2, (2, 0): 3},
        )


if __name__ == "__main__":
    unittest.main()
