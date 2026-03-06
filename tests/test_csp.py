import unittest
from importlib import import_module
from pathlib import Path
import shutil

from src.crossword_csp import build_slots, render_grid, solve_crossword
from src.csp_heuristics import build_solver_vocabulary

try:
    rust_csp = import_module("rust_csp")
except Exception:
    rust_csp = None


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

    def test_search_frontier_can_backtrack_to_earlier_branch(self) -> None:
        grid = [
            [".", "."],
            [".", "."],
        ]
        slots = build_slots(grid, min_len=2)
        words = ["AB", "AC", "AD", "AE", "BC", "BD", "EA", "EB"]
        result = solve_crossword(
            grid,
            slots,
            words,
            min_len=2,
            max_steps=100,
            use_ac3=False,
            beam_width=2,
            word_scores={"AC": 1.0},
        )

        self.assertTrue(result["solved"])
        self.assertEqual(len(result["assignments"]), len(slots))

    def test_build_solver_vocabulary_excludes_unpackageable_template_fallback(self) -> None:
        tmp_dir = Path("tests") / "tmp_outputs" / "csp_unpackageable_fallback"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        terms_path = tmp_dir / "answer_candidates.csv"
        clues_path = tmp_dir / "clues.csv"
        terms_path.write_text(
            "\n".join(
                [
                    "answer,normalized_answer,length,source_method,lead_bold_signal,source_titles,doc_frequency,theme_score,entity_type_score,crosswordability_score,lexicon_score,shape_penalty,answer_score",
                    "Alpha,ALPHA,5,spacy,False,Test,1,0.3,0.0,0.4,0.2,0.0,0.7",
                    "Beta,BETA,4,spacy,False,Test,1,0.3,0.0,0.4,0.2,0.0,0.6",
                ]
            ),
            encoding="utf-8",
        )
        clues_path.write_text(
            "\n".join(
                [
                    "answer,normalized_answer,clue,clue_score,clue_class,source_method,source_page,revid,sentence_offset,oldid_url",
                    "Alpha,ALPHA,Good clue,0.9,source_backed,spacy,Test,1,0,https://example.com/alpha",
                    "Beta,BETA,Fallback clue,0.1,template_fallback,title_tokens,Test,,-1,",
                ]
            ),
            encoding="utf-8",
        )

        vocabulary = build_solver_vocabulary(
            terms_path=terms_path,
            lang="en",
            filler_path=None,
            filler_min_len=3,
            filler_max_len=5,
            filler_max_per_length=0,
            filler_weight=0.01,
        )

        self.assertIn("ALPHA", vocabulary.clue_answers)
        self.assertNotIn("BETA", vocabulary.clue_answers)
        self.assertIn("BETA", vocabulary.unsupported_answers)

        shutil.rmtree(tmp_dir, ignore_errors=True)

    @unittest.skipIf(rust_csp is None, "rust_csp is not installed")
    def test_rust_solver_matches_python_on_branching_fixture(self) -> None:
        grid = [
            [".", "."],
            [".", "."],
        ]
        slots = build_slots(grid, min_len=2)
        words = ["AB", "AC", "AD", "AE", "BC", "BD", "EA", "EB"]
        kwargs = {
            "min_len": 2,
            "max_steps": 100,
            "use_ac3": False,
            "beam_width": 2,
            "word_scores": {"AC": 1.0},
        }

        python_result = solve_crossword(grid, slots, words, **kwargs)
        rust_result = rust_csp.solve_crossword(grid, slots, words, **kwargs)

        self.assertEqual(rust_result["solved"], python_result["solved"])
        self.assertEqual(len(rust_result["assignments"]), len(python_result["assignments"]))
        self.assertEqual(len(rust_result["assignments"]), len(slots))

    @unittest.skipIf(rust_csp is None, "rust_csp is not installed")
    def test_rust_solver_matches_python_on_scored_single_slot(self) -> None:
        grid = [
            [".", ".", "."],
            ["#", "#", "#"],
            ["#", "#", "#"],
        ]
        slots = build_slots(grid, min_len=3)
        words = ["CAT", "DOG"]
        kwargs = {
            "min_len": 3,
            "max_steps": 100,
            "use_ac3": False,
            "beam_width": 2,
            "word_scores": {"CAT": 0.1, "DOG": 0.9},
        }

        python_result = solve_crossword(grid, slots, words, **kwargs)
        rust_result = rust_csp.solve_crossword(grid, slots, words, **kwargs)

        self.assertEqual(rust_result["assignments"], python_result["assignments"])
        self.assertEqual(rust_result["steps"], python_result["steps"])


if __name__ == "__main__":
    unittest.main()
