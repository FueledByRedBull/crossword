import json
import os
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from src.clue_builder import BLANK, MAX_CLUE_WORDS, answer_pattern
from src.layout import validate_puzzle
from src.pipeline import run_generate_pipeline


class GenerateIntegrationTests(unittest.TestCase):
    def test_generate_pipeline_offline_cached_seed_corpus(self) -> None:
        cache_dir = Path("data") / "cache" / "wiki"
        if not cache_dir.exists() or not any(cache_dir.rglob("*.json")):
            self.skipTest("wiki cache missing; run a benchmark once to populate cache")

        # Ancient Rome links to 2,193 pages; the cache holds the evenly spaced 200-link sample.
        seed_configs = {
            "Thermodynamics": {},
            "Quantum mechanics": {},
            "Jazz": {},
            "Ancient Rome": {"max_links": 200},
        }

        with patch.dict(os.environ, {"CROSSWORD_OFFLINE": "1"}):
            for seed, kwargs in seed_configs.items():
                slug = seed.lower().replace(" ", "_")
                out_dir = Path("tests") / "tmp_outputs" / f"generate_integration_{slug}"
                if out_dir.exists():
                    shutil.rmtree(out_dir, ignore_errors=True)

                with self.subTest(seed=seed):
                    result = run_generate_pipeline(
                        seed_title=seed,
                        lang="en",
                        output_dir=out_dir,
                        cache_dir=cache_dir,
                        **kwargs,
                    )
                    puzzle = json.loads(Path(result["package"].puzzle_path).read_text(encoding="utf-8"))
                    self.assertEqual(puzzle["fill_status"], "complete")
                    self.assertEqual(puzzle["puzzle_status"], "ok")
                    self.assertEqual(validate_puzzle(puzzle), [])

                    # Floors frozen from the full run of 2026-10-11 on the refreshed cache (weakest seed in brackets).
                    layout = puzzle["layout"]
                    self.assertGreaterEqual(layout["word_count"], 17)  # [19]
                    self.assertGreaterEqual(layout["mean_crossings_per_word"], 2.1)  # [2.32]
                    self.assertGreaterEqual(layout["two_plus_crossings_ratio"], 0.6)  # [0.74]
                    self.assertGreaterEqual(layout["checked_letter_ratio"], 0.17)  # [0.190]

                    entries = puzzle["across_entries"] + puzzle["down_entries"]
                    definitions = [entry for entry in entries if entry["clue_style"] == "definition"]
                    self.assertGreaterEqual(len(definitions) / len(entries), 0.5)  # [0.53]
                    for entry in entries:
                        self.assertLessEqual(len([w for w in entry["clue"].split() if w != "..."]), MAX_CLUE_WORDS, entry)
                        self.assertFalse(entry["clue"].startswith("..."), entry)
                        self.assertIsNone(answer_pattern(entry["display_answer"]).search(entry["clue"]), entry)
                        if entry["clue_style"] == "fill_in_blank":
                            self.assertIn(BLANK, entry["clue"])
                    self.assertEqual(puzzle["diagnostics"]["duplicate_clue_text_count"], 0)
                    self.assertEqual(puzzle["diagnostics"]["used_clue_provenance_missing_count"], 0)

    def test_generate_pipeline_offline_uncached_seed_packages_insufficient_vocabulary(self) -> None:
        out_dir = Path("tests") / "tmp_outputs" / "generate_integration_uncached_seed"
        if out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)

        with patch.dict(os.environ, {"CROSSWORD_OFFLINE": "1"}):
            result = run_generate_pipeline(
                seed_title="Θερμοδυναμική",
                lang="el",
                output_dir=out_dir,
                cache_dir=out_dir / "cache",
            )

        payload = json.loads(Path(result["package"].puzzle_path).read_text(encoding="utf-8"))
        self.assertEqual(payload.get("lang"), "el")
        self.assertEqual(payload.get("fill_status"), "failed")
        self.assertEqual(result["package"].diagnostics.get("puzzle_status"), "insufficient_vocabulary")
        self.assertEqual(result["layout"].diagnostics.get("fill_status"), "failed")
        self.assertTrue((out_dir / "grid.json").exists())
        self.assertTrue((out_dir / "diagnostics_layout.json").exists())


if __name__ == "__main__":
    unittest.main()
