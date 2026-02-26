import json
import os
import shutil
import unittest
from pathlib import Path

from src.pipeline import run_generate_pipeline


class GenerateIntegrationTests(unittest.TestCase):
    def test_generate_pipeline_offline_cached(self) -> None:
        cache_dir = Path("data") / "cache" / "wiki"
        if not cache_dir.exists() or not any(cache_dir.rglob("*.json")):
            self.skipTest("wiki cache missing; run a benchmark once to populate cache")

        previous_offline = os.environ.get("CROSSWORD_OFFLINE")
        os.environ["CROSSWORD_OFFLINE"] = "1"
        out_dir = Path("tests") / "tmp_outputs" / "generate_integration"
        if out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)

        try:
            result = run_generate_pipeline(
                seed_title="Thermodynamics",
                lang="en",
                output_dir=out_dir,
                cache_dir=cache_dir,
                max_links=200,
                max_backlinks=200,
                expansion="one_hop_only",
                max_candidates=400,
                gate_max=250,
                max_steps=3000,
                max_restarts=1,
                template_trials=1,
                beam_width=16,
                filler_max_per_length=1000,
            )
        finally:
            if previous_offline is None:
                os.environ.pop("CROSSWORD_OFFLINE", None)
            else:
                os.environ["CROSSWORD_OFFLINE"] = previous_offline

        puzzle_path = Path(result["package"].puzzle_path)
        self.assertTrue(puzzle_path.exists())
        payload = json.loads(puzzle_path.read_text(encoding="utf-8"))
        self.assertIn("fill_status", payload)


if __name__ == "__main__":
    unittest.main()
