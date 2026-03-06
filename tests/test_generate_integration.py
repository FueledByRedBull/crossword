import json
import os
import shutil
import unittest
from importlib import import_module
from pathlib import Path

from src.pipeline import run_generate_pipeline

try:
    rust_csp = import_module("rust_csp")
except Exception:
    rust_csp = None


class GenerateIntegrationTests(unittest.TestCase):
    def test_generate_pipeline_offline_cached_seed_corpus(self) -> None:
        cache_dir = Path("data") / "cache" / "wiki"
        if not cache_dir.exists() or not any(cache_dir.rglob("*.json")):
            self.skipTest("wiki cache missing; run a benchmark once to populate cache")

        previous_offline = os.environ.get("CROSSWORD_OFFLINE")
        os.environ["CROSSWORD_OFFLINE"] = "1"
        backend_configs = [{"name": "python", "use_rust": False}]
        if rust_csp is not None:
            backend_configs.append({"name": "rust", "use_rust": True})
        seed_configs = [
            {
                "seed": "Thermodynamics",
                "kwargs": {
                    "expansion": "one_hop_only",
                    "gate_max": 250,
                    "max_steps": 6000,
                    "max_restarts": 1,
                    "template_trials": 2,
                    "beam_width": 16,
                    "filler_max_per_length": 1000,
                },
            },
            {
                "seed": "Quantum mechanics",
                "kwargs": {
                    "expansion": "one_hop_only",
                    "gate_max": 250,
                    "max_steps": 6000,
                    "max_restarts": 1,
                    "template_trials": 2,
                    "beam_width": 16,
                    "filler_max_per_length": 1000,
                },
            },
            {
                "seed": "Jazz",
                "kwargs": {
                    "max_links": 200,
                    "max_backlinks": 200,
                    "max_candidates": 400,
                    "expansion": "one_hop_only",
                    "gate_max": 250,
                    "max_steps": 6000,
                    "max_restarts": 1,
                    "template_trials": 3,
                    "beam_width": 16,
                    "filler_max_per_length": 1000,
                },
            },
            {
                "seed": "Ancient Rome",
                "kwargs": {
                    "max_links": 200,
                    "max_backlinks": 200,
                    "max_candidates": 400,
                    "expansion": "one_hop_only",
                    "gate_max": 250,
                    "max_steps": 6000,
                    "max_restarts": 1,
                    "template_trials": 2,
                    "beam_width": 16,
                    "filler_max_per_length": 1000,
                },
            },
        ]

        try:
            for backend in backend_configs:
                fill_values: list[float] = []
                filler_values: list[float] = []
                for config in seed_configs:
                    seed = config["seed"]
                    slug = seed.lower().replace(" ", "_")
                    out_dir = Path("tests") / "tmp_outputs" / f"generate_integration_{slug}_{backend['name']}"
                    if out_dir.exists():
                        shutil.rmtree(out_dir, ignore_errors=True)

                    with self.subTest(seed=seed, backend=backend["name"]):
                        result = run_generate_pipeline(
                            seed_title=seed,
                            lang="en",
                            output_dir=out_dir,
                            cache_dir=cache_dir,
                            use_rust=backend["use_rust"],
                            **config["kwargs"],
                        )
                        puzzle_path = Path(result["package"].puzzle_path)
                        self.assertTrue(puzzle_path.exists())
                        payload = json.loads(puzzle_path.read_text(encoding="utf-8"))
                        solve_diagnostics = result["solve"].diagnostics
                        fill_percent = float(payload.get("fill_percent", 0.0))
                        filler_ratio = float((solve_diagnostics.get("filler") or {}).get("used_ratio", 0.0))
                        fill_values.append(fill_percent)
                        filler_values.append(filler_ratio)
                        self.assertEqual(solve_diagnostics.get("solver_backend"), backend["name"])
                        self.assertIn(payload.get("fill_status"), {"partial", "complete"})
                        self.assertGreaterEqual(fill_percent, 0.70)
                        self.assertEqual(result["package"].diagnostics.get("puzzle_status"), "ok")

                average_fill = sum(fill_values) / len(fill_values)
                average_filler = sum(filler_values) / len(filler_values)
                self.assertGreaterEqual(average_fill, 0.73, backend["name"])
                self.assertLessEqual(average_filler, 0.10, backend["name"])
        finally:
            if previous_offline is None:
                os.environ.pop("CROSSWORD_OFFLINE", None)
            else:
                os.environ["CROSSWORD_OFFLINE"] = previous_offline


if __name__ == "__main__":
    unittest.main()
