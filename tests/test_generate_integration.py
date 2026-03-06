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
                long_slot_theme_values: list[float] = []
                used_source_backed_values: list[float] = []
                used_template_fallback_values: list[float] = []
                packaged_synthetic_filler_counts: list[int] = []
                used_provenance_missing_counts: list[int] = []
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
                        package_diagnostics = result["package"].diagnostics
                        fill_percent = float(payload.get("fill_percent", 0.0))
                        filler_ratio = float((solve_diagnostics.get("filler") or {}).get("used_ratio", 0.0))
                        long_slot_theme_ratio = float(solve_diagnostics.get("long_slot_theme_ratio", 0.0))
                        used_source_backed_ratio = float(
                            package_diagnostics.get(
                                "used_source_backed_entry_ratio",
                                package_diagnostics.get("source_backed_entry_ratio", 0.0),
                            )
                        )
                        used_template_fallback_ratio = float(
                            package_diagnostics.get(
                                "used_template_fallback_entry_ratio",
                                package_diagnostics.get("fallback_only_entry_ratio", 0.0),
                            )
                        )
                        packaged_synthetic_filler_count = int(
                            package_diagnostics.get(
                                "packaged_synthetic_filler_count",
                                package_diagnostics.get("synthetic_filler_clue_count", 0),
                            )
                        )
                        used_provenance_missing_count = int(
                            package_diagnostics.get(
                                "used_clue_provenance_missing_count",
                                package_diagnostics.get("provenance_missing_count", 0),
                            )
                        )
                        fill_values.append(fill_percent)
                        filler_values.append(filler_ratio)
                        long_slot_theme_values.append(long_slot_theme_ratio)
                        used_source_backed_values.append(used_source_backed_ratio)
                        used_template_fallback_values.append(used_template_fallback_ratio)
                        packaged_synthetic_filler_counts.append(packaged_synthetic_filler_count)
                        used_provenance_missing_counts.append(used_provenance_missing_count)
                        self.assertEqual(solve_diagnostics.get("solver_backend"), backend["name"])
                        self.assertIn(payload.get("fill_status"), {"partial", "complete"})
                        self.assertGreaterEqual(fill_percent, 0.70)
                        self.assertEqual(package_diagnostics.get("puzzle_status"), "ok")
                        self.assertEqual(used_provenance_missing_count, 0)
                        self.assertEqual(packaged_synthetic_filler_count, 0)

                average_fill = sum(fill_values) / len(fill_values)
                average_filler = sum(filler_values) / len(filler_values)
                average_long_slot_theme = sum(long_slot_theme_values) / len(long_slot_theme_values)
                average_used_source_backed = sum(used_source_backed_values) / len(used_source_backed_values)
                average_used_template_fallback = sum(used_template_fallback_values) / len(used_template_fallback_values)
                average_packaged_synthetic_filler = sum(packaged_synthetic_filler_counts) / len(
                    packaged_synthetic_filler_counts
                )
                average_used_provenance_missing = sum(used_provenance_missing_counts) / len(
                    used_provenance_missing_counts
                )
                self.assertGreaterEqual(average_fill, 0.73, backend["name"])
                self.assertLessEqual(average_filler, 0.10, backend["name"])
                self.assertGreaterEqual(average_long_slot_theme, 0.95, backend["name"])
                self.assertGreaterEqual(average_used_source_backed, 0.90, backend["name"])
                self.assertLessEqual(average_used_template_fallback, 0.10, backend["name"])
                self.assertEqual(average_packaged_synthetic_filler, 0.0, backend["name"])
                self.assertEqual(average_used_provenance_missing, 0.0, backend["name"])
        finally:
            if previous_offline is None:
                os.environ.pop("CROSSWORD_OFFLINE", None)
            else:
                os.environ["CROSSWORD_OFFLINE"] = previous_offline


if __name__ == "__main__":
    unittest.main()
