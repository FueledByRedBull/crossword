import unittest

from src.evaluation import summarize_benchmark, summarize_benchmark_collection


class EvaluationTests(unittest.TestCase):
    def test_summarize_benchmark_reads_quality_fields(self) -> None:
        result = summarize_benchmark(
            {
                "seed": "Thermodynamics",
                "solver_backend": "rust",
                "puzzle_status": "ok",
                "fill_status": "partial",
                "fill_percent": 0.74,
                "filler_used_ratio": 0.05,
                "clued_entry_ratio": 1.0,
                "source_backed_entry_ratio": 0.9,
                "used_source_backed_entry_ratio": 0.95,
                "fallback_only_entry_count": 1,
                "fallback_only_entry_ratio": 0.1,
                "used_template_fallback_entry_ratio": 0.05,
                "long_slot_theme_ratio": 1.0,
                "quality_objective": 1.42,
                "used_clue_provenance_missing_count": 0,
                "synthetic_filler_clue_count": 2,
                "packaged_synthetic_filler_count": 0,
                "preferred_fill_target": 0.85,
            }
        )

        self.assertEqual(result.seed, "Thermodynamics")
        self.assertEqual(result.solver_backend, "rust")
        self.assertEqual(result.puzzle_status, "ok")
        self.assertAlmostEqual(result.fill_percent, 0.74)
        self.assertAlmostEqual(result.filler_used_ratio, 0.05)
        self.assertAlmostEqual(result.source_backed_entry_ratio, 0.9)
        self.assertAlmostEqual(result.used_source_backed_entry_ratio, 0.95)
        self.assertEqual(result.fallback_only_entry_count, 1)
        self.assertAlmostEqual(result.used_template_fallback_entry_ratio, 0.05)
        self.assertAlmostEqual(result.long_slot_theme_ratio, 1.0)
        self.assertEqual(result.used_clue_provenance_missing_count, 0)
        self.assertEqual(result.synthetic_filler_clue_count, 2)
        self.assertEqual(result.packaged_synthetic_filler_count, 0)
        self.assertAlmostEqual(result.preferred_fill_target, 0.85)

    def test_summarize_benchmark_collection_computes_quality_aggregates(self) -> None:
        aggregate = summarize_benchmark_collection(
            [
                {
                    "seed": "A",
                    "solver_backend": "python",
                    "puzzle_status": "ok",
                    "fill_status": "partial",
                    "fill_percent": 0.72,
                    "filler_used_ratio": 0.10,
                    "clued_entry_ratio": 1.0,
                    "source_backed_entry_ratio": 0.8,
                    "used_source_backed_entry_ratio": 0.9,
                    "fallback_only_entry_count": 1,
                    "fallback_only_entry_ratio": 0.2,
                    "used_template_fallback_entry_ratio": 0.1,
                    "long_slot_theme_ratio": 1.0,
                    "leakage_rate": 0.0,
                    "used_clue_provenance_missing_count": 0,
                    "synthetic_filler_clue_count": 0,
                    "packaged_synthetic_filler_count": 0,
                },
                {
                    "seed": "B",
                    "solver_backend": "rust",
                    "puzzle_status": "insufficient_quality",
                    "fill_status": "failed",
                    "fill_percent": 0.60,
                    "filler_used_ratio": 0.20,
                    "clued_entry_ratio": 0.8,
                    "source_backed_entry_ratio": 0.3,
                    "used_source_backed_entry_ratio": 0.4,
                    "fallback_only_entry_count": 2,
                    "fallback_only_entry_ratio": 0.5,
                    "used_template_fallback_entry_ratio": 0.3,
                    "long_slot_theme_ratio": 0.5,
                    "leakage_rate": 0.1,
                    "used_clue_provenance_missing_count": 2,
                    "synthetic_filler_clue_count": 1,
                    "packaged_synthetic_filler_count": 0,
                },
            ]
        )

        self.assertEqual(aggregate.seed_count, 2)
        self.assertEqual(aggregate.backend_counts, {"python": 1, "rust": 1})
        self.assertEqual(aggregate.fill_status_counts, {"partial": 1, "failed": 1})
        self.assertAlmostEqual(aggregate.average_fill_percent, 0.66)
        self.assertAlmostEqual(aggregate.average_filler_used_ratio, 0.15)
        self.assertAlmostEqual(aggregate.average_long_slot_theme_ratio, 0.75)
        self.assertAlmostEqual(aggregate.average_clued_entry_ratio, 0.9)
        self.assertAlmostEqual(aggregate.average_source_backed_entry_ratio, 0.55)
        self.assertAlmostEqual(aggregate.average_fallback_only_entry_ratio, 0.35)
        self.assertAlmostEqual(aggregate.average_used_source_backed_entry_ratio, 0.65)
        self.assertAlmostEqual(aggregate.average_used_template_fallback_entry_ratio, 0.2)
        self.assertAlmostEqual(aggregate.average_synthetic_filler_clue_count, 0.5)
        self.assertAlmostEqual(aggregate.average_packaged_synthetic_filler_count, 0.0)
        self.assertAlmostEqual(aggregate.average_used_clue_provenance_missing_count, 1.0)
        self.assertAlmostEqual(aggregate.fill_pass_rate, 0.5)
        self.assertAlmostEqual(aggregate.puzzle_ok_rate, 0.5)


if __name__ == "__main__":
    unittest.main()
