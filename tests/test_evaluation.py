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
                "long_slot_theme_ratio": 1.0,
                "quality_objective": 1.42,
                "synthetic_filler_clue_count": 2,
            }
        )

        self.assertEqual(result.seed, "Thermodynamics")
        self.assertEqual(result.solver_backend, "rust")
        self.assertEqual(result.puzzle_status, "ok")
        self.assertAlmostEqual(result.fill_percent, 0.74)
        self.assertAlmostEqual(result.filler_used_ratio, 0.05)
        self.assertAlmostEqual(result.long_slot_theme_ratio, 1.0)
        self.assertEqual(result.synthetic_filler_clue_count, 2)

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
                    "long_slot_theme_ratio": 1.0,
                    "leakage_rate": 0.0,
                },
                {
                    "seed": "B",
                    "solver_backend": "rust",
                    "puzzle_status": "insufficient_quality",
                    "fill_status": "failed",
                    "fill_percent": 0.60,
                    "filler_used_ratio": 0.20,
                    "clued_entry_ratio": 0.8,
                    "long_slot_theme_ratio": 0.5,
                    "leakage_rate": 0.1,
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
        self.assertAlmostEqual(aggregate.fill_pass_rate, 0.5)
        self.assertAlmostEqual(aggregate.puzzle_ok_rate, 0.5)


if __name__ == "__main__":
    unittest.main()
