import unittest

from src.evaluation import summarize_benchmark, summarize_benchmark_collection

GOOD = {
    "seed": "A",
    "puzzle_status": "ok",
    "fill_status": "complete",
    "valid": True,
    "word_count": 14,
    "crossing_count": 18,
    "mean_crossings_per_word": 2.5,
    "two_plus_crossings_ratio": 0.8,
    "checked_letter_ratio": 0.4,
    "clue_count": 14,
    "definition_clue_ratio": 0.9,
    "gate_a_graded": True,
    "central_recall": 0.8,
    "off_theme_count": 0,
}


class EvaluationTests(unittest.TestCase):
    def test_summarize_benchmark_reads_layout_and_clue_fields(self) -> None:
        result = summarize_benchmark(
            {
                **GOOD,
                "seed": "Thermodynamics",
                "blank_missing_count": 1,
                "duplicate_clue_count": 2,
                "leakage_count": 1,
                "leakage_rate": 0.07,
                "used_clue_provenance_missing_count": 3,
            }
        )

        self.assertEqual(result.seed, "Thermodynamics")
        self.assertEqual(result.fill_status, "complete")
        self.assertTrue(result.valid)
        self.assertEqual(result.word_count, 14)
        self.assertEqual(result.crossing_count, 18)
        self.assertAlmostEqual(result.mean_crossings_per_word, 2.5)
        self.assertAlmostEqual(result.two_plus_crossings_ratio, 0.8)
        self.assertAlmostEqual(result.checked_letter_ratio, 0.4)
        self.assertAlmostEqual(result.definition_clue_ratio, 0.9)
        self.assertEqual(result.blank_missing_count, 1)
        self.assertEqual(result.duplicate_clue_count, 2)
        self.assertAlmostEqual(result.leakage_rate, 0.07)
        self.assertEqual(result.used_clue_provenance_missing_count, 3)

    def test_summarize_benchmark_collection_computes_aggregates(self) -> None:
        aggregate = summarize_benchmark_collection(
            [
                {**GOOD, "leakage_rate": 0.0},
                {
                    **GOOD,
                    "seed": "B",
                    "puzzle_status": "insufficient_quality",
                    "fill_status": "failed",
                    "valid": False,
                    "word_count": 6,
                    "crossing_count": 6,
                    "mean_crossings_per_word": 1.5,
                    "two_plus_crossings_ratio": 0.2,
                    "checked_letter_ratio": 0.2,
                    "definition_clue_ratio": 0.5,
                    "leakage_rate": 0.1,
                    "blank_missing_count": 1,
                    "duplicate_clue_count": 2,
                    "used_clue_provenance_missing_count": 2,
                },
            ]
        )

        self.assertEqual(aggregate.seed_count, 2)
        self.assertEqual(aggregate.fill_status_counts, {"complete": 1, "failed": 1})
        self.assertEqual(aggregate.puzzle_status_counts, {"ok": 1, "insufficient_quality": 1})
        self.assertAlmostEqual(aggregate.average_word_count, 10.0)
        self.assertAlmostEqual(aggregate.average_crossing_count, 12.0)
        self.assertAlmostEqual(aggregate.average_mean_crossings_per_word, 2.0)
        self.assertAlmostEqual(aggregate.average_two_plus_crossings_ratio, 0.5)
        self.assertAlmostEqual(aggregate.average_checked_letter_ratio, 0.3)
        self.assertAlmostEqual(aggregate.average_definition_clue_ratio, 0.7)
        self.assertAlmostEqual(aggregate.average_leakage_rate, 0.05)
        self.assertAlmostEqual(aggregate.average_used_clue_provenance_missing_count, 1.0)
        self.assertEqual(aggregate.blank_missing_count, 1)
        self.assertEqual(aggregate.duplicate_clue_count, 2)
        self.assertAlmostEqual(aggregate.valid_rate, 0.5)
        self.assertAlmostEqual(aggregate.fill_pass_rate, 0.5)
        self.assertAlmostEqual(aggregate.puzzle_ok_rate, 0.5)
        self.assertFalse(aggregate.release_candidate_pass)

    def test_gate_a_selection_quality(self) -> None:
        aggregate = summarize_benchmark_collection([GOOD, {**GOOD, "central_recall": 1.0}])
        self.assertTrue(aggregate.gate_a_pass)
        self.assertAlmostEqual(aggregate.average_central_recall, 0.9)
        self.assertEqual(aggregate.off_theme_total, 0)

        low_recall = summarize_benchmark_collection([GOOD, {**GOOD, "central_recall": 0.7}])
        self.assertFalse(low_recall.gate_a_pass)
        self.assertFalse(low_recall.release_candidate_pass)

        drift = summarize_benchmark_collection([GOOD, {**GOOD, "off_theme_count": 1}])
        self.assertFalse(drift.gate_a_pass)
        self.assertEqual(drift.off_theme_total, 1)
        self.assertFalse(drift.release_candidate_pass)

        no_fixture = summarize_benchmark_collection([GOOD, {**GOOD, "gate_a_graded": False}])
        self.assertFalse(no_fixture.gate_a_pass)
        self.assertFalse(no_fixture.release_candidate_pass)

        ungraded = {key: value for key, value in GOOD.items() if not key.startswith(("gate_a", "central", "off"))}
        self.assertFalse(summarize_benchmark_collection([ungraded]).gate_a_pass)
        self.assertFalse(summarize_benchmark_collection([]).gate_a_pass)

    def test_release_gates(self) -> None:
        passing = summarize_benchmark_collection([GOOD])
        self.assertTrue(passing.gate_a_pass and passing.gate_b_pass and passing.gate_c_pass and passing.gate_d_pass)
        self.assertTrue(passing.release_candidate_pass)

        not_ok = summarize_benchmark_collection([{**GOOD, "puzzle_status": "insufficient_clues"}])
        self.assertTrue(not_ok.gate_a_pass and not_ok.gate_b_pass and not_ok.gate_c_pass and not_ok.gate_d_pass)
        self.assertFalse(not_ok.release_candidate_pass)

        no_words = summarize_benchmark_collection([GOOD, {**GOOD, "word_count": 0}])
        self.assertFalse(no_words.gate_c_pass)
        self.assertFalse(no_words.gate_d_pass)
        self.assertFalse(no_words.release_candidate_pass)

        invalid = summarize_benchmark_collection([{**GOOD, "valid": False}])
        self.assertFalse(invalid.gate_b_pass)
        self.assertFalse(invalid.release_candidate_pass)

        duplicate = summarize_benchmark_collection([{**GOOD, "duplicate_clue_count": 1}])
        self.assertFalse(duplicate.gate_c_pass)
        self.assertTrue(duplicate.gate_d_pass)

        missing_blank = summarize_benchmark_collection([{**GOOD, "blank_missing_count": 1}])
        self.assertFalse(missing_blank.gate_c_pass)

        no_provenance = summarize_benchmark_collection([{**GOOD, "used_clue_provenance_missing_count": 1}])
        self.assertFalse(no_provenance.gate_d_pass)

        self.assertFalse(summarize_benchmark_collection([]).release_candidate_pass)


if __name__ == "__main__":
    unittest.main()
