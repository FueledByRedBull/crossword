import unittest

from src.vocab_gate import evaluate_vocab_gate


class VocabGateTests(unittest.TestCase):
    def test_minimum_only_by_default(self) -> None:
        self.assertFalse(evaluate_vocab_gate(29).passed)
        self.assertEqual(evaluate_vocab_gate(29).reason, "insufficient_terms")
        self.assertTrue(evaluate_vocab_gate(30).passed)
        self.assertTrue(evaluate_vocab_gate(5000).passed)

    def test_explicit_maximum_is_still_enforced(self) -> None:
        result = evaluate_vocab_gate(300, min_required=40, max_allowed=250)
        self.assertFalse(result.passed)
        self.assertEqual(result.reason, "too_many_terms")


if __name__ == "__main__":
    unittest.main()
