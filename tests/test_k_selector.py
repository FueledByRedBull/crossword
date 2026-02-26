import unittest

from src.k_selector import select_k


class KSelectorTests(unittest.TestCase):
    def test_select_k_stops_on_plateau(self) -> None:
        rel_scores = [0.9, 0.85, 0.83, 0.82, 0.81]
        size = len(rel_scores)
        pairwise = [[1.0 if i == j else 0.9 for j in range(size)] for i in range(size)]
        ranked = list(range(size))
        result = select_k(
            ranked_indices=ranked,
            rel_scores=rel_scores,
            pairwise=pairwise,
            min_k=2,
            max_k=5,
            epsilon=0.05,
            m=1,
        )
        self.assertGreaterEqual(result.selected_k, 2)
        self.assertLess(result.selected_k, 5)


if __name__ == "__main__":
    unittest.main()
