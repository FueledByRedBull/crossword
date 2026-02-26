import unittest

from src.semantic import build_pairwise_matrix, build_tfidf_vectors, cosine_similarity


class SemanticTests(unittest.TestCase):
    def test_cosine_similarity_identity(self) -> None:
        vectors = build_tfidf_vectors(
            ["alpha beta gamma", "alpha beta gamma"], lang="en"
        )
        sim = cosine_similarity(vectors[0], vectors[1])
        self.assertGreater(sim, 0.9)

    def test_cosine_similarity_difference(self) -> None:
        vectors = build_tfidf_vectors(
            ["alpha beta gamma", "delta epsilon"], lang="en"
        )
        sim = cosine_similarity(vectors[0], vectors[1])
        self.assertLess(sim, 0.5)

    def test_pairwise_matrix_symmetry(self) -> None:
        vectors = build_tfidf_vectors(
            ["alpha beta", "beta gamma", "gamma delta"], lang="en"
        )
        matrix = build_pairwise_matrix(vectors)
        self.assertEqual(len(matrix), 3)
        self.assertEqual(matrix[0][1], matrix[1][0])
        self.assertEqual(matrix[1][2], matrix[2][1])


if __name__ == "__main__":
    unittest.main()

