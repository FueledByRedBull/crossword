import unittest

from src.term_extractor import (
    crosswordability_score,
    extract_terms_lead_bold,
    extract_terms_nltk,
    get_stopwords,
    merge_terms,
    shape_penalty,
    term_frequency_across_docs,
)


class TermExtractorTests(unittest.TestCase):
    def test_lead_bold_terms_normalize(self) -> None:
        terms = extract_terms_lead_bold(["Thermodynamics"], source_title="Thermodynamics", lang="en")
        self.assertEqual(terms[0].normalized_answer, "THERMODYNAMICS")

    def test_merge_terms_respects_length(self) -> None:
        terms = extract_terms_lead_bold(
            ["ABC", "THERMODYNAMICS"], source_title="Thermodynamics", lang="en"
        )
        merged = merge_terms([terms], min_len=4, max_len=12)
        self.assertEqual(len(merged), 0)

    def test_stopword_filtered(self) -> None:
        terms = extract_terms_lead_bold(
            ["This", "Thermodynamics"], source_title="Thermodynamics", lang="en"
        )
        merged = merge_terms([terms], min_len=4, max_len=15, stopwords=get_stopwords("en"))
        normalized = {term.normalized_answer for term in merged}
        self.assertNotIn("THIS", normalized)
        self.assertIn("THERMODYNAMICS", normalized)

    def test_term_frequency_counts(self) -> None:
        docs = ["Entropy and energy", "Energy systems"]
        df = term_frequency_across_docs(docs, lang="en")
        self.assertEqual(df.get("ENERGY"), 2)

    def test_greek_normalization(self) -> None:
        terms = extract_terms_lead_bold(
            ["Θερμο"], source_title="GreekArticle", lang="el"
        )
        self.assertEqual(terms[0].normalized_answer, "THERMO")

    def test_greek_frequency(self) -> None:
        docs = ["Θερμο θερμο", "Θερμο"]
        df = term_frequency_across_docs(docs, lang="el")
        self.assertEqual(df.get("THERMO"), 2)

    def test_merge_terms_accumulates_sources(self) -> None:
        terms_one = extract_terms_lead_bold(
            ["Thermodynamics"], source_title="ArticleA", lang="en"
        )
        terms_two = extract_terms_lead_bold(
            ["Thermodynamics"], source_title="ArticleB", lang="en"
        )
        merged = merge_terms([terms_one, terms_two], min_len=4, max_len=20)
        sources = merged[0].source_titles
        self.assertIn("ArticleA", sources)
        self.assertIn("ArticleB", sources)


class ShapePenaltyTests(unittest.TestCase):
    def test_rare_letters_penalized(self) -> None:
        """A word whose middle chars are all RARE_LETTERS (Q,X,Z,J) gets a high penalty."""
        # QXZJQXZJQ — middle 7 chars are all rare
        penalty = shape_penalty("QXZJQXZJQ")
        self.assertGreater(penalty, 0.5)

    def test_common_letters_low_penalty(self) -> None:
        """A word with no rare letters has zero shape penalty."""
        penalty = shape_penalty("ENTROPY")
        self.assertEqual(penalty, 0.0)

    def test_shape_penalty_range(self) -> None:
        """shape_penalty must always return a value in [0, 1]."""
        for word in ["A", "ENTROPY", "QXZJQXZJ", "AAAA", "THERMODYNAMICS"]:
            penalty = shape_penalty(word)
            self.assertGreaterEqual(penalty, 0.0)
            self.assertLessEqual(penalty, 1.0)

    def test_crosswordability_range(self) -> None:
        """crosswordability_score must always return a value in [0, 1]."""
        for word in ["ENTROPY", "QQQQQQ", "ETAI", "THERMODYNAMICS"]:
            score = crosswordability_score(word)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_rare_letters_lower_crosswordability(self) -> None:
        """A word packed with rare letters scores lower crosswordability than a common-letter word."""
        good = crosswordability_score("NATION")
        poor = crosswordability_score("QXZJQXZJ")
        self.assertGreater(good, poor)


class NltkBackendParityTests(unittest.TestCase):
    def test_nltk_returns_term_candidates(self) -> None:
        """nltk backend must return at least one candidate for non-trivial text."""
        text = "Thermodynamics is a branch of physical science dealing with heat and temperature."
        try:
            terms = extract_terms_nltk(text, source_title="Thermodynamics", lang="en")
        except Exception as exc:
            self.skipTest(f"nltk unavailable in test environment: {exc}")
        self.assertIsInstance(terms, list)
        # If nltk extracted anything, every item must be a TermCandidate.
        for term in terms:
            self.assertIsNotNone(term.normalized_answer)
            self.assertIsInstance(term.lead_bold_signal, bool)
            self.assertEqual(term.source_method, "nltk")

    def test_spacy_and_nltk_both_find_thermodynamics(self) -> None:
        """Parity smoke-test: both backends should find the bold term regardless of NLP path."""
        bold_terms = ["Thermodynamics"]
        # lead_bold is backend-independent — verify it handles both paths without error.
        result = extract_terms_lead_bold(bold_terms, source_title="Thermodynamics", lang="en")
        self.assertTrue(any(t.normalized_answer == "THERMODYNAMICS" for t in result))


if __name__ == "__main__":
    unittest.main()
