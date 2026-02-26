import unittest

from src.clue_builder import (
    clue_pass_extract,
    clue_pass_extract_with_offset,
    clue_pass_mask_trim,
    clue_pass_validate,
    enforce_diversity,
    split_sentences,
)


class ClueBuilderTests(unittest.TestCase):
    def test_split_sentences(self) -> None:
        text = "First sentence. Second sentence! Third?"
        sentences = split_sentences(text)
        self.assertEqual(len(sentences), 3)

    def test_mask_and_validate(self) -> None:
        sentence = "Entropy is a measure of disorder in a system."
        clue = clue_pass_mask_trim(sentence, "Entropy", max_words=10)
        self.assertTrue(clue_pass_validate(clue, "Entropy", min_words=3, nlp=None))
        self.assertNotIn("Entropy", clue)

    def test_extract_sentence(self) -> None:
        text = "Heat is energy. Entropy is a measure."
        sentence = clue_pass_extract(text, "Entropy")
        self.assertEqual(sentence, "Entropy is a measure.")

    def test_extract_sentence_with_offset(self) -> None:
        text = "Heat is energy. Entropy is a measure."
        sentence, offset = clue_pass_extract_with_offset(text, "Entropy")
        self.assertEqual(sentence, "Entropy is a measure.")
        self.assertEqual(offset, 1)

    def test_diversity_enforcement_returns_tuple(self) -> None:
        clues = [
            {"answer": "A", "clue": "A measure of X"},
            {"answer": "B", "clue": "A measure of Y"},
            {"answer": "C", "clue": "A measure of Z"},
        ]
        kept, rejected = enforce_diversity(clues, max_per_bucket=2)
        self.assertEqual(len(kept), 2)
        self.assertIsInstance(rejected, list)
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0], "C")

    def test_diversity_rejection_preserves_answers(self) -> None:
        """Rejected answers should contain the answer field of dropped clues."""
        clues = [
            {"answer": "ALPHA", "clue": "First letter of the Greek alphabet"},
            {"answer": "BETA", "clue": "First letter in the second row"},
            {"answer": "GAMMA", "clue": "First letter that represents radiation"},
        ]
        kept, rejected = enforce_diversity(clues, max_per_bucket=2)
        # All three share the bucket "first letter of/in/that" but 3-word bucket is exact.
        # With different 3-word prefixes, they may all survive.  Force collision:
        clues2 = [
            {"answer": "ALPHA", "clue": "The standard measure of quality"},
            {"answer": "BETA", "clue": "The standard measure of price"},
            {"answer": "GAMMA", "clue": "The standard measure of value"},
        ]
        kept2, rejected2 = enforce_diversity(clues2, max_per_bucket=2)
        self.assertEqual(len(kept2), 2)
        self.assertEqual(len(rejected2), 1)
        self.assertIn(rejected2[0], ["ALPHA", "BETA", "GAMMA"])

    def test_morphological_leakage_detection(self) -> None:
        try:
            import spacy

            nlp = spacy.load("en_core_web_sm")
        except Exception:
            self.skipTest("spaCy model not available")
        clue = "Entropic processes increase disorder."
        self.assertFalse(clue_pass_validate(clue, "Entropy", min_words=2, nlp=nlp))

    def test_subject_rejection(self) -> None:
        try:
            import spacy

            nlp = spacy.load("en_core_web_sm")
        except Exception:
            self.skipTest("spaCy model not available")
        sentence = "Entropy increases with disorder."
        clue = clue_pass_mask_trim(sentence, "Entropy", max_words=10)
        self.assertFalse(
            clue_pass_validate(clue, "Entropy", min_words=2, nlp=nlp, original_sentence=sentence)
        )


if __name__ == "__main__":
    unittest.main()
