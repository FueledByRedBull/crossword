import unittest

from src.text_normalize import enumeration, tokenize


class EnumerationTests(unittest.TestCase):
    def test_single_word(self) -> None:
        self.assertEqual(enumeration("Bebop"), "5")

    def test_words_are_comma_separated(self) -> None:
        self.assertEqual(enumeration("Miles Davis"), "5,5")
        self.assertEqual(enumeration("Rock and roll"), "4,3,4")

    def test_hyphen_is_kept(self) -> None:
        self.assertEqual(enumeration("Synth-pop"), "5-3")

    def test_punctuation_is_not_counted(self) -> None:
        self.assertEqual(enumeration("Rock 'n' roll"), "4,1,4")

    def test_apostrophe_inside_a_word_joins_it(self) -> None:
        self.assertEqual(enumeration("Bell's theorem"), "5,7")
        self.assertEqual(enumeration("Anita O'Day"), "5,4")


class GreekTests(unittest.TestCase):
    def test_enumeration_counts_greek_letters(self) -> None:
        self.assertEqual(enumeration("Θερμότητα"), "9")
        self.assertEqual(enumeration("Πρώτος θερμοδυναμικός νόμος"), "6,14,5")

    def test_tokens_are_lowercase_without_accents(self) -> None:
        self.assertEqual(tokenize("Ο Ήρωνας Αϊνστάιν", lang="el"), ["ο", "ηρωνας", "αινσταιν"])


if __name__ == "__main__":
    unittest.main()
