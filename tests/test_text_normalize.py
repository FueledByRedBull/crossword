import unittest

from src.text_normalize import enumeration, is_sensitive, tokenize


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


class SensitiveTests(unittest.TestCase):
    def test_english_stems_and_words_match(self) -> None:
        for text in ("Child pornography", "Rape during the war", "Paedophilia", "Sexual abuse", "Sexual slavery", "Revenge porn", "Incestuous", "Prostitution"):
            self.assertTrue(is_sensitive(text, "en"), text)

    def test_english_short_words_match_whole_words_only(self) -> None:
        for text in ("Rapeseed", "Grape", "Sextus Empiricus", "Sextant", "Sexual selection", "Ruin porn", "The Rape of the Sabine Women", "Slavery", "Genocide", "Execution", "Plague"):
            self.assertFalse(is_sensitive(text, "en"), text)

    def test_greek_matches_whatever_the_accent_case_or_ending(self) -> None:
        for text in ("Παιδική πορνογραφία", "ΠΟΡΝΟΓΡΑΦΙΑ", "βιασμός", "Σεξουαλικής κακοποίησης", "Αιμομιξία", "Πορνεία"):
            self.assertTrue(is_sensitive(text, "el"), text)
        for text in ("Βιασύνη", "Βιαστικός", "Γλυπτική", "Σεξτάντας", "Δουλεία", "Γενοκτονία"):
            self.assertFalse(is_sensitive(text, "el"), text)

    def test_a_sensitive_seed_title_turns_the_filter_off(self) -> None:
        self.assertTrue(is_sensitive("Prostitution in Rome", "en"))
        self.assertFalse(is_sensitive("Prostitution in Rome", "en", seed_title="Prostitution"))
        self.assertTrue(is_sensitive("Prostitution in Rome", "en", seed_title="Rome"))


if __name__ == "__main__":
    unittest.main()
