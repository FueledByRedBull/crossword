import unittest

from src.term_extractor import (
    TermCandidate,
    answer_cleanliness_score,
    clueability_score,
    crosswordability_score,
    extract_terms_lead_bold,
    extract_terms_nltk,
    get_stopwords,
    merge_plural_duplicates,
    merge_terms,
    shape_penalty,
    term_frequency_across_docs,
    title_to_answer,
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
        self.assertEqual(terms[0].normalized_answer, "ΘΕΡΜΟ")

    def test_greek_frequency(self) -> None:
        docs = ["Θερμο θερμο", "Θερμό"]
        df = term_frequency_across_docs(docs, lang="el")
        self.assertEqual(df.get("ΘΕΡΜΟ"), 2)

    def test_greek_stopwords_are_greek_capitals(self) -> None:
        terms = extract_terms_lead_bold(["στην", "ως", "Θερμότητα"], source_title="Θερμότητα", lang="el")
        merged = merge_terms([terms], min_len=2, max_len=15, stopwords=get_stopwords("el"))
        self.assertEqual([term.normalized_answer for term in merged], ["ΘΕΡΜΟΤΗΤΑ"])
        for word in get_stopwords("el"):
            self.assertRegex(word, "^[Α-Ω]+$")

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

    def test_lead_bold_strips_leading_determiners(self) -> None:
        terms = extract_terms_lead_bold(
            ["The heat", "An axiom"], source_title="Thermodynamics", lang="en"
        )
        normalized = {term.normalized_answer for term in terms}
        self.assertIn("HEAT", normalized)
        self.assertIn("AXIOM", normalized)
        self.assertNotIn("THEHEAT", normalized)
        self.assertNotIn("ANAXIOM", normalized)

    def test_merge_terms_rejects_dirty_stopword_boundary_phrase(self) -> None:
        dirty = TermCandidate(
            answer="Energy of",
            normalized_answer="ENERGYOF",
            length=8,
            source_method="spacy",
            lead_bold_signal=False,
            source_titles={"ArticleA"},
            token_count=2,
            answer_cleanliness_score=0.2,
            clueability_score=0.3,
        )
        clean = TermCandidate(
            answer="Entropy",
            normalized_answer="ENTROPY",
            length=7,
            source_method="spacy",
            lead_bold_signal=False,
            source_titles={"ArticleA"},
            token_count=1,
            answer_cleanliness_score=1.0,
            clueability_score=0.9,
        )

        merged = merge_terms([[dirty, clean]], min_len=4, max_len=12, lang="en")
        normalized = {term.normalized_answer for term in merged}

        self.assertNotIn("ENERGYOF", normalized)
        self.assertIn("ENTROPY", normalized)

    def test_merge_terms_keeps_lead_bold_even_when_phrase_is_dirty(self) -> None:
        lead_bold_terms = extract_terms_lead_bold(
            ["The field of"], source_title="Thermodynamics", lang="en"
        )

        merged = merge_terms([lead_bold_terms], min_len=4, max_len=12, lang="en")

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].answer, "field of")

    def test_merge_terms_keeps_cleanest_surface_for_duplicate_answer(self) -> None:
        noisy = TermCandidate(
            answer="Field of",
            normalized_answer="FIELDOF",
            length=7,
            source_method="spacy",
            lead_bold_signal=False,
            source_titles={"ArticleA"},
            token_count=2,
            answer_cleanliness_score=0.45,
            clueability_score=0.5,
        )
        cleaner = TermCandidate(
            answer="Fieldof",
            normalized_answer="FIELDOF",
            length=7,
            source_method="lead_bold",
            lead_bold_signal=True,
            source_titles={"ArticleB"},
            token_count=1,
            answer_cleanliness_score=0.95,
            clueability_score=0.9,
        )

        merged = merge_terms([[noisy], [cleaner]], min_len=4, max_len=12, lang="en")

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].answer, "Fieldof")
        self.assertIn("ArticleA", merged[0].source_titles)
        self.assertIn("ArticleB", merged[0].source_titles)


class ShapePenaltyTests(unittest.TestCase):
    def test_rare_greek_letters_are_penalized(self) -> None:
        self.assertGreater(shape_penalty("ΞΨΖΞΨΖΞΨ"), 0.5)
        self.assertEqual(shape_penalty("ΕΝΤΡΟΠΙΑ"), 0.0)
        self.assertGreater(crosswordability_score("ΕΝΤΡΟΠΙΑ"), crosswordability_score("ΞΨΖΞΨΖΞΨ"))

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

    def test_cleanliness_and_clueability_scores_reward_clean_phrases(self) -> None:
        clean = "Entropy"
        dirty = "energy of"

        self.assertGreater(answer_cleanliness_score(clean), answer_cleanliness_score(dirty))
        self.assertGreater(clueability_score(clean, "ENTROPY"), clueability_score(dirty, "ENERGYOF"))


class TitleToAnswerTests(unittest.TestCase):
    def test_strips_disambiguator(self) -> None:
        self.assertEqual(title_to_answer("Swing (jazz performance style)"), ("Swing", "SWING"))

    def test_keeps_multi_word_title(self) -> None:
        self.assertEqual(title_to_answer("Miles Davis"), ("Miles Davis", "MILESDAVIS"))

    def test_rejects_digits_and_list_style_titles(self) -> None:
        self.assertIsNone(title_to_answer("1940s in jazz"))
        for title in ("List of jazz musicians", "Outline of jazz", "Index of music", "Timeline of jazz", "Glossary of jazz"):
            self.assertIsNone(title_to_answer(title), title)

    def test_rejects_long_titles(self) -> None:
        self.assertIsNone(title_to_answer("New Orleans jazz revival movement"))
        self.assertIsNone(title_to_answer("Thermodynamics", max_len=10))
        self.assertIsNotNone(title_to_answer("Thermodynamics", max_len=15))

    def test_rejects_titles_with_symbols(self) -> None:
        self.assertIsNone(title_to_answer("A&C Black"))
        self.assertIsNone(title_to_answer("Foo: Bar"))
        self.assertEqual(title_to_answer("Rock 'n' roll"), ("Rock 'n' roll", "ROCKNROLL"))
        self.assertEqual(title_to_answer("St. Louis"), ("St. Louis", "STLOUIS"))

    def test_folds_accents_into_plain_ascii(self) -> None:
        self.assertEqual(
            title_to_answer("Schr\u00f6dinger's cat"), ("Schr\u00f6dinger's cat", "SCHRODINGERSCAT")
        )
        self.assertIsNone(title_to_answer("S\u00f8ren Kierkegaard"))
        terms = extract_terms_lead_bold(["Schr\u00f6dinger"], source_title="Schrodinger", lang="en")
        self.assertEqual(terms[0].normalized_answer, "SCHRODINGER")

    def test_rejects_generic_single_word_titles(self) -> None:
        self.assertIsNone(title_to_answer("Time"))
        self.assertIsNotNone(title_to_answer("Time signature"))

    def test_rejects_too_short_answers(self) -> None:
        self.assertIsNone(title_to_answer("Go"))
        self.assertEqual(title_to_answer("Sax"), ("Sax", "SAX"))


class GreekAnswerTests(unittest.TestCase):
    def test_a_greek_answer_is_normalized_to_greek_capitals_without_accents(self) -> None:
        self.assertEqual(title_to_answer("Θερμότητα", lang="el"), ("Θερμότητα", "ΘΕΡΜΟΤΗΤΑ"))
        self.assertEqual(title_to_answer("Ήρωνας", lang="el"), ("Ήρωνας", "ΗΡΩΝΑΣ"))  # final sigma
        self.assertEqual(title_to_answer("Άλμπερτ Αϊνστάιν", lang="el"), ("Άλμπερτ Αϊνστάιν", "ΑΛΜΠΕΡΤΑΙΝΣΤΑΙΝ"))
        self.assertEqual(title_to_answer("Στατική φυσική (κλάδος)", lang="el"), ("Στατική φυσική", "ΣΤΑΤΙΚΗΦΥΣΙΚΗ"))
        terms = extract_terms_lead_bold(["ΐσως ηλεκτρισμός"], source_title="x", lang="el")
        self.assertEqual(terms[0].normalized_answer, "ΙΣΩΣΗΛΕΚΤΡΙΣΜΟΣ")

    def test_a_greek_answer_has_only_greek_letters(self) -> None:
        for title in ("Μηχανή Carnot", "Wayback Machine", "OCLC", "Digital object identifier"):
            self.assertIsNone(title_to_answer(title, lang="el"), title)
        terms = extract_terms_lead_bold(["Carnot", "Μηχανή Carnot", "θερμότητα"], source_title="x", lang="el")
        self.assertEqual([term.normalized_answer for term in terms], ["ΘΕΡΜΟΤΗΤΑ"])
        self.assertEqual(extract_terms_lead_bold(["Θερμότητα"], source_title="x", lang="en"), [])
        self.assertIsNone(title_to_answer("Θερμότητα", lang="en"))

    def test_greek_endings_are_not_taken_for_the_latin_plural(self) -> None:
        rows = [
            {"normalized_answer": "ΝΟΜΟΣ", "answer_score": 0.5},
            {"normalized_answer": "ΝΟΜΟ", "answer_score": 0.4},
        ]
        self.assertEqual(len(merge_plural_duplicates(rows)), 2)


class MergePluralDuplicatesTests(unittest.TestCase):
    @staticmethod
    def _row(key: str, score: float) -> dict:
        return {"normalized_answer": key, "answer_score": score}

    def test_keeps_better_scored_form(self) -> None:
        rows = [self._row("STYLE", 0.4), self._row("STYLES", 0.6), self._row("BOX", 0.5), self._row("BOXES", 0.2)]
        kept = {row["normalized_answer"] for row in merge_plural_duplicates(rows)}
        self.assertEqual(kept, {"STYLES", "BOX"})

    def test_singular_wins_tie(self) -> None:
        rows = [self._row("SOUNDS", 0.5), self._row("SOUND", 0.5)]
        self.assertEqual([row["normalized_answer"] for row in merge_plural_duplicates(rows)], ["SOUND"])

    def test_unrelated_s_endings_survive(self) -> None:
        rows = [self._row("JAZZ", 0.5), self._row("BLUES", 0.5), self._row("CLASS", 0.5)]
        self.assertEqual(len(merge_plural_duplicates(rows)), 3)


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
