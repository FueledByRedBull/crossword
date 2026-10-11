import unittest
from collections import Counter

from src.clue_builder import (
    clue_pass_extract,
    clue_pass_extract_with_offset,
    clue_pass_mask_trim,
    clue_pass_validate,
    definition_clue,
    enforce_diversity,
    has_morphological_leakage,
    has_verb,
    is_generic_definition,
    mask_answer,
    split_sentences,
    strip_trailing_sections,
    trim_clause,
)


def _load_nlp(model: str = "en_core_web_sm"):
    try:
        import spacy

        return spacy.load(model)
    except Exception:
        return None


class ClueBuilderTests(unittest.TestCase):
    def test_split_sentences(self) -> None:
        text = "First sentence. Second sentence! Third?"
        sentences = split_sentences(text)
        self.assertEqual(len(sentences), 3)

    def test_mask_and_validate(self) -> None:
        sentence = "Entropy is a measure of disorder in a system."
        clue, _ = clue_pass_mask_trim(sentence, "Entropy", max_words=10)
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
        clue, _ = clue_pass_mask_trim(sentence, "Entropy", max_words=10)
        self.assertFalse(
            clue_pass_validate(clue, "Entropy", min_words=2, nlp=nlp, original_sentence=sentence)
        )

    def test_split_sentences_keeps_initials_and_abbreviations_together(self) -> None:
        text = "Charles Mingus Jr. was a bassist. John F. Kennedy spoke."
        self.assertEqual(
            split_sentences(text),
            ["Charles Mingus Jr. was a bassist.", "John F. Kennedy spoke."],
        )


class WordBoundaryTests(unittest.TestCase):
    def test_answer_inside_longer_word_is_not_masked(self) -> None:
        self.assertEqual(mask_answer("An extension of bebop.", "TENSION"), "An extension of bebop.")
        self.assertEqual(mask_answer("Revivalists played on.", "REVIVAL"), "Revivalists played on.")
        self.assertEqual(mask_answer("Rising tension grew.", "TENSION"), "Rising ____ grew.")

    def test_hyphenated_compound_is_not_masked(self) -> None:
        self.assertEqual(mask_answer("African-American history", "American"), "African-American history")

    def test_multi_word_answer_becomes_single_blank(self) -> None:
        masked = mask_answer("Miles Davis recorded it. Later Miles  Davis left.", "Miles Davis")
        self.assertEqual(masked, "____ recorded it. Later ____ left.")

    def test_extract_and_validate_respect_word_boundaries(self) -> None:
        sentence, offset = clue_pass_extract_with_offset("An extension of bebop. Tension rose.", "TENSION")
        self.assertEqual((sentence, offset), ("Tension rose.", 1))
        self.assertIsNone(clue_pass_extract("An extension of bebop.", "TENSION"))
        # "extension" in the clue is not a leak of TENSION at the text level.
        self.assertTrue(clue_pass_validate("An extension of the idea", "TENSION", min_words=3))
        self.assertFalse(clue_pass_validate("Mounting tension in the idea", "TENSION", min_words=3))


class TrimClauseTests(unittest.TestCase):
    @staticmethod
    def _sentence(blank: int, length: int, commas: tuple[int, ...] = ()) -> str:
        words = [f"w{i}," if i in commas else f"w{i}" for i in range(length)]
        words[0], words[blank], words[-1] = "Start", "____", words[-1] + "."
        return " ".join(words)

    def test_sentence_up_to_the_cap_is_kept_whole(self) -> None:
        sentence = "Heat flows from a hot body to a ____ one."
        self.assertEqual(trim_clause(sentence), (sentence, True))
        long_sentence = self._sentence(blank=5, length=18)
        self.assertEqual(trim_clause(long_sentence), (long_sentence, True))

    def test_text_that_does_not_read_as_a_sentence_is_not_clean(self) -> None:
        self.assertFalse(trim_clause("is given by the frequency multiplied by the ____.")[1])
        self.assertFalse(trim_clause("Both these spinor polarizations satisfy the momentum space ____")[1])
        self.assertFalse(trim_clause(", known as the reduced ____ as the quantum of angular momentum.")[1])
        self.assertFalse(clue_pass_mask_trim('Of this line of thought, Bell asked, "Whose entropy?', "entropy")[1])

    def test_titles_and_citations_have_no_verb(self) -> None:
        nlp = _load_nlp()
        if nlp is None:
            self.skipTest("spaCy model not available")
        for title in ("Infinite Potential: the Life and Times of ____.", "____ on the cosmological constant problem."):
            self.assertFalse(has_verb(title, nlp), title)
        self.assertTrue(has_verb("Most fermions decay by a ____ over time.", nlp))

    def test_a_window_that_opens_mid_sentence_is_never_clean(self) -> None:
        # "... which was centered in the region of ____." reads as a fragment even though a comma precedes it.
        clue, clean = trim_clause(self._sentence(blank=26, length=30, commas=(17,)))
        self.assertFalse(clean)
        self.assertTrue(clue.startswith("... w18 ") and clue.endswith("w29."), clue)
        self.assertFalse(trim_clause(self._sentence(blank=26, length=30, commas=(17, 20)))[1])

    def test_blank_near_the_start_is_cut_once_at_a_clause_boundary(self) -> None:
        clue, clean = trim_clause(self._sentence(blank=3, length=30, commas=(8,)))
        self.assertTrue(clean)
        self.assertTrue(clue.startswith("Start") and clue.endswith("w8, ..."), clue)

    def test_the_longest_clean_window_wins(self) -> None:
        clue, clean = trim_clause(self._sentence(blank=3, length=40, commas=(8, 11)))
        self.assertTrue(clean)
        self.assertTrue(clue.startswith("Start") and clue.endswith("w11, ..."), clue)

    def test_mid_clause_cut_is_not_clean(self) -> None:
        # "... the substances are separately added to a ____ and the initial and ..."
        clue, clean = trim_clause(self._sentence(blank=8, length=30))
        self.assertFalse(clean)
        self.assertTrue(clue.endswith(" ..."), clue)

    def test_blank_in_the_middle_has_no_clean_window(self) -> None:
        clue, clean = trim_clause(self._sentence(blank=20, length=40, commas=(14, 25)))
        self.assertFalse(clean)
        self.assertTrue(clue.startswith("... ") and clue.endswith(" ..."), clue)

    def test_unclean_window_is_centred_on_the_blank(self) -> None:
        clue, _ = trim_clause(self._sentence(blank=22, length=40, commas=(17,)))
        self.assertTrue(clue.startswith("... w18 "), clue)
        self.assertEqual(len(clue.split()), 14)

    def test_without_blank_keeps_first_words(self) -> None:
        sentence = " ".join(f"w{i}" for i in range(30))
        self.assertEqual(trim_clause(sentence, max_words=5), ("w0 w1 w2 w3 w4", False))


class RequireBlankTests(unittest.TestCase):
    def test_clue_without_blank_is_rejected_when_blank_required(self) -> None:
        self.assertTrue(clue_pass_validate("Study opens the door to many ideas", "Entropy", min_words=3))
        self.assertFalse(
            clue_pass_validate("Study opens the door to many ideas", "Entropy", min_words=3, require_blank=True)
        )
        self.assertTrue(
            clue_pass_validate("Study opens ____ to many ideas", "Entropy", min_words=3, require_blank=True)
        )


class LeakageTests(unittest.TestCase):
    MUST_FLAG = [
        ("Entropic processes increase disorder.", "Entropy"),
        ("Musicians play it nightly.", "Music"),
        ("A jazzy tune from the club.", "Jazz"),
        ("A gentle rock and a roll.", "Rock and roll"),
    ]
    MUST_NOT_FLAG = [
        ("It is a measure of how hot or cold a body is.", "TEMPERATURE"),
        ("A woodwind instrument played by Charlie Parker and John Coltrane.", "SAXOPHONE"),
        ("This adjective describes a continent south of Europe and its people.", "AFRICAN"),
        ("A gap or interval between two events", "INTERNATIONAL"),
    ]

    def _check(self, nlp) -> None:
        for clue, answer in self.MUST_FLAG:
            self.assertTrue(has_morphological_leakage(clue, answer, nlp), (clue, answer))
        for clue, answer in self.MUST_NOT_FLAG:
            self.assertFalse(has_morphological_leakage(clue, answer, nlp), (clue, answer))

    def test_without_nlp_falls_back_to_plain_tokens(self) -> None:
        self._check(None)

    def test_with_spacy_lemmas(self) -> None:
        nlp = _load_nlp()
        if nlp is None:
            self.skipTest("spaCy model not available")
        self._check(nlp)

    def test_every_token_of_multi_word_answer_counts(self) -> None:
        self.assertTrue(has_morphological_leakage("The Davis quintet", "Miles Davis", None))


class DefinitionClueTests(unittest.TestCase):
    def test_strips_subject_and_copula(self) -> None:
        lead = (
            "Miles Dewey Davis III (May 26, 1926 - September 28, 1991) was an American "
            "jazz trumpeter, bandleader, and composer."
        )
        self.assertEqual(definition_clue(lead), "An American jazz trumpeter, bandleader, and composer")

    def test_no_copula_returns_none(self) -> None:
        self.assertIsNone(definition_clue("Jazz hands wave across the stage."))

    def test_too_short_remainder_returns_none(self) -> None:
        self.assertIsNone(definition_clue("Bebop is big."))

    def test_long_definition_stops_at_clean_boundary(self) -> None:
        lead = (
            "Jazz is a music genre that originated in the African-American communities of "
            "New Orleans, Louisiana, in the late 19th and early 20th centuries."
        )
        self.assertEqual(
            definition_clue(lead),
            "A music genre that originated in the African-American communities of New Orleans, Louisiana",
        )

    def test_whole_predicate_is_used_when_it_fits(self) -> None:
        enthalpy = (
            "Enthalpy is the sum of a thermodynamic system's internal energy and the product "
            "of its pressure and volume."
        )
        self.assertEqual(
            definition_clue(enthalpy),
            "The sum of a thermodynamic system's internal energy and the product of its pressure and volume",
        )

    def test_relative_clause_is_dropped_only_where_it_leaves_a_real_phrase(self) -> None:
        bohm = (
            "David Joseph Bohm (; 20 December 1917 - 27 October 1992) was an American scientist who has been "
            "described as one of the most significant theoretical physicists of the 20th century and who "
            "contributed unorthodox ideas to quantum theory."
        )
        self.assertEqual(definition_clue(bohm), "An American scientist")
        big_band = (
            "A big band is a type of musical ensemble of jazz music that usually consists of ten "
            "or more musicians with sections of saxophones, trumpets and trombones."
        )
        self.assertEqual(definition_clue(big_band), "A type of musical ensemble of jazz music")
        thermodynamics = (
            "Thermodynamics is a branch of physics that deals with heat, work, and temperature, and their "
            "relation to energy, entropy, and the physical properties of matter and radiation."
        )
        self.assertEqual(definition_clue(thermodynamics), "A branch of physics that deals with heat, work, and temperature")

    def test_cut_never_leaves_a_dangling_end(self) -> None:
        for lead, expected in (
            (
                "Photon is an elementary particle that is a quantum of the electromagnetic field, including "
                "electromagnetic radiation such as light and radio waves.",
                "An elementary particle that is a quantum of the electromagnetic field",
            ),
            (
                "Absolute zero is the lowest theoretically possible temperature, a state at which the enthalpy "
                "and entropy of a cooled ideal gas reach their minimum value, taken as 0 K.",
                "The lowest theoretically possible temperature",
            ),
            (
                "Banias is a site in the Israeli-occupied Golan Heights, Syria near a natural spring, once "
                "a place of worship for many different peoples and their gods.",
                "A site in the Israeli-occupied Golan Heights, Syria near a natural spring",
            ),
            (
                "Daniel Bernoulli was a Swiss mathematician and physicist and was one of the many prominent "
                "mathematicians in the Bernoulli family.",
                "A Swiss mathematician and physicist",
            ),
            (
                "Henry Moseley was an English physicist, whose contribution to the science of physics was the "
                "justification of the atomic number.",
                "An English physicist",
            ),
            (
                "Stephen Hawking was an English theoretical astrophysicist, cosmologist, and author who was "
                "director of research at the Centre for Theoretical Cosmology.",
                "An English theoretical astrophysicist, cosmologist, and author",
            ),
            (
                "A fume hood is a type of local exhaust ventilation device that is designed to prevent users "
                "from inhaling hazardous fumes.",
                "A type of local exhaust ventilation device",
            ),
            (
                "Scattering is a wide range of physical processes where moving particles or radiation of some "
                "form, such as light or sound, are forced to deviate.",
                "A wide range of physical processes",
            ),
        ):
            self.assertEqual(definition_clue(lead), expected, lead)
        bell = (
            "Bell's theorem is a term encompassing a number of closely related results in physics, all of "
            "which determine that quantum mechanics is incompatible with local hidden-variable theories."
        )
        self.assertEqual(definition_clue(bell), "A term encompassing a number of closely related results in physics")
        forshaw = (
            "Jeff Forshaw is a British particle physicist with a special interest in quantum chromodynamics: "
            "the study of the behaviour of subatomic particles, using data from several accelerators."
        )
        self.assertEqual(
            definition_clue(forshaw),
            "A British particle physicist with a special interest in quantum chromodynamics",
        )
        euler = (
            "Leonhard Euler was a Swiss polymath who was active as a mathematician, physicist, astronomer, "
            "logician, geographer, music theorist and engineer in the eighteenth century."
        )
        self.assertEqual(definition_clue(euler), "A Swiss polymath")
        bbc = (
            "BBC News is an operational business division of the British Broadcasting Corporation responsible "
            "for the gathering and broadcasting of news and current affairs in the UK and around the world."
        )
        self.assertEqual(
            definition_clue(bbc), "An operational business division of the British Broadcasting Corporation"
        )
        antakya = (
            "Antakya is a municipality and the capital district of Hatay Province, Turkey, with an area of "
            "703 km2 and a population of around 400,000 people as of 2022."
        )
        self.assertEqual(
            definition_clue(antakya), "A municipality and the capital district of Hatay Province, Turkey"
        )

    def test_no_clean_boundary_means_no_definition(self) -> None:
        for lead in (
            "Bones are a folk instrument that, in their original form, were made from animal ribs and used "
            "as a percussion instrument in many places.",
            "String theory is a theoretical framework in which the point-like particles of particle physics "
            "are replaced by one-dimensional objects called strings.",
            "Albert Gleizes was a French artist, theoretician, philosopher, a self-proclaimed founder of "
            "Cubism and an influence on the School of Paris.",
            "Bluegrass is a genre of roots music that developed in the 1940s in the Appalachian region of "
            "the United States.",
            "Agnosticism is the view or belief that the existence of God, the divine, or the supernatural is "
            "either unknowable in principle or unknown in fact.",
            "Science is a systematic discipline that builds and organises knowledge in the form of testable "
            "hypotheses and predictions about the universe.",
            "Born rule is a postulate of quantum mechanics that gives the probability that a measurement on "
            "a quantum system will yield a given result.",
        ):
            self.assertIsNone(definition_clue(lead), lead)

    def test_nested_parentheticals_are_stripped(self) -> None:
        lead = "Tempo (from Italian (plural tempi) for time) is the speed or pace of a given composition."
        self.assertEqual(definition_clue(lead), "The speed or pace of a given composition")
        self.assertEqual(split_sentences("Gell-Mann (; born 1929) was a physicist."), ["Gell-Mann was a physicist."])
        self.assertEqual(
            split_sentences("He was known as Bacchus ( or ; Greek: x) by the Romans."),
            ["He was known as Bacchus by the Romans."],
        )

    def test_definition_must_be_about_the_answer(self) -> None:
        tempo = "Tempo is the speed or pace of a given composition, and is often also an indication of mood."
        self.assertIsNone(definition_clue(tempo, answer="Beats per minute"))
        miles = "Miles Dewey Davis III (May 26, 1926 - September 28, 1991) was an American jazz trumpeter."
        self.assertEqual(definition_clue(miles, answer="Miles Davis"), "An American jazz trumpeter")
        cab = "Cabell Calloway III (December 25, 1907 - November 18, 1994) was an American jazz singer."
        self.assertEqual(definition_clue(cab, answer="Cab Calloway"), "An American jazz singer")
        plural = "Afro-Cubans (Spanish: Afrocubano) are citizens of Cuba with African ancestry."
        self.assertIsNotNone(definition_clue(plural, answer="Afro-Cuban"))
        accented = "Schr\u00f6dinger's cat is a thought experiment in quantum physics."
        self.assertIsNotNone(definition_clue(accented, answer="Schrodinger's cat"))

    def test_adverb_start_is_not_a_definition(self) -> None:
        self.assertIsNone(definition_clue("Bulgarian jazz is significantly influenced by the cultural changes."))
        self.assertIsNone(definition_clue("Heat is defined as the form of energy crossing the boundary of a system."))

    def test_possessive_or_demonstrative_start_is_not_a_definition(self) -> None:
        self.assertIsNone(definition_clue("State is its condition at a specific time."))

    def test_dangling_demonstrative_after_preposition_is_not_a_definition(self) -> None:
        hamiltonian = "Hamiltonian is an operator corresponding to the total energy of that system."
        self.assertIsNone(definition_clue(hamiltonian))
        genre = "Bebop is a genre that originated in the early 1940s."
        self.assertEqual(definition_clue(genre), "A genre that originated in the early 1940s")

    def test_trailing_dash_is_dropped(self) -> None:
        field = "A field is a physical quantity \u2013 represented by a scalar or tensor \u2013."
        self.assertEqual(definition_clue(field), "A physical quantity \u2013 represented by a scalar or tensor")

    def test_latex_markup_is_not_a_clue(self) -> None:
        clue = "In the discrete case, the ____ element F i {\\displaystyle F_{i}} is positive"
        self.assertFalse(clue_pass_validate(clue, "POVM", require_blank=True))


class DefinitionCapAndEdgesTests(unittest.TestCase):
    DAMPER = (
        "A damper is a valve or plate that stops or regulates the flow of air inside a duct, chimney, "
        "VAV box, air handler, or other air-handling equipment."
    )
    CONVECTION = (
        "Convection is single-phase or multiphase fluid flow that occurs spontaneously through the combined "
        "effects of material property heterogeneity and body forces on a fluid, most commonly density and gravity."
    )
    UMM_QAIS = (
        "Umm Qais, also known as Qays, is a town in northern Jordan principally known for its proximity "
        "to the ruins of the ancient Gadara."
    )
    DENARIUS = (
        "The denarius was the standard Roman silver coin from its introduction in the Second Punic War "
        "c. 211 BC to the reign of Gordian III, when it was gradually replaced by the antoninianus."
    )
    CARRHAE = (
        "The Battle of Carrhae was fought in 53 BC between the Roman Republic and the Parthian Empire "
        "near the ancient town of Carrhae."
    )
    DIRAC = (
        "Paul Adrien Maurice Dirac was an English theoretical physicist who is considered to be one of "
        "the founders of quantum mechanics."
    )

    def test_definition_with_no_clean_cut_within_the_cap_is_rejected(self) -> None:
        for lead in (self.DAMPER, self.CONVECTION):
            self.assertIsNone(definition_clue(lead, max_words=18), lead)

    def test_a_longer_cut_keeps_what_the_person_is_considered(self) -> None:
        self.assertIsNone(definition_clue(self.DIRAC, max_words=8))
        self.assertEqual(
            definition_clue(self.DIRAC, max_words=18),
            "An English theoretical physicist, considered to be one of the founders of quantum mechanics",
        )

    def test_definition_ending_in_an_adverb_is_rejected(self) -> None:
        nlp = _load_nlp()
        if nlp is None:
            self.skipTest("spaCy model not available")
        self.assertIsNone(definition_clue(self.UMM_QAIS, nlp=nlp))
        self.assertEqual(
            definition_clue(self.UMM_QAIS, max_words=18, nlp=nlp),
            "A town in northern Jordan principally known for its proximity to the ruins of the ancient Gadara",
        )
        # Without a tagger only the word lists apply.
        self.assertEqual(definition_clue(self.UMM_QAIS), "A town in northern Jordan principally")

    def test_closing_possessive_adjunct_is_cut_off(self) -> None:
        self.assertEqual(definition_clue(self.DENARIUS), "The standard Roman silver coin")

    def test_definition_starting_with_a_participle_is_rejected(self) -> None:
        nlp = _load_nlp()
        if nlp is None:
            self.skipTest("spaCy model not available")
        self.assertIsNone(definition_clue(self.CARRHAE, nlp=nlp))
        self.assertTrue(definition_clue(self.CARRHAE).startswith("Fought in 53 BC"))


class GenericDefinitionTests(unittest.TestCase):
    # Lead-sentence document frequencies measured on the offline runs of the four test seeds.
    COMMON = {"a": 101, "an": 47, "the": 140, "of": 120, "and": 118, "that": 60, "is": 100, "in": 90}
    POOLS = {
        "quantum": (150, {
            "physicist": 43, "theoretical": 31, "british": 6, "american": 19, "brazilian": 1, "particle": 17,
            "quantum": 62, "elementary": 5, "electromagnetic": 6, "field": 13, "form": 6, "angular": 2,
            "intrinsic": 1, "momentum": 4,
        }),
        "thermo": (82, {"german": 4, "physicist": 12, "mathematician": 4}),
        "jazz": (150, {
            "music": 74, "genre": 38, "musical": 18, "form": 15, "style": 22, "dance": 15, "electronic": 6,
            "subculture": 2,
        }),
        "rome": (105, {
            "god": 3, "war": 5, "capital": 7, "largest": 4, "city": 13, "lebanon": 1, "gallic": 2,
            "chieftain": 1, "senones": 1, "ancient": 21, "tribe": 2, "europe": 6,
        }),
    }

    def _generic(self, pool: str, clue: str, nlp=None) -> bool:
        size, counts = self.POOLS[pool]
        return is_generic_definition(clue, Counter({**self.COMMON, **counts}), size, nlp)

    def test_definitions_that_fit_many_answers_are_generic(self) -> None:
        for pool, clue in (
            ("quantum", "A British theoretical physicist"),
            ("thermo", "A German physicist and mathematician"),
            ("jazz", "A music genre and musical form"),
            ("jazz", "A style of music and dance form"),
            ("jazz", "A genre of electronic dance music"),
        ):
            self.assertTrue(self._generic(pool, clue), clue)

    def test_a_rare_nationality_does_not_make_a_definition_specific(self) -> None:
        nlp = _load_nlp()
        if nlp is None:
            self.skipTest("spaCy model not available")
        clue = "An American-Brazilian physicist"
        self.assertFalse(self._generic("quantum", clue))
        self.assertTrue(self._generic("quantum", clue, nlp))

    def test_definitions_with_a_rare_word_are_kept(self) -> None:
        for pool, clue in (
            ("rome", "The god of war"),
            ("rome", "The capital and largest city of Lebanon"),
            ("rome", "A Gallic chieftain of the Senones"),
            ("quantum", "An intrinsic form of angular momentum"),
            ("quantum", "An elementary particle that is a quantum of the electromagnetic field"),
        ):
            self.assertFalse(self._generic(pool, clue), clue)

    def test_known_misses_pass_because_their_nouns_are_rare_in_the_pool(self) -> None:
        # "subculture" is in 2 of 150 jazz leads and "tribe" in 2 of 105 Rome leads: a frequency test cannot see
        # that these two fit many answers.
        self.assertFalse(self._generic("jazz", "A genre of dance music and a subculture"))
        self.assertFalse(self._generic("rome", "An ancient tribe in Europe"))

    def test_trailing_sections_are_cut_at_the_first_of_them(self) -> None:
        body = "Weinberg won a prize."
        headings = (
            "See also", "Notes", "References", "Citations", "Sources", "Bibliography", "Further reading",
            "External links", "Works", "Publications", "Selected works", "Discography", "Filmography",
        )
        for heading in headings:
            text = f"{body}\n\n\nCareer\nHe taught.\n\n\n{heading}\nA ____ link.\n\n\nExternal links\nB"
            self.assertEqual(strip_trailing_sections(text), f"{body}\n\n\nCareer\nHe taught.", heading)
        self.assertEqual(strip_trailing_sections(f"{body}\n\n\nNotes"), body)

    def test_a_section_title_inside_the_text_is_not_a_trailing_section(self) -> None:
        text = "See also the notes.\n\n\nHistory\nThe Works of Plato and Notes\nwere read.\nReferences\nwere kept."
        self.assertEqual(strip_trailing_sections(text), text)

    def _fill_in_passes(self, sentence: str, answer: str) -> bool:
        clue, _ = clue_pass_mask_trim(sentence, answer)
        return clue_pass_validate(clue, answer, original_sentence=sentence, require_blank=True)

    def test_a_blank_inside_a_longer_proper_name_is_not_a_fill_in(self) -> None:
        dash = chr(0x2013)
        for sentence, answer in (
            ("In 2013, The Max Planck Florida Institute For Neuroscience opened in Jupiter, Florida.", "Max Planck"),
            ("The American Physical Society Hans Bethe Prize was named after him as well.", "Hans Bethe"),
            ("As of 2022, the Bardeen Professor is Yurii Vlasov.", "Bardeen"),
            (f"In October 1936, Germany and Italy formed the Rome{dash}Berlin Axis.", "Axis"),
            ("Prize money went to the winner of the Nobel Award last year.", "Award"),
            ("The Max Planck Florida Institute for Neuroscience opened in Jupiter in 2013.", "Max Planck"),
        ):
            with self.subTest(answer=answer):
                self.assertFalse(self._fill_in_passes(sentence, answer))

    def test_a_possessive_or_lowercase_neighbours_leave_a_fill_in_alone(self) -> None:
        for sentence, answer in (
            ("Modal jazz recordings, such as Miles Davis's Kind of Blue, became popular in the late 1950s.", "Miles Davis"),
            ("The first publication of jazz sheet music was in 1908.", "jazz"),
            ("Mithridates VI was the ruler of Pontus, a large kingdom in Anatolia, from 120 to 63 BC.", "Anatolia"),
            ("Bardeen won two Nobel Prizes in physics during his long career.", "Bardeen"),
            ("The Bardeen prize was awarded for work on superconductors in the 1950s.", "Bardeen"),
        ):
            with self.subTest(answer=answer):
                self.assertTrue(self._fill_in_passes(sentence, answer))

    def test_a_participle_phrase_after_a_second_predicate_or_who_is_kept(self) -> None:
        considered = "central founding fathers of the science of thermodynamics"
        lead = f"Rudolf Julius Emanuel Clausius was a German physicist and mathematician and is considered one of the {considered}."
        self.assertEqual(
            definition_clue(lead, answer="Rudolf Clausius", max_words=18),
            f"A German physicist and mathematician, considered one of the {considered}",
        )
        # At 16 words only the head fits.
        self.assertEqual(definition_clue(lead, answer="Rudolf Clausius", max_words=16), "A German physicist and mathematician")
        for verb, expected in (
            ("who is widely regarded as", "An English chemist, regarded as the father of modern gas laws"),
            ("and was credited with", "An English chemist, credited with the father of modern gas laws"),
            ("who was also known as", "An English chemist, known as the father of modern gas laws"),
            ("who is remembered as", "An English chemist, remembered as the father of modern gas laws"),
        ):
            lead = f"Robert Boyle was an English chemist {verb} the father of modern gas laws."
            self.assertEqual(definition_clue(lead, answer="Robert Boyle", max_words=18), expected, verb)

    def test_other_second_predicates_are_still_dropped(self) -> None:
        lead = "Daniel Bernoulli was a Swiss mathematician and physicist and was one of the many prominent mathematicians."
        self.assertEqual(definition_clue(lead, answer="Daniel Bernoulli", max_words=18), "A Swiss mathematician and physicist")


class FillInOpeningTests(unittest.TestCase):
    @staticmethod
    def _passes(clue: str) -> bool:
        return clue_pass_validate(clue, "Entropy", min_words=3, require_blank=True)

    def test_a_fill_in_that_opens_on_a_back_reference_is_rejected(self) -> None:
        for clue in (
            "It is the cause of the ____.",
            "It underlies modern ____ and the virial theorem.",
            "This is useful if the equation of ____ is known.",
            "These breakbeats enabled the subsequent evolution of the ____ style.",
            "But the devices used to build the experimental apparatus are themselves ____, ...",
            "Now pair it with a reversed ____ as shown by the right figure.",
            'He has also used the term "R&B" as a synonym for ____.',
            "And the ____ followed.",
            "Αυτό είναι το ____ του συστήματος.",
            "Και η ____ μεταφέρεται από το ένα σώμα στο άλλο.",
        ):
            with self.subTest(clue=clue):
                self.assertFalse(self._passes(clue))

    def test_other_openings_are_kept(self) -> None:
        for clue in (
            "Heat flows from a hot body to a ____ one.",
            "In 1833, Hamilton reformulated classical ____.",
            "The ____ of a system never decreases.",
            "Bell asked what the ____ meant.",
            "Η ____ μεταφέρεται από το ένα σώμα στο άλλο.",
        ):
            with self.subTest(clue=clue):
                self.assertTrue(self._passes(clue))

    def test_a_window_that_opens_mid_sentence_is_not_a_clue(self) -> None:
        sentence = (
            "The Kingdom of Pontus, which was founded by Mithridates in the fourth century BC and which grew "
            "rapidly under his successors, was centered in the region of Anatolia."
        )
        clue, clean = clue_pass_mask_trim(sentence, "Anatolia")
        self.assertEqual(clue, "... grew rapidly under his successors, was centered in the region of ____.")
        self.assertFalse(clean)
        # The start of a sentence cut once at a clause boundary still is.
        self.assertTrue(clue_pass_mask_trim(sentence.replace("Anatolia", "Pontus"), "Pontus")[1])


class DefinitionCutTests(unittest.TestCase):
    def test_a_cut_inside_a_list_is_not_clean(self) -> None:
        rhythm = (
            "A rhythm section is a group of musicians within a music ensemble or band that provides the underlying "
            "rhythm, harmony and pulse of the accompaniment."
        )
        self.assertEqual(definition_clue(rhythm), "A group of musicians within a music ensemble or band")

    def test_a_cut_that_leaves_a_demonstrative_is_not_clean(self) -> None:
        micro = (
            "The microscopic scale is the scale of objects and events smaller than those that can easily be seen "
            "by the naked eye, requiring a lens or microscope to see them clearly."
        )
        self.assertIsNone(definition_clue(micro))
        self.assertEqual(
            definition_clue(micro, max_words=18),
            "The scale of objects and events smaller than those that can easily be seen by the naked eye",
        )

    def test_a_cut_that_leaves_a_nationality_without_its_noun_is_not_clean(self) -> None:
        nlp = _load_nlp()
        if nlp is None:
            self.skipTest("spaCy model not available")
        city = (
            "Constantinople was a historical city located on the Bosporus, which served as the capital of the Roman, "
            "Byzantine, Latin, and Ottoman empires between its consecration in 330 and the formal abolition of the "
            "Ottoman sultanate in 1922."
        )
        self.assertEqual(
            definition_clue(city), "A historical city located on the Bosporus, which served as the capital of the Roman, Byzantine"
        )
        self.assertEqual(definition_clue(city, nlp=nlp), "A historical city located on the Bosporus")
        clave = (
            "The clave is a rhythmic pattern used as a tool for temporal organization in Brazilian, Cuban, and other "
            "Latin American music."
        )
        self.assertEqual(definition_clue(clave, nlp=nlp), "A rhythmic pattern used as a tool for temporal organization")

    def test_a_possessive_that_points_at_the_removed_subject_is_not_a_definition(self) -> None:
        volume = "In thermodynamics, the volume of a system is an important extensive parameter for describing its thermodynamic state."
        self.assertIsNone(definition_clue(volume, answer="Volume"))
        kinetic = "In physics, the kinetic energy of an object is the form of energy that it possesses due to its motion."
        self.assertIsNone(definition_clue(kinetic, answer="Kinetic energy"))
        # The possessor is the answer, or something inside the clue.
        physics = (
            "Physics is the scientific study of matter, its fundamental constituents, its motion and behavior "
            "through space and time, and the related entities of energy and force."
        )
        self.assertEqual(
            definition_clue(physics, answer="Physics"),
            "The scientific study of matter, its fundamental constituents, its motion and behavior",
        )
        diocletian = "Diocletian, nicknamed Jovius, was Roman emperor from 284 until his abdication in 305."
        self.assertEqual(definition_clue(diocletian, answer="Diocletian"), "Roman emperor from 284 until his abdication in 305")

    def test_a_tail_the_cut_left_unfinished_is_dropped(self) -> None:
        roman_law = (
            "Roman law is the legal system of ancient Rome, including the legal developments spanning over a "
            "thousand years of jurisprudence, from the Twelve Tables, to the Corpus Juris Civilis."
        )
        self.assertEqual(definition_clue(roman_law, answer="Roman law"), "The legal system of ancient Rome")
        # Closed by a comma, the tail is whole.
        self.assertEqual(
            definition_clue(roman_law, answer="Roman law", max_words=18),
            "The legal system of ancient Rome, including the legal developments spanning over a thousand years of jurisprudence",
        )


GREEK_LEADS = {
    "Θερμότητα": (
        "Η θερμότητα είναι μορφή ενέργειας που αφορά μακροσκοπικά αντικείμενα, επί της ουσίας όμως πρόκειται για "
        "την κινητική ενέργεια και την ενέργεια ταλάντωσης των μορίων, ατόμων ή ιόντων ενός σώματος."
    ),
    "Εντροπία": (
        "Εντροπία είναι η έννοια μέσω της οποίας μετράται η αταξία, της οποίας η μέγιστη τιμή αντικατοπτρίζει "
        "την πλήρη αποδιοργάνωση και ισοδυναμεί με την παύση της ζωής."
    ),
    "Κλασική μηχανική": (
        "Στη φυσική, η Κλασική Μηχανική είναι ένας από τούς δύο κύριους κλάδους της Μηχανικής, η οποία ασχολείται "
        "με το σύνολο των Φυσικών Νόμων που περιγράφουν την κίνηση σωμάτων υπό την επίδραση ενός συστήματος δυνάμεων."
    ),
    "Κινηματική": (
        "Η Κινηματική είναι κλάδος της μηχανικής που περιγράφει την κίνηση των σωμάτων αδιαφορώντας για τη μάζα "
        "τους ή τις αιτίες, δυνάμεις, που προκαλούν την κίνησή τους."
    ),
    "Θερμοκρασία": (
        "Θερμοκρασία είναι το φυσικό μέγεθος που μετρά τη μέση κινητική ενέργεια από μεταφορά,ταλάντωση ή "
        "περιστροφή των δομικών συστατικών της ύλης."
    ),
    "Ουίλιαμ Ράνκιν": (
        "Ο Ουίλιαμ Ράνκιν ήταν Σκωτσέζος μηχανικός και φυσικός, θεωρούμενος ως ένας από τους πατέρες της "
        "επιστήμης της Θερμοδυναμικής."
    ),
    "Πολυμερές": (
        "Πολυμερή ονομάζονται οι χημικές ενώσεις με μεγάλα μόρια, τα μακρομόρια, που σχηματίζονται από τη σύνδεση "
        "πολλών όμοιων μικρών μορίων που λέγονται μονομερή."
    ),
    "Αεροστατική": (
        "Η αεροστατική είναι ο κλάδος της Φυσικής, και ειδικότερα της Μηχανικής, που πραγματεύεται τις μηχανικές "
        "ιδιότητες των αερίων όταν αυτά βρίσκονται σε ισορροπία."
    ),
}


class GreekDefinitionTests(unittest.TestCase):
    def _definition(self, answer: str, **kwargs) -> str | None:
        return definition_clue(GREEK_LEADS[answer], answer=answer, lang="el", **kwargs)

    def test_copulas_articles_and_relative_clauses(self) -> None:
        for answer, expected in (
            ("Θερμότητα", "Μορφή ενέργειας που αφορά μακροσκοπικά αντικείμενα"),
            ("Εντροπία", "Η έννοια μέσω της οποίας μετράται η αταξία"),
            ("Κλασική μηχανική", "Ένας από τούς δύο κύριους κλάδους της Μηχανικής"),
            ("Ουίλιαμ Ράνκιν", "Σκωτσέζος μηχανικός και φυσικός, θεωρούμενος ως ένας από τους πατέρες της επιστήμης της Θερμοδυναμικής"),
            ("Πολυμερές", "Οι χημικές ενώσεις με μεγάλα μόρια, τα μακρομόρια"),
            ("Αεροστατική", "Ο κλάδος της Φυσικής, και ειδικότερα της Μηχανικής"),
        ):
            with self.subTest(answer=answer):
                self.assertEqual(self._definition(answer), expected)

    def test_a_gerund_opens_a_clause_that_can_be_cut_off(self) -> None:
        self.assertEqual(
            self._definition("Κινηματική"), "Κλάδος της μηχανικής που περιγράφει την κίνηση των σωμάτων"
        )

    def test_the_cap_is_kept_and_a_relative_clause_is_not_cut_in_the_middle(self) -> None:
        self.assertIsNone(self._definition("Θερμοκρασία", max_words=16))
        self.assertEqual(
            self._definition("Θερμοκρασία", max_words=18),
            "Το φυσικό μέγεθος που μετρά τη μέση κινητική ενέργεια από μεταφορά,ταλάντωση ή περιστροφή των δομικών συστατικών της ύλης",
        )

    def test_the_sentence_must_be_about_the_answer_whatever_its_case_or_accents(self) -> None:
        lead = GREEK_LEADS["Θερμότητα"]
        self.assertIsNone(definition_clue(lead, answer="Εντροπία", lang="el"))
        self.assertIsNotNone(definition_clue(lead.upper(), answer="Θερμότητα", lang="el"))
        self.assertEqual(
            definition_clue("η θερμοτητα ειναι μορφη ενεργειας που αφορα αντικειμενα", answer="Θερμότητα", lang="el"),
            "Μορφη ενεργειας που αφορα αντικειμενα",
        )
        # An inflected subject is still the subject.
        self.assertIsNotNone(definition_clue("Της Εντροπίας ο ορισμός είναι μια έννοια της φυσικής", answer="Εντροπία", lang="el"))

    def test_an_accusative_predicate_after_apotelei_is_not_a_definition(self) -> None:
        for lead, answer in (
            ("Η Μηχανική αποτελεί έναν ευρύ κλάδο της φυσικής που ασχολείται με τη συμπεριφορά των σωμάτων.", "Μηχανική"),
            ("Η Ουράνια Μηχανική αποτελεί έναν από τους βασικούς κλάδους της Αστρονομίας.", "Ουράνια Μηχανική"),
            ("Η Αστροφυσική αποτελούσε τον κλάδο της αστρονομίας που μελετά τα άστρα.", "Αστροφυσική"),
        ):
            with self.subTest(lead=lead):
                self.assertIsNone(definition_clue(lead, answer=answer, lang="el"))
        # Nominative verbs, and feminine or neuter predicates, are fine.
        self.assertIsNotNone(
            definition_clue("Η Μηχανική είναι έναν ευρύ κλάδο της φυσικής που ασχολείται με τα σώματα.", answer="Μηχανική", lang="el")
        )
        self.assertEqual(
            definition_clue(
                "Ο πρώτος θερμοδυναμικός νόμος αποτελεί μια έκφραση της αρχής διατήρησης της ενέργειας.",
                answer="Πρώτος θερμοδυναμικός νόμος",
                lang="el",
            ),
            "Μια έκφραση της αρχής διατήρησης της ενέργειας",
        )
        self.assertEqual(
            definition_clue(
                "Η Υδροστατική αποτελεί το μέρος εκείνο της Φυσικής που πραγματεύεται κυρίως τα ηρεμούντα υγρά.",
                answer="Υδροστατική",
                lang="el",
            ),
            "Το μέρος εκείνο της Φυσικής που πραγματεύεται κυρίως τα ηρεμούντα υγρά",
        )

    def test_a_masculine_accusative_head_noun_is_caught_with_the_tagger(self) -> None:
        nlp = _load_nlp("el_core_news_sm")
        if nlp is None:
            self.skipTest("Greek spaCy model not available")
        optics = "Η Γεωμετρική οπτική αποτελεί ιδιαίτερο κλάδο της Οπτικής."
        self.assertIsNotNone(definition_clue(optics, answer="Γεωμετρική οπτική", lang="el"))
        self.assertIsNone(definition_clue(optics, answer="Γεωμετρική οπτική", lang="el", nlp=nlp))
        kinematics = "Η Κινηματική είναι κλάδος της μηχανικής που περιγράφει την κίνηση των σωμάτων."
        self.assertIsNotNone(definition_clue(kinematics, answer="Κινηματική", lang="el", nlp=nlp))
        neuter = "Η Κβαντική οπτική αποτελεί πεδίο της φυσικής και της μηχανικής των κυμάτων."
        self.assertIsNotNone(definition_clue(neuter, answer="Κβαντική οπτική", lang="el", nlp=nlp))

    def test_no_copula_or_the_wrong_language_means_no_definition(self) -> None:
        self.assertIsNone(definition_clue("Η Μετεωρολογία ανήκει στις θετικές επιστήμες, με κύριο αντικείμενο την έρευνα.", lang="el"))
        self.assertIsNone(definition_clue(GREEK_LEADS["Θερμότητα"], answer="Θερμότητα"))
        self.assertIsNone(definition_clue("Heat is a form of energy that concerns bodies.", answer="Heat", lang="el"))

    def test_a_cut_never_ends_on_an_article_preposition_conjunction_or_relative(self) -> None:
        for answer in GREEK_LEADS:
            for max_words in (10, 12, 14, 16, 18):
                clue = self._definition(answer, max_words=max_words)
                if clue is None:
                    continue
                with self.subTest(answer=answer, max_words=max_words):
                    self.assertLessEqual(len(clue.split()), max_words)
                    self.assertNotIn(clue.split()[-1].lower(), {"ο", "η", "το", "οι", "τα", "του", "της", "των", "τον", "την", "τους", "τις", "και", "που", "με", "σε", "από", "για", "στο", "στη", "στην"})

    def test_the_result_must_read_as_a_noun_phrase_and_not_leak_the_answer(self) -> None:
        nlp = _load_nlp("el_core_news_sm")
        if nlp is None:
            self.skipTest("Greek spaCy model not available")
        self.assertEqual(self._definition("Θερμότητα", nlp=nlp), "Μορφή ενέργειας που αφορά μακροσκοπικά αντικείμενα")
        for clue, answer, leaks in (
            ("Ο κλάδος της φυσικής που μελετά την ύλη", "Φυσική", True),
            ("Η μεταφορά της θερμότητας μεταξύ σωμάτων", "Θερμότητα", True),
            ("Μορφή ενέργειας που αφορά μακροσκοπικά αντικείμενα", "Ενέργεια", True),
            ("Κλάδος της μηχανικής που περιγράφει την κίνηση των σωμάτων", "Κινηματική", False),
            ("Η έννοια μέσω της οποίας μετράται η αταξία", "Εντροπία", False),
        ):
            for model in (nlp, None):
                with self.subTest(clue=clue, nlp=model is not None):
                    self.assertEqual(has_morphological_leakage(clue, answer, model, lang="el"), leaks)
        self.assertFalse(
            clue_pass_validate("Ο κλάδος της φυσικής που μελετά την ύλη", "Φυσική", min_words=3, nlp=nlp, lang="el")
        )


class GreekTextTests(unittest.TestCase):
    def test_a_blank_inside_a_greek_proper_name_is_not_a_fill_in(self) -> None:
        for sentence, answer in (
            ("Ο Γουΐλλιαμ Ρόουαν Χάμιλτον επαναδιαμόρφωσε τη Λαγκρανζιανή μηχανική, το 1833.", "μηχανική"),
            ("Η Βιοφυσική ή Βιολογική Φυσική είναι ο διεπιστημονικός κλάδος της Βιολογίας.", "Φυσική"),
            ("Η Γενική Σχετικότητα ενώνει την Ειδική Σχετικότητα με τον νόμο της βαρύτητας.", "Σχετικότητα"),
            ("Ο Άλμπερτ Αϊνστάιν διατύπωσε τη θεωρία της σχετικότητας το 1905.", "Άλμπερτ"),
            ("Ο Άλμπερτ Αϊνστάιν διατύπωσε τη θεωρία της σχετικότητας το 1905.", "Αϊνστάιν"),
        ):
            clue, _ = clue_pass_mask_trim(sentence, answer)
            with self.subTest(sentence=sentence, answer=answer):
                self.assertFalse(clue_pass_validate(clue, answer, original_sentence=sentence, require_blank=True, lang="el"))

    def test_lowercase_neighbours_leave_a_greek_fill_in_alone(self) -> None:
        sentence = "Η μετάδοση της θερμότητας γίνεται με αγωγή, με μεταφορά και με ακτινοβολία."
        clue, clean = clue_pass_mask_trim(sentence, "θερμότητας")
        self.assertTrue(clean)
        self.assertTrue(clue_pass_validate(clue, "θερμότητας", original_sentence=sentence, require_blank=True, lang="el"))

    def test_sentences_do_not_split_after_greek_initials_and_abbreviations(self) -> None:
        self.assertEqual(
            split_sentences("Ο Γ. Παπανδρέου γεννήθηκε το 1919. Ήταν πολιτικός."),
            ["Ο Γ. Παπανδρέου γεννήθηκε το 1919.", "Ήταν πολιτικός."],
        )
        self.assertEqual(
            split_sentences("Ο Αρχιμήδης πέθανε το 212 π.Χ. Ο Ευκλείδης έζησε νωρίτερα."),
            ["Ο Αρχιμήδης πέθανε το 212 π.Χ. Ο Ευκλείδης έζησε νωρίτερα."],
        )

    def test_a_greek_sentence_opens_with_a_capital(self) -> None:
        self.assertTrue(trim_clause("Ένα ____ σώμα εκπέμπει θερμότητα προς το περιβάλλον.")[1])
        self.assertFalse(trim_clause("σώμα εκπέμπει θερμότητα προς το ____ περιβάλλον.")[1])

    def test_an_unclosed_guillemet_is_a_fragment(self) -> None:
        self.assertFalse(clue_pass_mask_trim("Ονομάστηκε «η τέχνη των ____ και συνέχισε.", "μηχανών")[1])
        self.assertTrue(clue_pass_mask_trim("Ονομάστηκε «η τέχνη των ____» και συνέχισε.", "μηχανών")[1])

    def test_greek_trailing_sections_are_cut(self) -> None:
        body = "Η θερμότητα είναι μορφή ενέργειας."
        for heading in (
            "Δείτε επίσης", "Δείτε Επίσης", "Παραπομπές", "Παραπομπές και παρατηρήσεις", "Αναφορές", "Σημειώσεις",
            "Πηγές", "Βιβλιογραφία", "Προτεινόμενη βιβλιογραφία", "Περισσότερες πληροφορίες", "Εξωτερικοί σύνδεσμοι",
        ):
            text = f"{body}\n\n\nΙστορία\nΜια πρόταση.\n\n\n{heading}\nΜια ____ σύνδεση."
            self.assertEqual(strip_trailing_sections(text), f"{body}\n\n\nΙστορία\nΜια πρόταση.", heading)
        text = f"{body}\n\n\nΙστορία των Πηγές και Παραπομπές\nΜια πρόταση."
        self.assertEqual(strip_trailing_sections(text), text)

    def test_greek_parentheticals_are_stripped(self) -> None:
        self.assertEqual(
            split_sentences("Ο Κέλβιν (1824 - 1907) ήταν φυσικός (Σκωτσέζος)."), ["Ο Κέλβιν ήταν φυσικός."]
        )


if __name__ == "__main__":
    unittest.main()
