import unittest
from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch

from src import clue_builder
from src.clue_builder import (
    clue_pass_extract_with_offset,
    clue_pass_mask_trim,
    clue_pass_validate,
    definition_clue,
    enforce_diversity,
    follow_up_definition,
    has_morphological_leakage,
    has_verb,
    is_generic_definition,
    is_thin_definition,
    mask_answer,
    name_the_person,
    split_sentences,
    strip_trailing_sections,
    trim_clause,
    who_definition,
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
        self.assertEqual(split_sentences("The Spy vs. Spy album came out in 1986."), ["The Spy vs. Spy album came out in 1986."])


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
        self.assertIsNone(clue_pass_extract_with_offset("An extension of bebop.", "TENSION")[0])
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

    def test_a_clause_cut_before_its_main_verb_has_no_verb(self) -> None:
        nlp = _load_nlp()
        if nlp is None:
            self.skipTest("spaCy model not available")
        for fragment in (
            "Apart from informing the public about ____ history, ...",
            "The centerpiece of that analysis, now known as a ____, ...",
            "Regarding the ____ revival of the 1940s, ...",
        ):
            self.assertFalse(has_verb(fragment, nlp), fragment)
        for clue in (
            "If the ____ can be measured, ...",  # a modal is finite
            "A rapid succession of rhythm and ____ hits followed, ...",  # tagged a participle, but it has its subject
            # A whole sentence is not held to it: the tagger calls "evolved" and "painted" participles.
            "____ evolved from historical ideas about the vacuum.",
            "During the 1860s, Monet and ____ sometimes painted on canvases prepared with the traditional red-brown or grey ground.",
        ):
            self.assertTrue(has_verb(clue, nlp), clue)

    def test_a_greek_clause_cut_before_its_main_verb_has_no_verb(self) -> None:
        nlp = _load_nlp("el_core_news_sm")
        if nlp is None:
            self.skipTest("Greek spaCy model not available")
        # The tagger calls one underscore of this blank a finite verb.
        self.assertFalse(has_verb("Στην Αστρονομία γενικά ____ ή απλανής, ...", nlp))
        self.assertFalse(has_verb("Χρησιμοποιώντας την ____ ως βάση, ...", nlp))
        self.assertTrue(has_verb("Το μέγεθος που δείχνει, πόσο ισχυρό είναι το ____, ...", nlp))

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

    def test_the_same_short_content_word_leaks_but_function_words_do_not(self) -> None:
        nlp = _load_nlp()
        if nlp is None:
            self.skipTest("spaCy model not available")
        for clue, answer in (
            ("It consists of the visual and plastic arts originating from France", "French art"),
            ("A theoretical gas composed of many randomly moving point particles", "Ideal gas"),
            ("A new city in the south", "New Orleans"),
            ("A big noise in a school of fish", "Big band"),
        ):
            self.assertTrue(has_morphological_leakage(clue, answer, nlp), (clue, answer))
        for model in (None, nlp):
            for clue, answer in (
                ("The bird of the field and of the sky", "The Magpie"),
                ("A town on the river in the north and of the plain", "School of Paris"),
            ):
                self.assertFalse(has_morphological_leakage(clue, answer, model), (clue, answer))

    def test_every_token_of_multi_word_answer_counts(self) -> None:
        self.assertTrue(has_morphological_leakage("The Davis quintet", "Miles Davis", None))

    def test_modifier_letters_and_apostrophes_do_not_hide_the_answer(self) -> None:
        nlp = _load_nlp()
        if nlp is None:
            self.skipTest("spaCy model not available")
        for clue, answer in (
            ("Hawaiʻi, sometimes written ____, is the largest island in the United States, "
             "located in the state of ____.", "Hawaii"),
            ("O’Neill wrote plays about a troubled family", "O'Neill"),
        ):
            self.assertFalse(clue_pass_validate(clue, answer, min_words=3, nlp=nlp), (clue, answer))
        self.assertTrue(
            clue_pass_validate("Eugene O’Hara wrote a play about a family", "O'Neill", min_words=3, nlp=nlp)
        )


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

    def test_two_capitalised_items_are_a_place_name_unless_a_list_of_names_goes_on(self) -> None:
        judaea = (
            "Judaea was a Roman province from 6 to 135 AD, which at its height encompassed the regions of Judea, Idumea, "
            "Peraea, Samaria, and Galilee, as well as parts of the coastal plain of the southern Levant."
        )
        self.assertEqual(definition_clue(judaea, answer="Judaea", max_words=18), "A Roman province from 6 to 135 AD")
        # Or its last item follows the conjunction.
        judea = judaea.replace("Idumea, Peraea, Samaria, and Galilee", "Idumea, and Samaria")
        self.assertEqual(definition_clue(judea, answer="Judaea", max_words=18), "A Roman province from 6 to 135 AD")

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

    def test_the_verb_of_a_relative_clause_or_an_aside_is_not_the_copula(self) -> None:
        habitat = "Habitat, which is the array of resources, is a term of ecology that describes the place where an organism lives."
        self.assertEqual(
            definition_clue(habitat, answer="Habitat"), "A term of ecology that describes the place where an organism lives"
        )
        # The relative clause has its own verb before the copula, and a comma closes it.
        farinelli = "Carlo Broschi, who performed under the stage name Farinelli, was an Italian castrato singer."
        self.assertEqual(definition_clue(farinelli, answer="Farinelli"), "An Italian castrato singer")
        expansion = (
            "In physics, the term expansion means the release of tensions that are accumulated in a system that may "
            "consist of solids, liquids or gases."
        )
        self.assertIsNone(definition_clue(expansion, answer="Expansion"))

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

    def test_a_greek_noun_the_tagger_takes_for_a_function_word_does_not_fail_the_noun_phrase_test(self) -> None:
        def tagged(*tags: str):
            def nlp(clue: str):
                return [SimpleNamespace(text=word, pos_=pos, is_punct=False) for word, pos in zip(clue.split(), tags)]

            return nlp

        el, en = clue_builder._WORDS["el"], clue_builder._WORDS["en"]
        is_noun_phrase = clue_builder._is_noun_phrase
        # ADP that is no preposition, and ADV or VERB in -ος/-ης: nouns.
        self.assertTrue(is_noun_phrase("Γάλλος φιλόσοφος και συγγραφέας", tagged("ADP", "NOUN", "CCONJ", "ADP"), el))
        self.assertTrue(is_noun_phrase("Ολλανδός ανθρωπιστής και γλύπτης", tagged("VERB", "NOUN", "CCONJ", "ADV"), el))
        # A preposition (an archaic one too), an initial, an adverb, a participle and an English tag keep failing.
        self.assertFalse(is_noun_phrase("Γάλλος φιλόσοφος σε", tagged("NOUN", "NOUN", "ADP"), el))
        self.assertFalse(is_noun_phrase("Οπτικό όργανο διάταξης δια", tagged("ADJ", "NOUN", "NOUN", "ADP"), el))
        self.assertFalse(is_noun_phrase("Γάλλος φιλόσοφος Χ", tagged("NOUN", "NOUN", "ADP"), el))
        self.assertFalse(is_noun_phrase("Ευρέως Γάλλος φιλόσοφος", tagged("ADV", "NOUN", "NOUN"), el))
        self.assertFalse(is_noun_phrase("Γεννημένος Γάλλος φιλόσοφος", tagged("VERB", "NOUN", "NOUN"), el))
        self.assertFalse(is_noun_phrase("Γάλλος φιλόσοφος και συγγραφέας", tagged("ADP", "NOUN", "CCONJ", "ADP"), en))


class GenericDefinitionTests(unittest.TestCase):
    # Lead-sentence document frequencies measured on the offline runs of the four test seeds.
    COMMON = {"a": 101, "an": 47, "the": 140, "of": 120, "and": 118, "that": 60, "is": 100, "in": 90, "from": 30, "to": 100}
    POOLS = {
        "quantum": (150, {
            "physicist": 43, "theoretical": 31, "british": 6, "american": 19, "brazilian": 1, "particle": 17,
            "quantum": 62, "elementary": 5, "electromagnetic": 6, "field": 13, "form": 6, "angular": 2,
            "intrinsic": 1, "momentum": 4,
        }),
        "thermo": (82, {"german": 4, "physicist": 12, "mathematician": 4, "process": 9}),
        "jazz": (150, {
            "music": 74, "genre": 38, "musical": 18, "form": 15, "style": 22, "dance": 15, "electronic": 6,
            "subculture": 2, "american": 68, "jazz": 99, "drummer": 4, "composer": 37, "brass": 3, "instrument": 6,
        }),
        "impressionism": (150, {
            "french": 62, "impressionist": 50, "painter": 57, "claude": 20, "monet": 20, "canvas": 9, "by": 30,
        }),
        "rome": (105, {
            "god": 3, "war": 5, "capital": 7, "largest": 4, "city": 13, "lebanon": 1, "gallic": 2,
            "chieftain": 1, "senones": 1, "ancient": 21, "tribe": 2, "europe": 6, "roman": 40, "emperor": 25,
        }),
    }

    def _generic(self, pool: str, clue: str, nlp=None, seed_title: str | None = None) -> bool:
        size, counts = self.POOLS[pool]
        return is_generic_definition(clue, Counter({**self.COMMON, **counts}), size, nlp, seed_title=seed_title)

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
        clue = "An American-Brazilian polymath and physicist"
        self.assertFalse(self._generic("quantum", clue))
        self.assertTrue(self._generic("quantum", clue, nlp))

    def test_a_nationality_asks_for_a_second_informative_word(self) -> None:
        nlp = _load_nlp()
        if nlp is None:
            self.skipTest("spaCy model not available")
        # "drummer" is rare, but "An American jazz drummer" names a crowd; "brass" alone tells the trumpet apart.
        self.assertTrue(self._generic("jazz", "An American jazz drummer and composer", nlp))
        self.assertFalse(self._generic("jazz", "A brass instrument", nlp))
        # The nationality of someone else, after a preposition, asks for nothing.
        monet = "An 1881 canvas by the French Impressionist painter Claude Monet"
        self.assertFalse(self._generic("impressionism", monet, nlp))
        with patch.object(clue_builder, "_MIN_INFORMATIVE_WORDS", 1):
            self.assertFalse(self._generic("jazz", "An American jazz drummer and composer", nlp))

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

    def test_numbers_and_seed_words(self) -> None:
        self.assertFalse(self._generic("rome", "Roman emperor from 81 to 96"))
        self.assertTrue(self._generic("rome", "Roman emperor from the city"))
        # "thermodynamic" tells nothing in a pool seeded with Thermodynamics.
        self.assertFalse(self._generic("thermo", "A thermodynamic process"))
        self.assertTrue(self._generic("thermo", "A thermodynamic process", seed_title="Thermodynamics"))

    def test_a_head_noun_that_names_a_word_is_generic_unless_it_says_what_the_word_means(self) -> None:
        for clue in (
            "A term coined in the 1980s",
            "A jazz term with several possible definitions and usages",
            "A fundamental concept in the field of thermodynamics and engineering",
            "The name given to oil painters and others",
        ):
            self.assertTrue(self._generic("jazz", clue), clue)
        for clue in (
            "An emblematic phrase referring to the government of the Roman Republic",
            "A scientific concept, most commonly associated with states of disorder, randomness, or uncertainty",
            "A term encompassing a number of closely related results in physics",
        ):
            self.assertFalse(self._generic("jazz", clue), clue)

    def test_a_kind_of_word_that_only_modifies_the_head_is_not_the_head(self) -> None:
        nlp = _load_nlp()
        if nlp is None:
            self.skipTest("spaCy model not available")
        clue = "A word processor developed by Microsoft"
        self.assertTrue(self._generic("jazz", clue))
        self.assertFalse(self._generic("jazz", clue, nlp))

    def test_a_nationality_on_a_kind_of_word_names_a_language(self) -> None:
        nlp = _load_nlp()
        if nlp is None:
            self.skipTest("spaCy model not available")
        counts = Counter({
            **self.COMMON, "english": 3, "common": 1, "title": 4, "series": 20, "paintings": 30, "claude": 20,
            "monet": 20, "impressionist": 50, "painter": 57,
        })
        haystacks = "The common English title for a series of impressionist paintings by Claude Monet"
        self.assertFalse(is_generic_definition(haystacks, counts, 150, nlp, seed_title="Impressionism"))
        # The answer's own nationality still asks for a second informative word.
        self.assertTrue(is_generic_definition("A common English painter", counts, 150, nlp, seed_title="Impressionism"))

    def test_a_vague_quantifier_opens_a_generic_definition(self) -> None:
        for clue in (
            "Any of many different quantities",
            "One of several unrelated wave phenomena",
            "Various small birds",
            "A wide range of physical processes",
            "A large group of musicians",
            "A separate part of a larger system",
            "The component of a bigger machine",
        ):
            self.assertTrue(self._generic("quantum", clue), clue)
        # What follows the group pins it down.
        for clue in (
            "A group of young French artists active in Paris from 1888 until 1900",
            "A large group of Germanic peoples first reported by Julius Caesar in the 1st century BC",
            "A part of the larger Hellenistic world",
        ):
            self.assertFalse(self._generic("impressionism", clue), clue)
        for clue in (
            "One of the earliest foundational tunes of Afro-Cuban jazz",
            "One of the world's earliest sewage systems",
        ):
            self.assertFalse(self._generic("jazz", clue), clue)

    def test_a_definition_is_thin_only_when_its_informative_words_are_too_few(self) -> None:
        size, counts = self.POOLS["thermo"]
        counts = Counter({**self.COMMON, **counts})
        self.assertTrue(is_thin_definition("A German physicist", counts, size))
        for clue in ("Any of many different quantities", "A term coined in the 1980s"):
            self.assertTrue(is_generic_definition(clue, counts, size), clue)
            self.assertFalse(is_thin_definition(clue, counts, size), clue)

    def test_a_person_is_named_by_the_given_names_of_the_page_title_and_a_blank_for_the_surname(self) -> None:
        for title, answer, expected in (
            ("Art Blakey", "Blakey", "An American jazz drummer (Art ____)"),
            ("Benny Green (pianist)", "Green", "An American jazz drummer (Benny ____)"),
            ("Howard M. Wiseman", "Wiseman", "An American jazz drummer (Howard M. ____)"),
        ):
            self.assertEqual(name_the_person("An American jazz drummer", title, answer), expected, title)
        self.assertEqual(name_the_person("Ένας Γάλλος φιλόσοφος", "Ζαν-Πωλ Σαρτρ", "Σαρτρ"), "Ένας Γάλλος φιλόσοφος (Ζαν-Πωλ ____)")
        # Another name of the person, a title of one word, and a clue past the word limit have no such form.
        self.assertIsNone(name_the_person("An American jazz drummer", "Art Blakey", "Art"))
        self.assertIsNone(name_the_person("An American jazz drummer", "Blakey", "Blakey"))
        self.assertIsNone(name_the_person(" ".join(["word"] * 17), "Art Blakey", "Blakey"))

    def test_trailing_sections_are_cut_at_the_first_of_them(self) -> None:
        body = "Weinberg won a prize."
        headings = (
            "See also", "Notes", "References", "Citations", "Sources", "Bibliography", "Further reading",
            "External links", "Works", "Publications", "Selected works", "Selected publications", "Published works",
            "Discography", "Filmography",
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
            "Hence, no real heat engine could realize the ____'s reversibility and was condemned to be less efficient.",
            "Therefore, even at ____, atoms and molecules retain some vibrational motion.",
            "However, AllMusic separates it from ____ because of R&B's stronger gospel influences.",
            "Rather like ____'s statement is that of George Uhlenbeck and G. W. Ford for irreversible phenomena.",
            "Αυτό είναι το ____ του συστήματος.",
            "Και η ____ μεταφέρεται από το ένα σώμα στο άλλο.",
            "Πρόκειται για την ____ του συστήματος.",
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


class FillInReferenceTests(unittest.TestCase):
    NEEDS_NO_TAGGER = (
        "In 1870 he introduced the virial theorem, which applied to ____.",
        "One proposal for doing so is ____ theory, which is popular.",
        "The formula for this calculation is known as the ____.",
        "The former is stochastic and discontinuous, writes von Neumann, and the latter ____ and continuous.",
        "Using this pump, Boyle and Hooke noticed a correlation between pressure, temperature, and ____.",
        'The following pentatonic lick was played over ____ changes by Joe Henderson on Horace Silver\'s "African Queen".',
        "For this reason, some philosophers of science have deemed ____ a form of anti-realism.",
        "Writing a ____ does not provide the complete information necessary to describe this state-change process.",
        "The team said it did so and the ____ rose.",
        "____ elements can be represented likewise, ...",
        "As noted above, jazz has incorporated from its inception aspects of African-American sacred music including ____ and hymns.",
        "Each ____ is, as described below, a measure of the disorder in a closed system.",
        "The ____ of the gas can be found as mentioned earlier from its pressure and volume.",
        "For the proof see below, where the ____ is defined for an ideal gas.",
        "The aforementioned ____ is a measure of the disorder in a closed system.",
        "The above-mentioned ____ is a measure of the disorder in a closed system.",
    )
    KEPT = (
        "The balloon rose above the ____ and drifted slowly away from the village.",
        "Divers who went below the ____ found a cave that nobody had explored before.",
        "Segments of the ____ are often described as walls; they have respective defined 'permeabilities'.",
        "Heraclius, son of the exarch of ____, set sail for the city and assumed the throne.",
        "The wolf was the sacred animal of ____, with the she-wolf nursing the two founders as children.",
        "Mithridates VI was the ruler of Pontus, a large kingdom in ____, from 120 to 63 BC.",
        "____'s 1954 performance of \"Walkin'\" at the first Newport Jazz Festival introduced the style to the jazz world.",
        "The ____ of the pump was designed so this pump can run, and the pump is old.",
    )

    @staticmethod
    def _passes(clue: str, nlp=None, lang: str = "en") -> bool:
        return clue_pass_validate(clue, "Entropy", min_words=3, nlp=nlp, require_blank=True, lang=lang)

    def test_a_fill_in_that_leans_on_an_earlier_sentence_is_rejected(self) -> None:
        for nlp in (None, _load_nlp()):
            for clue in self.NEEDS_NO_TAGGER:
                with self.subTest(clue=clue, tagger=nlp is not None):
                    self.assertFalse(self._passes(clue, nlp))

    def test_a_pronoun_with_a_noun_before_it_is_kept(self) -> None:
        for clue in self.KEPT:
            with self.subTest(clue=clue):
                self.assertTrue(self._passes(clue))
        self.assertTrue(self._passes("Heat flows when it is wanted, and he knows the ____."))

    def test_the_tagger_sees_a_pronoun_after_a_function_word_and_a_noun_before_it(self) -> None:
        nlp = _load_nlp()
        if nlp is None:
            self.skipTest("spaCy model not available")
        for clue in self.KEPT:
            with self.subTest(clue=clue):
                self.assertTrue(self._passes(clue, nlp))
        # Without a tagger "Because" counts as a noun that "it" could refer to.
        clue = "Because of this, it is often important to distinguish between heterogeneous ____ and homogeneous ____."
        self.assertTrue(self._passes(clue))
        self.assertFalse(self._passes(clue, nlp))

    def test_a_his_or_her_with_a_pronoun_subject_and_nobody_named_is_a_back_reference(self) -> None:
        nlp = _load_nlp()
        if nlp is None:
            self.skipTest("spaCy model not available")
        clue = "On his 1961 tour of ____, he was booed, but persevered, signing with the new Impulse!"
        self.assertFalse(self._passes(clue, nlp))
        self.assertTrue(self._passes(clue))  # the rule needs the tagger
        for kept in (
            "In his 1905 paper, Einstein proposed the ____ of light.",
            "In her memoir, ____ describes the years in Paris.",
            "Coltrane recorded ____ with his quartet, and he toured.",
        ):
            with self.subTest(clue=kept):
                self.assertTrue(self._passes(kept, nlp))

    def test_the_term_that_an_earlier_sentence_named_is_a_back_reference(self) -> None:
        nlp = _load_nlp()
        if nlp is None:
            self.skipTest("spaCy model not available")
        for clue in (
            "Originally, the term was confined to light ____.",
            'The term is sometimes also used as synonymous with "____" and is defined as such in some dictionaries.',
            "In 1910 the name has been given to ____ by many authors.",
        ):
            with self.subTest(clue=clue):
                self.assertFalse(self._passes(clue, nlp))
                self.assertTrue(self._passes(clue))  # the rule needs the tagger
        for clue in (
            'The term "jazz" was first used for the music of ____ in Chicago.',
            "Here the term ____ was used for the music of New Orleans.",
            "The name of the band was ____ until 1960.",
            "The Goth sub-culture continued the tradition, echoing the name and reputation of the ____ outsider Goths.",
        ):
            with self.subTest(clue=clue):
                self.assertTrue(self._passes(clue, nlp))
        # The tagger takes "entropy" for a verb; the sentence still says which word.
        clue = "I have designedly coined the word entropy to be similar to '____', ..."
        self.assertTrue(clue_pass_validate(clue, "Energy", min_words=3, nlp=nlp, require_blank=True))

    def test_greek_fill_ins_that_lean_on_an_earlier_sentence_are_rejected(self) -> None:
        for clue in (
            "Με αυτή τη μέθοδο υπολογίζεται η ____ του αερίου.",
            "Υπολογίζονται τα ακόλουθα μεγέθη: η ____ και η πίεση.",
            "Οι παρακάτω τύποι δίνουν την ____ του συστήματος.",
            "Ο πρώτος νόμος αφορά την ενέργεια, ο δεύτερος την ____.",
            "Όπως αναφέρθηκε, η ____ μεταφέρεται από το ένα σώμα στο άλλο.",
            "Η προαναφερθείσα ____ μεταφέρεται από το ένα σώμα στο άλλο.",
            "Η ____ μεταφέρεται, όπως είδαμε, από το ένα σώμα στο άλλο.",
            "Η μεταφορά της ____ δίνεται από τον τύπο που αναφέρεται ανωτέρω.",
        ):
            with self.subTest(clue=clue):
                self.assertFalse(self._passes(clue, lang="el"))
        for clue in (
            "Η ____ είναι αυτό που μετράμε στο σύστημα.",
            "Ο πρώτος νόμος της θερμοδυναμικής αφορά την ____.",
            "Η ____ μεταφέρεται από το ένα σώμα στο άλλο.",
        ):
            with self.subTest(clue=clue):
                self.assertTrue(self._passes(clue, lang="el"))

    def test_a_greek_verb_without_a_subject_or_a_possessive_without_a_noun_points_outside(self) -> None:
        nlp = _load_nlp("el_core_news_sm")
        if nlp is None:
            self.skipTest("Greek spaCy model not available")
        rejected = (
            "Εφαρμόζεται στην κοσμολογία και την ____, συμπεριλαμβανομένης της αστρονομίας.",
            "Μέσα από τους νόμους εισάγονται δύο κεντρικές της έννοιες, η ____ και η εντροπία.",
        )
        kept = (
            "Η ____ και οι εφαρμογές της εξελίσσονται σήμερα, με ηγέτη τις Η.Π.Α., ...",  # της refers to the blank
            "Ο μοναδικός φυσικός δορυφόρος της Γης είναι η ____.",  # της is an article
            "Το 594 π.Χ. ο Σόλων ο Αθηναίος καθιέρωσε το ____ ως μέτρηση του χρόνου.",
            "Η ____ ήταν ο πρώτος κλάδος της φυσικής.",
            "Ο Ήλιος είναι ο κοντινότερος ____ στη Γη.",
            "Γενικά, η ____ γεννήθηκε με την εμφάνιση του «διανοούμενου ανθρώπου» στον ημέτερο πλανήτη.",
            "Μπορούμε, όμως, να προσεγγίσουμε το ____ εντελώς οριακά, ...",  # "we" is not in the previous sentence
        )
        for clue in rejected:
            with self.subTest(clue=clue):
                self.assertFalse(self._passes(clue, nlp, lang="el"))
                self.assertTrue(self._passes(clue, lang="el"))  # without a tagger they are not caught
        for clue in kept:
            with self.subTest(clue=clue):
                self.assertTrue(self._passes(clue, nlp, lang="el"))

    def test_a_greek_demonstrative_points_outside_before_or_after_its_noun(self) -> None:
        rejected = (
            "Λόγω της εμφάνισης αυτής, ονομάστηκε στα ελληνικά «____ κύκλος» ή και «γάλακτος κύκλος» ή και σκέτο «γάλα».",
            "Σύμφωνα με την αντίληψη αυτή, η ____, αντιμετωπίζεται ως ψευδεπιστήμη.",
            "Ιδιαίτεροι κλάδοι αυτής της Επιστήμης είναι: η Πρακτική ____, η Δυναμική ____, ...",
            # The article came before, the noun did not.
            "Τη θερμότητα μετρά το θερμιδόμετρο, και με αυτή τη μέθοδο υπολογίζεται η ____.",
        )
        kept = (
            # The noun came before, in another case.
            "Η εμφάνιση του είναι γαλακτώδης, και λόγω της εμφάνισης αυτής ονομάστηκε ____.",
            "Η μέθοδος είναι απλή, και με αυτή τη μέθοδο υπολογίζεται η ____.",
            # "that of light" and "the one who": pronouns that point inside the clue.
            "Σε περιπτώσεις που η ταχύτητα των αντικειμένων πλησιάζει αυτήν του φωτός, η ____ επεκτείνεται από την Ειδική Σχετικότητα.",
            "Ο Κέπλερ ήταν αυτός που διατύπωσε τους νόμους της ____.",
        )
        for nlp in (None, _load_nlp("el_core_news_sm")):
            for clue in rejected:
                with self.subTest(clue=clue, tagger=nlp is not None):
                    self.assertFalse(self._passes(clue, nlp, lang="el"))
            for clue in kept:
                with self.subTest(clue=clue, tagger=nlp is not None):
                    self.assertTrue(self._passes(clue, nlp, lang="el"))


class FillInListTests(unittest.TestCase):
    """A blank that is one item of a list of three or more: any of the other items would fit."""

    # (answer, original sentence, language). Two commas and a coordinator: seen with and without the tagger.
    LISTS = (
        (
            "Paul Cézanne",
            "Modern art begins with the post-impressionist painters like Vincent van Gogh, Paul Cézanne, Paul Gauguin, "
            "Georges Seurat and Henri de Toulouse-Lautrec.",
            "en",
        ),
        ("Hip-hop", "The mid-2010s saw an increased influence of R&B, hip-hop, and pop music on jazz.", "en"),
        ("Temperature", "James Joule studied the relationship between heat, work, and temperature.", "en"),
        (
            "Thelonious Monk",
            "Charlie Christian, Dizzy Gillespie, Charlie Parker, and Thelonious Monk were among the major innovators "
            "of the style.",
            "en",
        ),
        (
            "Χημεία",
            "Η Φασματοσκοπία, που έχει αστρονομική εξ ανάγκης προέλευση, χρησιμοποιείται σήμερα από τη Φυσική, Χημεία, "
            "Μεταλλουργία, Βιολογία κλπ.",
            "el",
        ),
        (
            "Ιστορία",
            "Ο όρος «πλανήτης» είναι αρχαίος, με δεσμούς με την ιστορία, με την αστρολογία, με την επιστήμη, με τη "
            "μυθολογία και με τη θρησκεία.",
            "el",
        ),
        ("Γαλαξίες", "Ο Σαρλ Μεσιέ κατάρτισε κατάλογο με τα απομακρυσμένα αντικείμενα όπως Γαλαξίες, Νεφελώματα, κ.ά..", "el"),
        ("Γεωδαισία", "Ακόμα η Χρονομετρία, η Ναυτιλία, και η Γεωδαισία σχετίζονται στενά με την Αστρονομία.", "el"),
    )
    # One comma: a list only for the parser, which coordinates the answer with two other words.
    LISTS_FOR_THE_PARSER = (
        ("Sicily", "The annexation of Egypt, Sicily and Tunisia in North Africa provided a continuous supply of grains.", "en"),
        ("Renoir", "In 1874, Monet, Renoir and Sisley exhibited together.", "en"),
    )
    NO_LISTS = (
        # A pair is not a list, whatever stands before it.
        (
            "Renoir",
            "During the 1860s, Monet and Renoir sometimes painted on canvases prepared with the traditional red-brown "
            "or grey ground.",
            "en",
        ),
        ("Sicily", "The annexation of Sicily and Tunisia in North Africa provided a continuous supply of grains.", "en"),
        ("Ενέργεια", "Μέσα από τους νόμους εισάγονται δύο κεντρικές έννοιες, η ενέργεια και η εντροπία.", "el"),
        ("Φιλοσοφία", "Τέλος η Αστρονομία σχετίζεται στενά με τη Φιλοσοφία και τη Μεταφυσική.", "el"),
        # A comma after the blank that opens no list.
        (
            "Limes",
            "The assembled warbands of the Alamanni frequently crossed the limes, attacking Germania Superior such "
            "that they were almost continually engaged in conflicts with the Roman Empire, whilst both Goths and "
            "Heruli ravaged the Aegean and, later, Greece, Thrace and Macedonia.",
            "en",
        ),
        ("Curia", "In medieval times, a king's council was often referred to as a curia.", "en"),
        # A list the blank is no item of.
        ("Mars", "The wolf was the sacred animal of Mars, with the she-wolf nursing the two founders as children.", "en"),
        ("Sicily", "In Sicily, wine, oil and grain were the main exports.", "en"),
        ("Thermodynamics", "Most often, in thermodynamics, mass and energy are treated as separately conserved.", "en"),
        # The blank is part of an item that says more.
        ("Olive", "In turn, olive oil and wine were Italy's main exports.", "en"),
        (
            "Herbie Hancock",
            "Popular modal standards include Davis's \"All Blues\" and \"So What\", John Coltrane's \"Impressions\" "
            "and Herbie Hancock's \"Maiden Voyage\".",
            "en",
        ),
    )
    # The parser alone tells these from a list: an item with a modifier of its own, and an apposition.
    NO_LISTS_FOR_THE_PARSER = (
        ("Optics", "Engineers have taken an interest in integrated optics, machine vision, and photonic computing.", "en"),
        ("Titus", "The emperor's son, Titus, conquered Jerusalem and Masada.", "en"),
    )

    @staticmethod
    def _passes(answer: str, sentence: str, lang: str, nlp) -> bool:
        clue, _ = clue_pass_mask_trim(sentence, answer, max_words=12)
        return clue_pass_validate(clue, answer, min_words=6, nlp=nlp, original_sentence=sentence, require_blank=True, lang=lang)

    def _check(self, cases, expected: bool, *, tagger: bool) -> None:
        for answer, sentence, lang in cases:
            nlp = _load_nlp("el_core_news_sm" if lang == "el" else "en_core_web_sm") if tagger else None
            if tagger and nlp is None:
                continue
            with self.subTest(answer=answer, tagger=tagger):
                self.assertEqual(self._passes(answer, sentence, lang, nlp), expected)

    def test_an_item_of_a_list_is_rejected_with_and_without_the_tagger(self) -> None:
        self._check(self.LISTS, False, tagger=False)
        self._check(self.LISTS, False, tagger=True)

    def test_the_cut_clue_cannot_hide_the_list(self) -> None:
        answer, sentence, _ = self.LISTS[0]
        clue, _ = clue_pass_mask_trim(sentence, answer, max_words=12)
        self.assertEqual(clue, "Modern art begins with the post-impressionist painters like Vincent van Gogh, ____, ...")
        self.assertTrue(clue_pass_validate(clue, answer, min_words=6, require_blank=True))
        self.assertFalse(clue_pass_validate(clue, answer, min_words=6, original_sentence=sentence, require_blank=True))

    def test_a_list_with_one_comma_needs_the_parser(self) -> None:
        self._check(self.LISTS_FOR_THE_PARSER, True, tagger=False)
        self._check(self.LISTS_FOR_THE_PARSER, False, tagger=True)

    def test_pairs_and_other_commas_are_not_lists(self) -> None:
        self._check(self.NO_LISTS, True, tagger=False)
        self._check(self.NO_LISTS, True, tagger=True)

    def test_the_parser_keeps_an_item_with_a_modifier_and_an_apposition(self) -> None:
        self._check(self.NO_LISTS_FOR_THE_PARSER, False, tagger=False)
        self._check(self.NO_LISTS_FOR_THE_PARSER, True, tagger=True)


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
        # "Roman, Byzantine," is two items of a list that goes on, not a place name: no cut after it, with or without a parser.
        self.assertEqual(definition_clue(city), "A historical city located on the Bosporus")
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

    def test_and_before_a_new_preposition_opens_a_tail(self) -> None:
        mainz = (
            "Mainz is the capital and largest city of the German state of Rhineland-Palatinate, and with around "
            "223,000 inhabitants, it is Germany's 35th-largest city."
        )
        self.assertEqual(
            definition_clue(mainz, answer="Mainz"), "The capital and largest city of the German state of Rhineland-Palatinate"
        )
        # A comma after another one closes a list: no cut before its last item.
        time = (
            "Time is the continuous progression of existence that occurs in an apparently irreversible succession "
            "from the past, through the present, and into the future."
        )
        self.assertIsNone(definition_clue(time, answer="Time", max_words=18))
        # A preposition used before continues a list: "in his native country" does not end the definition.
        weill = (
            "Kurt Julian Weill was a German-born American composer active from the 1920s in his native country, "
            "and in his later years in the United States."
        )
        self.assertEqual(definition_clue(weill, answer="Kurt Weill"), "A German-born American composer active from the 1920s")

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

    def test_a_tail_opened_by_with_or_a_participle_is_dropped_when_the_cut_leaves_its_noun_unfinished(self) -> None:
        nlp = _load_nlp()
        if nlp is None:
            self.skipTest("spaCy model not available")
        for sentence, answer, expected in (
            (
                "Boden's mate is a checkmating pattern in chess characterized by bishops on two criss-crossing diagonals, with "
                "possible flight squares for the king being occupied by friendly pieces or under attack by enemy pieces.",
                "Boden's Mate",
                "A checkmating pattern in chess characterized by bishops on two criss-crossing diagonals",
            ),
            (
                "Macbeth is an opera in four acts by Giuseppe Verdi, with an Italian libretto by Francesco Maria Piave and "
                "additions by Andrea Maffei, based on William Shakespeare's play of the same name.",
                "Macbeth",
                "An opera in four acts by Giuseppe Verdi",
            ),
            (
                "Evansia is a quarterly, peer-reviewed scientific journal, publishing research on issues in biology and "
                "environmental preservation related to lichenology and bryology, primarily in North America.",
                "Evansia",
                "A quarterly, peer-reviewed scientific journal",
            ),
            (
                "Christiaan Huygens was a Dutch mathematician, physicist, engineer, astronomer, and inventor who is regarded as a "
                "key figure in the Scientific Revolution.",
                "Huygens",
                "A Dutch mathematician, physicist, engineer, astronomer, and inventor",
            ),
            # A tail that goes on with something else than a preposition of its last noun is whole, and so is a relative clause.
            (
                "Alan Cyril Crundwell was a British bryologist, known for categorizing approximately 20 species of "
                "bryophytes, around six of which were previously unknown to science.",
                "Crundwell",
                "A British bryologist, known for categorizing approximately 20 species of bryophytes",
            ),
            (
                "Garry Kasparov is a Russian chess grandmaster, political activist and writer, who was the World Chess "
                "Champion from 1985 to 2000.",
                "Kasparov",
                "A Russian chess grandmaster, political activist and writer, who was the World Chess Champion",
            ),
        ):
            with self.subTest(answer=answer):
                self.assertEqual(definition_clue(sentence, answer=answer, nlp=nlp), expected)

    def test_the_cut_runs_up_to_the_relative_clause_when_that_is_the_longest_clean_end(self) -> None:
        morisot = (
            "Berthe Marie Pauline Morisot was a French painter, printmaker and a member of the circle of painters "
            "in Paris who became known as the Impressionists."
        )
        self.assertEqual(
            definition_clue(morisot, answer="Berthe Morisot"),
            "A French painter, printmaker and a member of the circle of painters in Paris",
        )
        # The clause is no part of the definition after a comma, which has its cut, or a participle that wants it.
        spirituals = (
            "Spirituals is a genre of Christian music that is associated with African Americans, which merged varied "
            "African cultural influences with the experiences of being held in bondage in slavery."
        )
        self.assertEqual(
            definition_clue(spirituals, answer="Spirituals"), "A genre of Christian music that is associated with African Americans"
        )
        mind = (
            "Quantum mind is a large group of theoretical hypotheses in physics proposing that local physical laws "
            "and interactions from classical mechanics alone cannot explain consciousness."
        )
        self.assertEqual(definition_clue(mind, answer="Quantum mind"), "A large group of theoretical hypotheses")

    def test_and_followed_by_a_comma_opens_an_aside_that_is_cut_off(self) -> None:
        hines = (
            "Hines was one of the most influential figures in the development of jazz piano and, according to one "
            "source, one of a small number of pianists whose playing shaped the history of jazz."
        )
        self.assertEqual(
            definition_clue(hines, answer="Earl Hines"), "One of the most influential figures in the development of jazz piano"
        )

    def test_a_predicate_that_opens_on_an_adjective_with_its_complement_is_not_a_noun_phrase(self) -> None:
        nlp = _load_nlp()
        if nlp is None:
            self.skipTest("spaCy model not available")
        lead = "Jump blues was popular in the 1940s and was a precursor of rhythm and blues."
        self.assertEqual(definition_clue(lead), "Popular in the 1940s")
        self.assertIsNone(definition_clue(lead, nlp=nlp))

    def test_a_definition_stops_at_a_colon(self) -> None:
        density = (
            "A density matrix is a generalization of the state vectors or wavefunctions: while those can only "
            "represent pure states, density matrices can also represent mixed states."
        )
        self.assertEqual(definition_clue(density, answer="Density matrix"), "A generalization of the state vectors or wavefunctions")
        # What is left before the colon must still end cleanly.
        self.assertIsNone(definition_clue("Heat is a form of energy such as: electrical, thermal and chemical energy.", answer="Heat"))

    def test_a_cut_never_falls_inside_a_name(self) -> None:
        nlp = _load_nlp()
        if nlp is None:
            self.skipTest("spaCy model not available")
        kent = (
            "Adrian Kent is a professor at the University of Cambridge, member of the Centre for Quantum Information "
            "and Foundations, and a fellow of the Institute."
        )
        self.assertEqual(definition_clue(kent, answer="Adrian Kent", nlp=nlp), "A professor at the University of Cambridge")
        # "Honorius at Mediolanum" is a person and a place: the cut may fall between them.
        claudian = (
            "Claudius Claudianus, known in English as Claudian, was a Latin poet of late antiquity associated with the court of "
            "the Roman emperor Honorius at Mediolanum, and particularly with the general Stilicho."
        )
        self.assertEqual(
            definition_clue(claudian, answer="Claudian", nlp=nlp),
            "A Latin poet of late antiquity associated with the court of the Roman emperor Honorius",
        )
        # A date is not cut in two either.
        blue = "Kind of Blue is a studio album by American jazz musician Miles Davis, released on August 17, 1959, by Columbia Records."
        self.assertEqual(definition_clue(blue, answer="Kind of Blue", nlp=nlp), "A studio album by American jazz musician Miles Davis")

    def test_the_tagger_finds_a_second_predicate_after_a_noun_phrase_but_not_inside_a_relative_clause(self) -> None:
        nlp = _load_nlp()
        if nlp is None:
            self.skipTest("spaCy model not available")
        monet = "Jean Monet was the subject of several paintings by his father and married his step-sister, Blanche Hoschedé."
        self.assertEqual(definition_clue(monet, answer="Jean Monet", nlp=nlp), "The subject of several paintings by his father")
        # Without a tagger only the listed verbs ("was", "is", "also") start one.
        self.assertEqual(
            definition_clue(monet, answer="Jean Monet"),
            "The subject of several paintings by his father and married his step-sister, Blanche Hoschedé",
        )
        matter = "Matter is anything that has mass and occupies space by having volume."
        self.assertEqual(definition_clue(matter, answer="Matter", nlp=nlp), "Anything that has mass and occupies space by having volume")

    def test_a_comma_and_one_word_that_lost_its_complement_is_dropped(self) -> None:
        boudin = "Eugène Louis Boudin was a marine painter, and expert in the rendering of all that goes upon the sea and along its shores."
        spin = (
            "Spin is an intrinsic form of angular momentum carried by elementary particles, and thus by composite "
            "particles such as hadrons, atomic nuclei, and atoms."
        )
        for nlp in (None, _load_nlp()):
            self.assertIsNone(definition_clue(boudin, answer="Eugène Boudin", nlp=nlp))
            self.assertEqual(
                definition_clue(spin, answer="Spin", nlp=nlp), "An intrinsic form of angular momentum carried by elementary particles"
            )

    def test_a_relative_clause_with_its_object_may_end_before_a_time_adjunct(self) -> None:
        nlp = _load_nlp()
        if nlp is None:
            self.skipTest("spaCy model not available")
        lepidus = (
            "Marcus Aemilius Lepidus was a Roman general and statesman who formed the Second Triumvirate alongside "
            "Octavian and Mark Antony during the final years of the Roman Republic."
        )
        self.assertEqual(
            definition_clue(lepidus, answer="Lepidus", nlp=nlp),
            "A Roman general and statesman who formed the Second Triumvirate alongside Octavian and Mark Antony",
        )
        self.assertEqual(definition_clue(lepidus, answer="Lepidus"), "A Roman general and statesman")
        # "in" may belong to the noun ("a role in ..."): the clause is not whole before it.
        gell_mann = (
            "Murray Gell-Mann was an American theoretical physicist who played a preeminent role in the development "
            "of the theory of elementary particles."
        )
        self.assertEqual(definition_clue(gell_mann, answer="Murray Gell-Mann", nlp=nlp), "An American theoretical physicist")

    def test_a_relative_clause_may_end_before_a_preposition_with_a_date(self) -> None:
        nlp = _load_nlp()
        if nlp is None:
            self.skipTest("spaCy model not available")
        funk = (
            "Funk is a music genre that originated in African-American communities in the mid-1960s when musicians "
            "created a rhythmic and danceable new form of music."
        )
        self.assertEqual(definition_clue(funk, answer="Funk", nlp=nlp), "A music genre that originated in African-American communities")
        # Without a parser nothing tells a date from another noun.
        self.assertIsNone(definition_clue(funk, answer="Funk"))

    def test_a_cut_may_stop_right_before_the_word_at_the_limit(self) -> None:
        nlp = _load_nlp()
        if nlp is None:
            self.skipTest("spaCy model not available")
        botany = (
            "Botany, also called plant science, is the branch of natural science and biology that studies the growth, "
            "reproduction and evolution of plants over the last 3.5 billion years of life on earth."
        )
        self.assertEqual(
            definition_clue(botany, answer="Botany", nlp=nlp),
            "The branch of natural science and biology that studies the growth, reproduction and evolution of plants",
        )
        # A cut that ends on an adverb is no candidate: the shorter one is still there.
        synergetics = (
            "Synergetics is an interdisciplinary science explaining the formation and self-organization of patterns and "
            "structures in open systems far from thermodynamic equilibrium."
        )
        self.assertEqual(
            definition_clue(synergetics, answer="Synergetics", nlp=nlp),
            "An interdisciplinary science explaining the formation and self-organization of patterns and structures",
        )

    def test_a_cut_never_falls_between_a_noun_and_the_preposition_it_wants(self) -> None:
        duncan = (
            "Ursula Katherine Duncan was a botanist with a special interest in mosses and lichens, and a lifelong love "
            "and knowledge of flowers."
        )
        self.assertIsNone(definition_clue(duncan, answer="Ursula Katherine Duncan"))
        self.assertEqual(
            definition_clue(duncan, answer="Ursula Katherine Duncan", max_words=18),
            "A botanist with a special interest in mosses and lichens, and a lifelong love and knowledge of flowers",
        )
        # "to an Italian libretto" is a complete ending.
        giovanni = "Don Giovanni is an opera in two acts with music by Wolfgang Amadeus Mozart to an Italian libretto by Lorenzo Da Ponte."
        self.assertEqual(
            definition_clue(giovanni, answer="Don Giovanni"), "An opera in two acts with music by Wolfgang Amadeus Mozart to an Italian libretto"
        )

    def test_the_head_of_a_person_may_stop_at_a_comma_with_fewer_than_five_words(self) -> None:
        vale = (
            "Ana Vale was a Danish physicist, one of the main pioneers of the theory of heat flow and a principal "
            "scientist of the Kyoto program of the war effort."
        )
        self.assertIsNone(definition_clue(vale, answer="Vale"))
        self.assertEqual(definition_clue(vale, answer="Vale", person=True), "A Danish physicist")
        # Before a preposition a cut still keeps five words.
        fraser = (
            "Alex Fraser was a major innovator in the development of the computer modeling of population genetics and "
            "his work has stimulated many advances in genetic research."
        )
        self.assertIsNone(definition_clue(fraser, answer="Fraser", person=True))

    def test_the_head_of_a_person_stops_at_the_only_comma_before_and_a_determiner(self) -> None:
        oller = (
            "Francisco Manuel Oller y Cestero was a Puerto Rican painter, and the only Latin American artist to have played "
            "a role in the development of Impressionism."
        )
        self.assertEqual(definition_clue(oller, answer="Oller", person=True), "A Puerto Rican painter")
        # Not a list: the next comma goes on with it.
        listed = "Ana Vale was a Danish painter, and the first woman to teach, to write, and to lead the national school of art."
        self.assertIsNone(definition_clue(listed, answer="Vale", person=True))

    def test_wherein_opens_a_relative_clause_and_which_after_a_preposition_has_no_verb_yet(self) -> None:
        zugzwang = (
            "Zugzwang is a situation found in chess and other turn-based games wherein one player is put at a disadvantage "
            "because of their obligation to make a move."
        )
        self.assertEqual(definition_clue(zugzwang, answer="Zugzwang"), "A situation found in chess and other turn-based games")
        pointillism = "Pointillism is a technique of painting in which small, distinct dots of color are applied in patterns to form an image."
        self.assertIsNone(definition_clue(pointillism, answer="Pointillism"))

    def test_rich_wants_its_preposition(self) -> None:
        basalt = (
            "Basalt is an aphanitic extrusive igneous rock formed from the rapid cooling of low-viscosity lava rich in "
            "magnesium and iron exposed at or very near the surface of a rocky planet or moon."
        )
        self.assertEqual(
            definition_clue(basalt, answer="Basalt"), "An aphanitic extrusive igneous rock formed from the rapid cooling of low-viscosity lava"
        )

    def test_a_cut_does_not_end_on_a_time_noun_that_waits_for_its_complement(self) -> None:
        laine = (
            'John Joseph "Papa Jack" Laine was an American musician and a pioneering band leader in New Orleans '
            "in the years from the Spanish\u2013American War to World War I."
        )
        self.assertEqual(definition_clue(laine), "An American musician and a pioneering band leader in New Orleans")
        # A complete phrase stays.
        for tail, expected in (
            ("in the years 1914 to 1918 and also later in other cities.", "in the years 1914 to 1918"),
            ("in the 1880s and also later in other cities abroad.", "in the 1880s"),
        ):
            self.assertEqual(
                definition_clue(f"Laine was an American bandleader who worked in New Orleans {tail}"),
                f"An American bandleader who worked in New Orleans {expected}",
            )
        self.assertEqual(
            definition_clue("Laine was an American bandleader of the 19th century who also later played in other cities abroad."),
            "An American bandleader of the 19th century who also later played in other cities abroad",
        )

    SYNTHETISM = (
        "Synthetism is a term used by Post-Impressionist artists like Paul Gauguin, Émile Bernard and Louis Anquetin "
        "to distinguish their work stylistically from Impressionism."
    )

    def test_examples_in_the_middle_are_dropped_when_the_cut_would_stop_inside_them(self) -> None:
        nlp = _load_nlp()
        if nlp is None:
            self.skipTest("spaCy model not available")
        self.assertEqual(
            definition_clue(self.SYNTHETISM, nlp=nlp),
            "A term used by Post-Impressionist artists to distinguish their work stylistically from Impressionism",
        )
        # Without a parser the cut stops inside the list.
        self.assertEqual(definition_clue(self.SYNTHETISM), "A term used by Post-Impressionist artists like Paul Gauguin")

    def test_examples_the_cut_does_not_split_stay_and_an_open_list_is_no_phrase(self) -> None:
        nlp = _load_nlp()
        if nlp is None:
            self.skipTest("spaCy model not available")
        bound = (
            "A bound state is a composite of two or more fundamental building blocks, such as particles, atoms, or "
            "bodies, that behaves as a single object and in which energy is required to split them."
        )
        self.assertEqual(
            definition_clue(bound, nlp=nlp),
            "A composite of two or more fundamental building blocks, such as particles, atoms, or bodies",
        )
        ai = (
            "Artificial intelligence is the capability of computational systems to perform tasks typically associated "
            "with human intelligence, such as learning, reasoning, problem-solving, perception, and decision-making."
        )
        self.assertIsNone(definition_clue(ai, max_words=18, nlp=nlp))


class FollowUpDefinitionTests(unittest.TestCase):
    PIANO = (
        "He was one of the most influential figures in the development of jazz piano and, according to one "
        "source, one of a small number of pianists whose playing shaped the history of jazz."
    )

    def test_a_pronoun_or_the_name_before_a_copula_gives_the_definition(self) -> None:
        pianist = "An American pianist, keyboardist, bandleader and composer"
        for sentence, answer, expected in (
            (self.PIANO, "Earl Hines", "One of the most influential figures in the development of jazz piano"),
            ("Hancock was an American pianist, keyboardist, bandleader and composer.", "Herbie Hancock", pianist),
            ("Herbie Hancock was an American pianist, keyboardist, bandleader and composer.", "Herbie Hancock", pianist),
            ("They were an American rock band formed in Los Angeles in 1965.", "The Doors", "An American rock band formed in Los Angeles in 1965"),
            ("It is a genre of popular music that emerged in the 1970s.", "Hip-hop", "A genre of popular music that emerged in the 1970s"),
            ("The electron is an elementary particle contained in matter.", "Electron", "An elementary particle contained in matter"),
        ):
            with self.subTest(sentence=sentence):
                self.assertEqual(follow_up_definition(sentence, answer), expected)

    def test_it_cuts_like_definition_clue(self) -> None:
        # No cut between "figures" and the "in ..." it wants.
        self.assertIsNone(follow_up_definition(self.PIANO, "Earl Hines", max_words=8))
        self.assertEqual(
            follow_up_definition(self.PIANO, "Earl Hines", max_words=18),
            "One of the most influential figures in the development of jazz piano",
        )

    def test_a_capitalised_compound_does_not_name_the_answer_but_a_lower_case_prefix_does(self) -> None:
        self.assertIsNone(follow_up_definition("Neo-Impressionism was a movement of French painters in the 1880s.", "Impressionism"))
        self.assertIsNone(follow_up_definition("Post-bop was a style of jazz that developed in the 1960s.", "Bop"))
        khwarizmi = "Muhammad ibn Musa al-Khwarizmi was a Persian mathematician, astronomer, and geographer who worked in Baghdad."
        self.assertEqual(
            definition_clue(khwarizmi, answer="Khwarizmi"), "A Persian mathematician, astronomer, and geographer who worked in Baghdad"
        )

    def test_another_subject_or_a_verb_before_the_copula_is_refused(self) -> None:
        for sentence in (
            "He played with Miles Davis and was one of the most influential figures in jazz.",
            "Hancock's mother was a teacher of the piano at the local school.",
            "The album by Hancock was a landmark of the jazz fusion movement.",
            "Hancock played the piano in many bands.",
            "Miles Davis was an American jazz trumpeter, bandleader and composer.",
        ):
            with self.subTest(sentence=sentence):
                self.assertIsNone(follow_up_definition(sentence, "Herbie Hancock"))

    def test_a_predicate_that_says_what_the_answer_is_known_for_is_refused(self) -> None:
        nlp = _load_nlp()
        if nlp is None:
            self.skipTest("spaCy model not available")
        for sentence in (
            "He was best known for his work with Miles Davis and the Headhunters.",
            "She was known as the queen of soul and gospel music in the United States.",
            "He was born in Chicago, Illinois, to a family of professional musicians.",
        ):
            for model in (None, nlp):
                with self.subTest(sentence=sentence, tagger=model is not None):
                    self.assertIsNone(follow_up_definition(sentence, "Herbie Hancock", nlp=model))
        # An adjective with its complement says little more.
        popular = "It was popular in the 1940s and was a precursor of rhythm and blues."
        self.assertEqual(follow_up_definition(popular, "Jump blues"), "Popular in the 1940s")
        self.assertIsNone(follow_up_definition(popular, "Jump blues", nlp=nlp))

    def test_a_greek_bare_copula_or_the_name_opens_the_follow_up(self) -> None:
        nlp = _load_nlp("el_core_news_sm")
        for sentence, answer, expected in (
            ("Ήταν Σκωτσέζος μηχανικός και φυσικός.", "Ουίλιαμ Ράνκιν", "Σκωτσέζος μηχανικός και φυσικός"),
            ("Υπήρξε ένας από τους πρωτοπόρους της θερμοδυναμικής.", "Ουίλιαμ Ράνκιν", "Ένας από τους πρωτοπόρους της θερμοδυναμικής"),
            ("Ο Ράνκιν ήταν Σκωτσέζος μηχανικός και φυσικός.", "Ουίλιαμ Ράνκιν", "Σκωτσέζος μηχανικός και φυσικός"),
            ("Η φασματοσκοπία είναι κλάδος της φυσικής και ιδιαίτερα της οπτικής.", "Φασματοσκοπία", "Κλάδος της φυσικής και ιδιαίτερα της οπτικής"),
        ):
            for model in (None, nlp):
                with self.subTest(sentence=sentence, tagger=model is not None):
                    self.assertEqual(follow_up_definition(sentence, answer, nlp=model, lang="el"), expected)
        self.assertIsNone(follow_up_definition("Ο Ράνκιν μελέτησε τη θερμοδυναμική και ήταν καθηγητής.", "Ουίλιαμ Ράνκιν", lang="el"))
        # English does not drop its subject.
        self.assertIsNone(follow_up_definition("Was an American pianist and bandleader.", "Herbie Hancock"))


class ItDefinitionTests(unittest.TestCase):
    NLP = _load_nlp()
    TEMPERATURE = (
        "In classical thermodynamics and kinetic theory, temperature reflects the average kinetic energy of the "
        "particles in a system, providing a quantitative measure of how energy is distributed among microscopic "
        "degrees of freedom."
    )
    EQUITES = "The equites constituted the second of the property/social-based classes of ancient Rome, ranking below the senatorial class."

    def setUp(self) -> None:
        if self.NLP is None:
            self.skipTest("spaCy model not available")

    def _define(self, lead: str, answer: str, **kwargs) -> str | None:
        return definition_clue(lead, answer=answer, nlp=self.NLP, **kwargs)

    def test_a_main_verb_with_the_answer_as_its_subject_gives_it_or_they(self) -> None:
        for lead, answer, expected in (
            (self.TEMPERATURE, "Temperature", "It reflects the average kinetic energy of the particles in a system"),
            (
                "Temperature quantitatively expresses the attribute of hotness or coldness.",
                "Temperature",
                "It quantitatively expresses the attribute of hotness or coldness",
            ),
            (
                "In physics, coherence expresses the potential for two waves to interfere.",
                "Coherence",
                "It expresses the potential for two waves to interfere",
            ),
            (
                self.EQUITES,
                "Equites",
                "They constituted the second of the property/social-based classes of ancient Rome, ranking below the senatorial class",
            ),
        ):
            with self.subTest(answer=answer):
                self.assertEqual(self._define(lead, answer, max_words=18), expected)

    def test_a_long_predicate_is_cut_at_a_clean_boundary(self) -> None:
        self.assertEqual(
            self._define(self.EQUITES, "Equites", max_words=12),
            "They constituted the second of the property/social-based classes of ancient Rome",
        )
        self.assertIsNone(self._define(self.TEMPERATURE, "Temperature", max_words=8))

    def test_a_subject_that_is_more_than_the_answer_is_refused(self) -> None:
        for lead, answer in (
            (
                "Research concerning the relationship between the thermodynamic quantity entropy and both the origin "
                "and evolution of life began around the turn of the 20th century.",
                "Entropy and life",
            ),
            ("The mathematical concept of a Hilbert space generalizes the notion of Euclidean space.", "Hilbert space"),
            ("In music, the term swing has two main uses.", "Swing time"),
            ("Curia in ancient Rome referred to one of the original groupings of the citizenry.", "Curia"),
            ("The Clausius theorem, also known as the Clausius inequality, states that the inequality holds.", "Clausius theorem"),
        ):
            with self.subTest(answer=answer):
                self.assertIsNone(self._define(lead, answer, max_words=18))

    def test_a_sentence_with_a_copula_is_defined_through_it_as_before(self) -> None:
        bebop = "Bebop is a style of jazz that developed in the 1940s."
        self.assertEqual(self._define(bebop, "Bebop"), "A style of jazz that developed in the 1940s")
        self.assertEqual(definition_clue(bebop, answer="Bebop"), "A style of jazz that developed in the 1940s")
        # The copula path refuses this one, and "was" is no main verb for "It".
        best_known = "Hancock was best known for his work with Miles Davis and the Headhunters."
        self.assertIsNone(self._define(best_known, "Herbie Hancock"))

    def test_have_and_an_object_that_only_counts_are_refused(self) -> None:
        for lead, answer in (
            ("Sport in Italy has a long tradition.", "Sport in Italy"),
            ("Adaptation has three related meanings.", "Adaptation"),
            ("Christian humanism refers to two related concepts.", "Christian humanism"),
            ("Music of New Orleans assumes various styles of music which have often borrowed from earlier traditions.", "Music of New Orleans"),
        ):
            with self.subTest(answer=answer):
                self.assertIsNone(self._define(lead, answer, max_words=18))
        # A date range or a named object still says what the thing is.
        self.assertEqual(
            self._define("The Kitos War took place from 116 to 118.", "Kitos War"), "It took place from 116 to 118"
        )

    def test_it_needs_the_parser_the_answer_and_english(self) -> None:
        lead = "Temperature quantitatively expresses the attribute of hotness or coldness."
        self.assertIsNone(definition_clue(lead, answer="Temperature"))
        self.assertIsNone(definition_clue(lead, nlp=self.NLP))
        self.assertIsNone(definition_clue(lead, answer="Temperature", nlp=self.NLP, lang="el"))


class WhoDefinitionTests(unittest.TestCase):
    NLP = _load_nlp()
    JOULE = "Joule studied the nature of heat, and discovered its relationship to mechanical work."

    def setUp(self) -> None:
        if self.NLP is None:
            self.skipTest("spaCy model not available")

    def _who(self, head: str, sentence: str, answer: str, **kwargs) -> str | None:
        return who_definition(head, sentence, answer, nlp=self.NLP, **kwargs)

    def test_the_head_and_a_sentence_about_the_person_make_one_definition(self) -> None:
        head = "An English physicist, mathematician and brewer"
        self.assertEqual(
            self._who(head, self.JOULE, "James Joule"),
            "An English physicist who studied the nature of heat and discovered its relationship to mechanical work",
        )
        # An adverb between the subject and the verb stays: "never" would change the meaning.
        self.assertEqual(
            self._who("An American jazz pianist and composer", "He never recorded for a major label.", "Bud Powell"),
            "An American jazz pianist and composer who never recorded for a major label",
        )

    def test_the_head_is_cut_back_to_its_first_item_only_when_the_predicate_needs_the_room(self) -> None:
        head = "An American jazz musician, bandleader, and composer"
        sentence = "He started his career with trumpeter Donald Byrd's group."
        self.assertEqual(
            self._who(head, sentence, "Herbie Hancock"),
            "An American jazz musician, bandleader, and composer who started his career with trumpeter Donald Byrd's group",
        )
        self.assertEqual(
            self._who(head, sentence, "Herbie Hancock", max_words=14),
            "An American jazz musician who started his career with trumpeter Donald Byrd's group",
        )

    def test_a_long_predicate_is_cut_where_an_adjunct_ends_never_inside_a_complement(self) -> None:
        feynman = (
            "He shared the 1965 Nobel Prize in Physics with Julian Schwinger and Shin'ichirō Tomonaga "
            '"for their fundamental work in quantum electrodynamics".'
        )
        self.assertEqual(
            self._who("An American theoretical physicist", feynman, "Richard Feynman"),
            "An American theoretical physicist who shared the 1965 Nobel Prize in Physics",
        )
        davis = "He is among the most influential and acclaimed figures in the history of jazz and 20th-century music."
        self.assertIsNone(self._who("An American trumpeter, bandleader and composer", davis, "Miles Davis"))
        titles = (
            "He is noted for Chemin en été, La dispersion des Acadiens, L'arrivée de Champlain à Québec, "
            "and Les Noces de Cana."
        )
        self.assertIsNone(self._who("A French-Canadian Impressionist painter", titles, "Henri Beau", max_words=12))

    def test_another_subject_is_refused(self) -> None:
        head = "An American jazz pianist and composer"
        for sentence in (
            "Hancock's mother was a teacher of the piano at the local school.",
            "The album was released in 1962.",
            "It was released in 1962.",
        ):
            with self.subTest(sentence=sentence):
                self.assertIsNone(self._who(head, sentence, "Herbie Hancock"))

    def test_a_clause_with_a_subject_of_its_own_is_left_out(self) -> None:
        sentence = "He joined the band in 1940, and his influence crossed genres."
        self.assertEqual(
            self._who("An American jazz pianist and composer", sentence, "Bud Powell"),
            "An American jazz pianist and composer who joined the band in 1940",
        )

    def test_a_head_that_is_no_plain_occupation_or_a_vague_object_is_refused(self) -> None:
        planck = "Planck made many contributions to theoretical physics, primarily as the originator of quantum theory."
        for head, sentence in (
            ("A British historian specializing in ancient Rome", "He studied at University College London."),
            ("A German theoretical physicist", planck),
        ):
            with self.subTest(head=head):
                self.assertIsNone(self._who(head, sentence, "Max Planck"))

    def test_a_head_that_is_no_plain_list_gives_its_first_item(self) -> None:
        morisot = "In 1864, Morisot exhibited for the first time in the highly esteemed Paris Salon, listed as a student of Corot."
        self.assertEqual(
            self._who("A French painter, printmaker and a member of the circle of painters in Paris", morisot, "Berthe Morisot"),
            "A French painter who exhibited for the first time in the highly esteemed Paris Salon",
        )
        self.assertEqual(
            self._who("A French engineer and physicist, one of the founders of thermodynamics", "He studied heat engines.", "Émile Clapeyron"),
            "A French engineer and physicist who studied heat engines",
        )
        # No comma, no first item.
        self.assertIsNone(self._who("A British historian specializing in ancient Rome", "He studied at University College London.", "Tim Cornell"))

    def test_one_opening_phrase_is_left_out_unless_the_rest_leans_on_it(self) -> None:
        getz = (
            'Playing primarily the tenor saxophone, Getz was known as "The Sound" because of his warm, lyrical tone, '
            "with his prime influence being the wispy, mellow timbre of his idol, Lester Young."
        )
        self.assertEqual(
            self._who("An American jazz saxophonist", getz, "Stan Getz"),
            'An American jazz saxophonist who was known as "The Sound" because of his warm, lyrical tone',
        )
        self.assertEqual(
            self._who("A German-American physicist", "For most of his career, Bethe was a professor at Cornell University.", "Hans Bethe"),
            "A German-American physicist who was a professor at Cornell University",
        )
        head = "An American jazz pianist"
        for sentence in (
            "Since the early 1970s, he has also been a group leader and solo performer in jazz.",  # "also"
            "After the war, Powell worked with other bandleaders in New York.",  # "other"
            "In 1950, Powell recorded it with a trio.",  # "it"
            "Later, Powell joined the band of Cootie Williams.",  # the opening points back
            "When he left the band, Powell joined Cootie Williams.",  # a clause of its own
            "A pioneer of bebop, Powell's application of complex phrasing influenced his contemporaries.",  # another subject
        ):
            with self.subTest(sentence=sentence):
                self.assertIsNone(self._who(head, sentence, "Bud Powell"))

    def test_a_sequencing_adverb_or_a_birthplace_is_refused(self) -> None:
        head = "An American jazz musician, bandleader, and composer"
        for sentence in (
            "He then moved to the U.S.",
            "She soon moved to New York City.",
            "He moved afterwards to Paris.",
            "He subsequently left the band.",
            "She was born in Allegheny, Pennsylvania, but lived most of her adult life in France.",
            "Hancock was later born in Chicago.",
        ):
            with self.subTest(sentence=sentence):
                self.assertIsNone(self._who(head, sentence, "Herbie Hancock"))
        # "later" and "also" read on their own.
        self.assertEqual(
            self._who(head, "He later joined Miles Davis's quintet.", "Herbie Hancock"),
            f"{head} who later joined Miles Davis's quintet",
        )
        self.assertEqual(
            self._who(head, "He also played the piano.", "Herbie Hancock"), f"{head} who also played the piano"
        )

    def test_it_needs_the_parser_and_english(self) -> None:
        self.assertIsNone(who_definition("An English physicist", self.JOULE, "James Joule"))
        self.assertIsNone(self._who("An English physicist", self.JOULE, "James Joule", lang="el"))


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

    def test_a_cut_never_comes_after_the_first_item_of_a_list(self) -> None:
        photonics = (
            "Φωτονική ονομάζεται ο κλάδος των επιστημών και της τεχνολογίας που ασχολείται με την δημιουργία, τον "
            "έλεγχο και την ανίχνευση φωτονίων ιδιαίτερα στην περιοχή του ορατού φωτός."
        )
        micro = (
            "Μικροηλεκτρονική καλείται ο κλάδος της ηλεκτρονικής τεχνολογίας που εφαρμόζει τις θεωρίες, τις "
            "τεχνικές και τις μεθόδους της ηλεκτρονικής στην κατασκευή μικροσκοπικών συσκευών."
        )
        for nlp in (None, _load_nlp("el_core_news_sm")):
            self.assertEqual(
                definition_clue(photonics, answer="Φωτονική", lang="el", nlp=nlp), "Ο κλάδος των επιστημών και της τεχνολογίας"
            )
            self.assertIsNone(definition_clue(micro, answer="Μικροηλεκτρονική", lang="el", nlp=nlp))
        # A comma before "η οποία" or a clause opener is not a list.
        self.assertEqual(self._definition("Εντροπία"), "Η έννοια μέσω της οποίας μετράται η αταξία")

    def test_the_tagger_finds_a_second_predicate_whatever_its_verb(self) -> None:
        nlp = _load_nlp("el_core_news_sm")
        if nlp is None:
            self.skipTest("Greek spaCy model not available")
        seasons = "Οι εποχές είναι μία από τις μείζονες διαιρέσεις του έτους και βασίζεται γενικά σε ομοιογενή κλιματικά χαρακτηριστικά."
        self.assertEqual(definition_clue(seasons, answer="Εποχές", lang="el", nlp=nlp), "Μία από τις μείζονες διαιρέσεις του έτους")
        # The last verb of a list of verbs, and the "και" of "μιας και" (since), do not start a second predicate.
        geography = (
            "Η Γεωγραφία είναι η επιστήμη η οποία περιγράφει, αναλύει και προβλέπει την εξέλιξη της Γης στις βασικές "
            "πτυχές της, τόσο του ανθρωπογενούς, όσο και του φυσικού χώρου."
        )
        self.assertNotEqual(definition_clue(geography, answer="Γεωγραφία", lang="el", nlp=nlp), "Η επιστήμη η οποία περιγράφει, αναλύει")
        astrology = (
            "Η αστρολογία είναι μια πρακτική με μακραίωνη παράδοση και σήμερα αναγνωρίζεται ως ψευδοεπιστήμη, παραεπιστήμη ή "
            "απλή προκατάληψη μιας και απέτυχε επανειλημμένα να αποδείξει την αξία της σε αριθμό ελεγχόμενων μελετών."
        )
        self.assertEqual(
            definition_clue(astrology, answer="Αστρολογία", lang="el", nlp=nlp),
            "Μια πρακτική με μακραίωνη παράδοση και σήμερα αναγνωρίζεται ως ψευδοεπιστήμη",
        )

    def test_a_cut_inside_a_relative_clause_that_opens_on_an_aside_is_cut_back(self) -> None:
        nlp = _load_nlp("el_core_news_sm")
        if nlp is None:
            self.skipTest("Greek spaCy model not available")
        rest = ", εκφράζει την ποσότητα της ενέργειας που παράγεται ή καταναλώνεται από ένα σώμα."
        work = "Το έργο είναι όρος της Φυσικής, που σε πολύ απλή απόδοση" + rest
        # "που σε πολύ απλή απόδοση," never reaches its verb, and "Όρος της Φυσικής" alone defines nothing.
        self.assertIsNone(definition_clue(work, answer="Έργο", lang="el", nlp=nlp))
        longer = "Το έργο είναι ένα από τα βασικά μεγέθη της Φυσικής, που σε πολύ απλή απόδοση" + rest
        self.assertEqual(
            definition_clue(longer, answer="Έργο", lang="el", nlp=nlp), "Ένα από τα βασικά μεγέθη της Φυσικής"
        )
        # Without a tagger nothing tells an aside from a clause.
        self.assertEqual(definition_clue(work, answer="Έργο", lang="el"), "Όρος της Φυσικής, που σε πολύ απλή απόδοση")
        # A clause that reaches its verb stays, also when the tagger does not see the verb ("μελετά").
        reaches = (
            "Το έργο είναι όρος της Φυσικής, που σε πολύ απλή απόδοση εκφράζει την ποσότητα της ενέργειας, η οποία "
            "παράγεται ή καταναλώνεται από ένα σώμα κατά τη διάρκεια μιας μεταβολής."
        )
        self.assertEqual(
            definition_clue(reaches, answer="Έργο", lang="el", nlp=nlp),
            "Όρος της Φυσικής, που σε πολύ απλή απόδοση εκφράζει την ποσότητα της ενέργειας",
        )
        biology = (
            "Η βιολογία είναι η επιστήμη της φύσης που μελετά τη ζωή και τους ζωντανούς οργανισμούς, δηλαδή τις φυσικές "
            "δομές, χημικές διεργασίες, μοριακές αλληλεπιδράσεις, φυσιολογικούς μηχανισμούς, την ανάπτυξη και την εξέλιξη."
        )
        self.assertEqual(
            definition_clue(biology, answer="Βιολογία", lang="el", nlp=nlp),
            "Η επιστήμη της φύσης που μελετά τη ζωή και τους ζωντανούς οργανισμούς",
        )

    def test_a_definition_that_promises_a_list_is_cut_before_the_promise(self) -> None:
        planet = (
            "Πλανήτης, σύμφωνα με τον σύγχρονο ορισμό της Διεθνούς Αστρονομικής Ένωσης, ονομάζεται κάθε ουράνιο σώμα "
            "που περιφέρεται γύρω από έναν αστέρα ή αστρικό υπόλειμμα και καλύπτει τις ακόλουθες πρόσθετες προϋποθέσεις:"
        )
        nlp = _load_nlp("el_core_news_sm")
        if nlp is not None:
            self.assertEqual(
                definition_clue(planet, answer="Πλανήτης", lang="el", nlp=nlp),
                "Κάθε ουράνιο σώμα που περιφέρεται γύρω από έναν αστέρα ή αστρικό υπόλειμμα",
            )
        self.assertIsNone(definition_clue(planet, answer="Πλανήτης", lang="el"))
        self.assertIsNone(definition_clue("Heat is the energy of the following kinds of motion in bodies.", answer="Heat"))
        self.assertIsNone(definition_clue("A bra is likewise a vector of the dual space.", answer="Bra"))

    def test_the_verb_of_a_relative_clause_or_an_aside_is_not_the_copula(self) -> None:
        expansion = (
            "Στη Φυσική γενικά με τον όρο εκτόνωση εννοείται η αποδέσμευση των τάσεων που είναι συσσωρευμένες σε ένα "
            "σύστημα που μπορεί να αποτελείται από στερεά, υγρά, αέρια ή και άλλους συνδυασμούς."
        )
        self.assertIsNone(definition_clue(expansion, answer="Εκτόνωση", lang="el"))
        # "όπως είναι ευρέως γνωστός" is an aside; a relative clause with a verb of its own before the copula is not one.
        feyerabend = (
            "Ο Πωλ Φεγεράμπεντ ή, όπως είναι ευρέως γνωστός, Πολ Φεγεράμπεντ ήταν Αυστριακός φιλόσοφος της επιστήμης γνωστός "
            "για την εργασία του ως καθηγητής φιλοσοφίας στο Πανεπιστήμιο Μπέρκλεϊ της Καλιφόρνιας, όπου και εργάστηκε."
        )
        self.assertEqual(
            definition_clue(feyerabend, answer="Φεγεράμπεντ", lang="el"),
            "Αυστριακός φιλόσοφος της επιστήμης γνωστός για την εργασία του ως καθηγητής φιλοσοφίας",
        )
        statue = (
            "Η Κόρη της Ακρόπολης που εκτίθεται στο εκεί μουσείο με αριθμό ευρετηρίου 680 είναι γλυπτικό έργο, σημαντικό "
            "δείγμα κοροπλαστικής με απελευθερωμένη κίνηση."
        )
        self.assertEqual(
            definition_clue(statue, answer="Κόρη Ακρόπολης", lang="el"),
            "Γλυπτικό έργο, σημαντικό δείγμα κοροπλαστικής με απελευθερωμένη κίνηση",
        )

    def test_a_hyphenated_prefix_belongs_to_the_subject_and_a_noun_keeps_its_preposition(self) -> None:
        khwarizmi = "Ο Αμπού Αμπντουλάχ Μοχάμεντ ιμπν Μουσά αλ-Χουαρίζμι ήταν Πέρσης μαθηματικός, αστρονόμος και γεωγράφος, ένας λόγιος στον Οίκο της Σοφίας στη Βαγδάτη."
        self.assertEqual(
            definition_clue(khwarizmi, answer="Χουαρίζμι", lang="el"),
            "Πέρσης μαθηματικός, αστρονόμος και γεωγράφος, ένας λόγιος στον Οίκο της Σοφίας στη Βαγδάτη",
        )
        mill = (
            "Ο Τζον Στιούαρτ Μιλ ήταν Βρετανός φιλόσοφος, πολιτικός οικονομολόγος και δημόσιος υπάλληλος με ισχυρή "
            "συνεισφορά στην κοινωνική και πολιτική θεωρία καθώς και την οικονομία."
        )
        self.assertEqual(
            definition_clue(mill, answer="Μιλ", lang="el"), "Βρετανός φιλόσοφος, πολιτικός οικονομολόγος και δημόσιος υπάλληλος"
        )

    def test_the_head_of_a_person_may_be_two_words(self) -> None:
        messier = (
            "Ο Σαρλ Μεσιέ ήταν Γάλλος αστρονόμος, γνωστός κυρίως από τη σύνταξη του πρώτου ομώνυμου αστρονομικού καταλόγου "
            "ουράνιων σωμάτων του «βαθέος ουρανού», όπως νεφελώματα, γαλαξίες και αστρικά σμήνη."
        )
        self.assertIsNone(definition_clue(messier, answer="Μεσιέ", lang="el"))
        self.assertEqual(definition_clue(messier, answer="Μεσιέ", lang="el", person=True), "Γάλλος αστρονόμος")
        verdet = "Ο Μαρσέλ Εμίλ Βερντέ ήταν Γάλλος φυσικός."
        self.assertIsNone(definition_clue(verdet, answer="Βερντέ", lang="el"))
        self.assertEqual(definition_clue(verdet, answer="Βερντέ", lang="el", person=True), "Γάλλος φυσικός")
        # Before a relative word, with no comma: "που ..." starts the clause that is cut off.
        bjerknes = (
            "Ο Βίλχελμ Μπγιέρκνες ήταν Νορβηγός φυσικός και μετεωρολόγος που διακρίθηκε για τη συμβολή του στη σύγχρονη "
            "πρόγνωση του καιρού και στην ανάπτυξη της μετεωρολογικής θεωρίας."
        )
        self.assertEqual(definition_clue(bjerknes, answer="Μπγιέρκνες", lang="el", person=True), "Νορβηγός φυσικός και μετεωρολόγος")

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
            "Σημειώσεις και παραπομπές", "Περαιτέρω ανάγνωση", "Εσωτερικοί σύνδεσμοι",
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
