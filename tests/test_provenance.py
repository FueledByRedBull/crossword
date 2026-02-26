import unittest

from src.provenance import build_oldid_url, validate_clue_provenance


class ProvenanceTests(unittest.TestCase):
    def test_build_oldid_url(self) -> None:
        url = build_oldid_url("Thermodynamics", 12345, lang="en")
        self.assertIn("oldid=12345", url)
        self.assertIn("Thermodynamics", url)

    def test_validate_clue_provenance(self) -> None:
        clues = [
            {
                "answer": "ENTROPY",
                "clue": "A measure of disorder ____",
                "source_method": "spacy",
                "source_page": "Entropy",
                "revid": 1,
                "sentence_offset": 0,
                "oldid_url": "https://en.wikipedia.org/w/index.php?title=Entropy&oldid=1",
            }
        ]
        self.assertEqual(validate_clue_provenance(clues), [])
        clues[0]["revid"] = None
        missing = validate_clue_provenance(clues)
        self.assertTrue(missing)


if __name__ == "__main__":
    unittest.main()

