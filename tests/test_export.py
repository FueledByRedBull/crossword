import json
import unittest

from src.export import export_html, export_ipuz

PUZZLE = {
    "seed_title": "Cats & <Dogs>",
    "lang": "en",
    "grid_cells": [list("####"), list("#CAT"), list("#A##"), list("#R##")],
    "across_entries": [
        {"number": 1, "answer": "CAT", "clue": "Pet with <b>claws</b> & fur", "enumeration": "3", "row": 1, "col": 1, "length": 3},
    ],
    "down_entries": [
        {"number": 1, "answer": "CAR", "clue": "Vehicle", "enumeration": "3", "row": 1, "col": 1, "length": 3},
    ],
    "attribution": [
        {"answer": "Cat", "source_page": "Cat (animal)", "oldid_url": "https://en.wikipedia.org/w/index.php?title=Cat&oldid=1"},
        {"answer": "Car", "source_page": "Car & Co", "oldid_url": "https://en.wikipedia.org/w/index.php?title=Car&oldid=2"},
    ],
}


class ExportTests(unittest.TestCase):
    def test_ipuz_is_consistent_with_puzzle(self) -> None:
        data = json.loads(export_ipuz(PUZZLE))
        self.assertEqual(data["version"], "http://ipuz.org/v2")
        self.assertEqual(data["dimensions"], {"width": 3, "height": 3})  # cropped to letters
        self.assertEqual(data["puzzle"], [[1, 0, 0], [0, "#", "#"], [0, "#", "#"]])
        self.assertEqual(data["solution"], [list("CAT"), ["A", "#", "#"], ["R", "#", "#"]])
        self.assertEqual(len(data["clues"]["Across"]), 1)
        self.assertEqual(len(data["clues"]["Down"]), 1)
        across = data["clues"]["Across"][0]
        self.assertEqual(across, {"number": 1, "clue": "Pet with <b>claws</b> & fur", "enumeration": "3"})
        self.assertIn("Cat (animal)", data["notes"])
        self.assertIn("CC BY-SA", data["copyright"])
        # every number sits on its entry's start cell and the solution spells the answer
        for key, step in (("Across", (0, 1)), ("Down", (1, 0))):
            entry = PUZZLE[key.lower() + "_entries"][0]
            r, c = entry["row"] - 1, entry["col"] - 1
            self.assertEqual(data["puzzle"][r][c], entry["number"])
            word = "".join(data["solution"][r + step[0] * i][c + step[1] * i] for i in range(entry["length"]))
            self.assertEqual(word, entry["answer"])

    def test_html_escapes_and_lists_sources(self) -> None:
        page = export_html(PUZZLE)
        self.assertIn("Pet with &lt;b&gt;claws&lt;/b&gt; &amp; fur (3)", page)
        self.assertNotIn("<b>claws</b>", page)
        self.assertIn("<title>Cats &amp; &lt;Dogs&gt;</title>", page)
        self.assertIn("<sup>1</sup>", page)
        self.assertIn('<a href="https://en.wikipedia.org/w/index.php?title=Cat&amp;oldid=1">Cat (animal)</a>', page)
        self.assertIn("Car &amp; Co", page)
        self.assertNotIn("<script", page)

    def test_html_solution_only_when_asked(self) -> None:
        self.assertNotIn("Solution", export_html(PUZZLE))
        solved = export_html(PUZZLE, solution=True)
        self.assertIn('class="solution"', solved)
        self.assertIn("</sup>C</td>", solved)
        self.assertNotIn("</sup>C</td>", export_html(PUZZLE))


if __name__ == "__main__":
    unittest.main()
