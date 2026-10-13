import base64
import json
import re
import unittest

from src.export import export_html, export_ipuz, export_solvable_html
from src.text_normalize import strip_diacritics

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


# Greek labels and a grid of capitals without accents: ΑΤΟΜΟ across, ΑΣΤΡΟ down.
GREEK_PUZZLE = {
    "seed_title": "Φυσική",
    "lang": "el",
    "grid_cells": [list("ΑΤΟΜΟ"), list("Σ####"), list("Τ####"), list("Ρ####"), list("Ο####")],
    "across_entries": [
        {"number": 1, "answer": "ΑΤΟΜΟ", "clue": "Το μικρότερο τμήμα ενός στοιχείου", "enumeration": "5", "row": 0, "col": 0, "length": 5},
    ],
    "down_entries": [
        {"number": 1, "answer": "ΑΣΤΡΟ", "clue": "Ουράνιο σώμα που λάμπει", "enumeration": "5", "row": 0, "col": 0, "length": 5},
    ],
    "attribution": [
        {"answer": "Άτομο", "source_page": "Άτομο", "oldid_url": "https://el.wikipedia.org/w/index.php?title=Άτομο&oldid=1"},
    ],
}

LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"


def _embedded_data(page: str) -> dict:
    return json.loads(re.search(r'<script type="application/json" id="data">(.*?)</script>', page, re.S).group(1))


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
        # ipuz display strings are HTML: Wikipedia text is escaped so a reader shows it as written.
        self.assertEqual(across, {"number": 1, "clue": "Pet with &lt;b&gt;claws&lt;/b&gt; &amp; fur", "enumeration": "3"})
        self.assertEqual(data["title"], "Cats &amp; &lt;Dogs&gt;")
        self.assertIn("Cat (animal) (https://en.wikipedia.org/w/index.php?title=Cat&amp;oldid=1)", data["notes"])
        self.assertIn("Car &amp; Co", data["notes"])
        self.assertNotIn("<", data["notes"])
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


class SolvableExportTests(unittest.TestCase):
    def test_page_is_one_document_without_external_references(self) -> None:
        page = export_solvable_html(PUZZLE)
        self.assertEqual(page.count("<html"), 1)
        self.assertTrue(page.endswith("</html>\n"))
        self.assertNotIn("\r", page)
        links = set(re.findall(r"https?://[^\s\"'<>]+", page))
        self.assertEqual(
            links,
            {
                LICENSE_URL,
                "https://en.wikipedia.org/w/index.php?title=Cat&amp;oldid=1",
                "https://en.wikipedia.org/w/index.php?title=Car&amp;oldid=2",
            },
        )
        for loaded in ("<link", " src=", "url(", "@import", "<img", "<iframe"):
            self.assertNotIn(loaded, page)

    def test_title_and_clues_are_escaped_and_listed_as_buttons(self) -> None:
        page = export_solvable_html(PUZZLE)
        self.assertIn("<title>Cats &amp; &lt;Dogs&gt;</title>", page)
        self.assertIn('aria-label="Cats &amp; &lt;Dogs&gt;"', page)
        self.assertIn("Pet with &lt;b&gt;claws&lt;/b&gt; &amp; fur", page)
        self.assertNotIn("<b>claws</b>", page)
        self.assertIn('<button type="button" class="clue" data-word="0"><b class="clue-number">1</b>', page)
        self.assertIn('<button type="button" class="clue" data-word="1"><b class="clue-number">1</b>', page)
        self.assertIn('<span class="enumeration">(3)</span>', page)

    def test_hostile_text_cannot_break_out_of_the_embedded_data(self) -> None:
        hostile = "</script><script>alert(1)</script>" + chr(0x2028) + chr(0x2029) + "<!--"
        puzzle = {**PUZZLE, "seed_title": hostile}
        puzzle["across_entries"] = [{**PUZZLE["across_entries"][0], "clue": hostile}]
        page = export_solvable_html(puzzle)
        # the page's own two script blocks and no more
        self.assertEqual(page.count("<script"), 2)
        self.assertEqual(page.count("</script"), 2)
        data_block = re.search(r'<script type="application/json" id="data">(.*?)</script>', page, re.S).group(1)
        for character in ("<", chr(0x2028), chr(0x2029)):
            self.assertNotIn(character, data_block)
        self.assertTrue(json.loads(data_block)["storageKey"].startswith("crossword:" + hostile + ":"))
        self.assertIn("&lt;/script&gt;&lt;script&gt;alert(1)&lt;/script&gt;", page)

    def test_solution_is_not_readable_outside_the_sources(self) -> None:
        page = export_solvable_html(GREEK_PUZZLE)
        outside_sources = re.sub(r"<details.*?</details>", "", page, flags=re.S)
        for answer in ("ΑΤΟΜΟ", "ΑΣΤΡΟ"):
            self.assertNotIn(answer, outside_sources)
            self.assertNotIn(strip_diacritics(answer).upper(), strip_diacritics(outside_sources).upper())
        self.assertIn("Άτομο", page)  # the source title is in the page, in the sources block

    def test_embedded_data_decodes_to_the_cropped_grid_and_entries(self) -> None:
        data = _embedded_data(export_solvable_html(PUZZLE))
        self.assertEqual(json.loads(base64.b64decode(data["solution"])), ["CAT", "A..", "R.."])
        self.assertEqual((data["columns"], data["rows"]), (3, 3))
        self.assertEqual(
            data["words"],
            [
                {"direction": 0, "number": 1, "cells": [0, 1, 2]},
                {"direction": 1, "number": 1, "cells": [0, 3, 6]},
            ],
        )

    def test_cells_are_inputs_with_accessible_names_and_blanks_are_empty(self) -> None:
        page = export_solvable_html(PUZZLE)
        self.assertEqual(page.count("<input "), 5)
        self.assertEqual(page.count("<div></div>"), 4)
        self.assertIn('aria-label="Row 1, column 1, 1 across (letter 1 of 3), 1 down (letter 1 of 3)"', page)
        self.assertIn('aria-label="Row 1, column 3, 1 across (letter 3 of 3)"', page)
        self.assertIn('aria-label="Row 3, column 1, 1 down (letter 3 of 3)"', page)
        self.assertEqual(page.count('<i class="number" aria-hidden="true">1</i>'), 1)

    def test_progress_key_does_not_change(self) -> None:
        # Progress saved by a page is found again under this key, so it must not change with the page code.
        data = _embedded_data(export_solvable_html(PUZZLE))
        self.assertEqual(data["storageKey"], "crossword:Cats & <Dogs>:421d98a03799")

    def test_sources_are_in_a_closed_block_with_the_licence_line(self) -> None:
        page = export_solvable_html(PUZZLE)
        block = re.search(r"<details[^>]*>(.*?)</details>", page, re.S).group(1)
        self.assertNotIn(" open", re.search(r"<details[^>]*>", page).group(0))
        self.assertIn("give away answers", block)
        self.assertIn('<a href="https://en.wikipedia.org/w/index.php?title=Cat&amp;oldid=1">Cat (animal)</a>', block)
        self.assertIn("Car &amp; Co", block)
        self.assertIn(f'Clues are adapted from Wikipedia articles, licensed <a href="{LICENSE_URL}">CC BY-SA 4.0</a>.', page)
        self.assertNotIn("CC BY-SA", block)

    def test_labels_follow_the_puzzle_language(self) -> None:
        english = export_solvable_html(PUZZLE)
        self.assertIn('<html lang="en">', english)
        self.assertIn("<h2>Across</h2>", english)
        self.assertIn("<h2>Down</h2>", english)
        self.assertIn(">Check</b>", english)
        greek = export_solvable_html(GREEK_PUZZLE)
        self.assertIn('<html lang="el">', greek)
        self.assertIn("<h2>Οριζόντια</h2>", greek)
        self.assertIn("<h2>Κάθετα</h2>", greek)
        self.assertIn(">Έλεγχος</b>", greek)
        self.assertIn("Βικιπαίδειας", greek)
        self.assertIn("Γραμμή 1, στήλη 1, 1 οριζόντια (γράμμα 1 από 5)", greek)
        self.assertIn("Ολοκληρώθηκε σωστά", _embedded_data(greek)["labels"]["done"])
        # a language without labels falls back to English
        self.assertIn("<h2>Across</h2>", export_solvable_html({**PUZZLE, "lang": "fr"}))

    def test_an_entry_that_does_not_match_the_grid_is_an_error(self) -> None:
        puzzle = {**PUZZLE, "across_entries": [{**PUZZLE["across_entries"][0], "answer": "COT"}]}
        with self.assertRaises(ValueError):
            export_solvable_html(puzzle)


if __name__ == "__main__":
    unittest.main()
