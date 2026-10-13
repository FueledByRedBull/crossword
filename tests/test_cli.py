import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cli


def _report(fill_status: str, puzzle_status: str, **recorded: list[str]) -> tuple[str, bool]:
    with tempfile.TemporaryDirectory() as tmp, redirect_stderr(io.StringIO()) as stderr:
        for stage, errors in recorded.items():
            (Path(tmp) / f"diagnostics_{stage}.json").write_text(json.dumps({"errors": errors}), encoding="utf-8")
        package = SimpleNamespace(diagnostics={"fill_status": fill_status, "puzzle_status": puzzle_status})
        usable = cli._report_problems({"output_dir": tmp, "package": package}, started=0.0)
    return stderr.getvalue(), usable


class GenerateSummaryTests(unittest.TestCase):
    def test_failed_run_names_its_statuses_files_and_first_messages(self) -> None:
        errors = [f"link_fetch_failed: offline_cache_miss:{n}" for n in range(6)]
        errors[0] = "x" * 300
        text, usable = _report("failed", "insufficient_vocabulary", scores=errors, k=errors[:2], package=[])

        self.assertFalse(usable)
        self.assertIn("fill_status=failed, puzzle_status=insufficient_vocabulary, 6 distinct errors", text)
        self.assertIn("Recorded in: diagnostics_k.json, diagnostics_scores.json", text)
        self.assertIn("  " + "x" * 160 + "\n", text)
        self.assertIn("offline_cache_miss:4", text)
        self.assertNotIn("offline_cache_miss:5", text)

    def test_clean_run_prints_nothing(self) -> None:
        self.assertEqual(_report("complete", "ok", scores=[], package=[]), ("", True))

    def test_recorded_errors_alone_do_not_fail_a_complete_puzzle(self) -> None:
        text, usable = _report("complete", "ok", scores=["link_fetch_failed: offline_cache_miss:Page"])

        self.assertIn("fill_status=complete, puzzle_status=ok, 1 distinct errors", text)
        self.assertTrue(usable)


TINY_PUZZLE = {
    "seed_title": "Tiny",
    "lang": "en",
    "grid_cells": [list("CAT")],
    "across_entries": [
        {"number": 1, "answer": "CAT", "clue": "Pet", "enumeration": "3", "row": 0, "col": 0, "length": 3},
    ],
    "down_entries": [],
    "attribution": [],
}


def _export(*flags: str) -> tuple[int, str, str | None]:
    """Run `cli.py export` on a tiny puzzle: the exit code, what it printed to stderr and the file it wrote."""
    with tempfile.TemporaryDirectory() as tmp:
        puzzle, output = Path(tmp) / "puzzle.json", Path(tmp) / "out.html"
        puzzle.write_text(json.dumps(TINY_PUZZLE), encoding="utf-8")
        argv = ["cli.py", "export", "--puzzle", str(puzzle), "--output", str(output), *flags]
        with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as stderr:
            try:
                cli.main()
                code = 0
            except SystemExit as stop:
                code = stop.code
        return code, stderr.getvalue(), output.read_text(encoding="utf-8") if output.exists() else None


class ExportCommandTests(unittest.TestCase):
    def test_solvable_writes_the_page_to_solve(self) -> None:
        code, _, page = _export("--format", "html", "--solvable")

        self.assertEqual(code, 0)
        self.assertIn('id="data"', page)
        self.assertNotIn('id="data"', _export("--format", "html")[2])  # the printable page stays the default

    def test_solvable_with_solution_is_an_error_and_writes_nothing(self) -> None:
        code, stderr, page = _export("--format", "html", "--solvable", "--solution")

        self.assertEqual(code, 2)
        self.assertIn("--solution belongs to the printable page", stderr)
        self.assertIsNone(page)

    def test_solvable_with_ipuz_is_an_error_and_writes_nothing(self) -> None:
        code, stderr, page = _export("--format", "ipuz", "--solvable")

        self.assertEqual(code, 2)
        self.assertIn("--solvable needs --format html", stderr)
        self.assertIsNone(page)


if __name__ == "__main__":
    unittest.main()
