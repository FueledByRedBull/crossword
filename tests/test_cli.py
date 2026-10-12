import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace

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


if __name__ == "__main__":
    unittest.main()
