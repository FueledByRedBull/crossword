import shutil
import unittest
from pathlib import Path

from src.lexicon import (
    lexicon_score_for_token,
    lexicon_score_for_tokens,
    load_lexicon_scores,
)


class LexiconTests(unittest.TestCase):
    def test_load_lexicon_scores_normalizes_numeric_values(self) -> None:
        tmp_dir = Path("tests") / "tmp_outputs" / "lexicon"
        tmp_file = tmp_dir / "sample.txt"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_file.write_text("alpha 10\nbeta 20\ngamma 15\n", encoding="utf-8")

        scores = load_lexicon_scores(tmp_file)
        self.assertIn("ALPHA", scores)
        self.assertIn("BETA", scores)
        self.assertLess(scores["ALPHA"], scores["GAMMA"])
        self.assertLess(scores["GAMMA"], scores["BETA"])

        shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_lexicon_score_for_tokens_uses_mean_of_hits(self) -> None:
        lexicon = {"ALPHA": 1.0, "BETA": 0.5}
        score = lexicon_score_for_tokens(["alpha", "beta", "missing"], lexicon)
        self.assertAlmostEqual(score, 0.75, places=4)
        self.assertEqual(lexicon_score_for_token("missing", lexicon), 0.0)


if __name__ == "__main__":
    unittest.main()
