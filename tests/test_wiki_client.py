import shutil
import unittest
from pathlib import Path

from src.cache import DiskCache
from src.wiki_client import WikiClient


class WikiClientTests(unittest.TestCase):
    def test_extract_lead_bold_terms_from_wikitext(self) -> None:
        cache_dir = Path("tests") / "tmp_cache_wiki_client"
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            cache = DiskCache(cache_dir)
            client = WikiClient(cache)
            wikitext = "'''Thermodynamics''' is the branch of physics that deals with heat."
            terms = client.extract_lead_bold_terms(wikitext)
            try:
                import mwparserfromhell  # noqa: F401
            except ImportError:
                self.skipTest("mwparserfromhell not installed in test environment")
            else:
                self.assertIn("Thermodynamics", terms)
        finally:
            shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
