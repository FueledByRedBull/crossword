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

    def test_fetch_links_paginates_and_dedupes(self) -> None:
        cache_dir = Path("tests") / "tmp_cache_wiki_client_links"
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            cache = DiskCache(cache_dir)
            client = WikiClient(cache)
            responses = [
                {
                    "query": {
                        "pages": [
                            {
                                "pageid": 1,
                                "title": "Seed",
                                "links": [
                                    {"title": "Alpha", "ns": 0},
                                    {"title": "Beta", "ns": 0},
                                    {"title": "Alpha", "ns": 0},
                                    {"title": "Talk:Skip", "ns": 1},
                                ],
                            }
                        ]
                    },
                    "continue": {"plcontinue": "x"},
                },
                {
                    "query": {
                        "pages": [
                            {
                                "pageid": 1,
                                "title": "Seed",
                                "links": [
                                    {"title": "Gamma", "ns": 0},
                                    {"title": "Beta", "ns": 0},
                                ],
                            }
                        ]
                    }
                },
            ]

            def _query(_endpoint, _params):
                return responses.pop(0)

            client._query = _query  # type: ignore[assignment]
            result = client.fetch_links("Seed")
            titles = [item["title"] for item in result["links"]]
            self.assertEqual(titles, ["Alpha", "Beta", "Gamma"])
        finally:
            shutil.rmtree(cache_dir, ignore_errors=True)

    def test_fetch_backlinks_paginates_and_dedupes(self) -> None:
        cache_dir = Path("tests") / "tmp_cache_wiki_client_backlinks"
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            cache = DiskCache(cache_dir)
            client = WikiClient(cache)
            responses = [
                {
                    "query": {"backlinks": [{"title": "Alpha"}, {"title": "Beta"}]},
                    "continue": {"blcontinue": "x"},
                },
                {
                    "query": {"backlinks": [{"title": "Beta"}, {"title": "Gamma"}]},
                },
            ]

            def _query(_endpoint, _params):
                return responses.pop(0)

            client._query = _query  # type: ignore[assignment]
            backlinks = client.fetch_backlinks("Seed")
            self.assertEqual(backlinks, {"Alpha", "Beta", "Gamma"})
        finally:
            shutil.rmtree(cache_dir, ignore_errors=True)

    def test_fetch_page_extracts_concurrent_returns_payloads_and_errors(self) -> None:
        cache_dir = Path("tests") / "tmp_cache_wiki_client_extracts"
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            cache = DiskCache(cache_dir)
            client = WikiClient(cache)

            def _fetch_page_extract(title: str, *, intro_only: bool = True):  # noqa: ARG001
                if title == "Broken":
                    raise RuntimeError("boom")
                return {"title": title, "page_id": 1, "extract": f"{title} extract"}

            client.fetch_page_extract = _fetch_page_extract  # type: ignore[assignment]
            payloads, errors = client.fetch_page_extracts_concurrent(["Alpha", "Broken", "Beta"])

            self.assertEqual(payloads["Alpha"]["extract"], "Alpha extract")
            self.assertEqual(payloads["Beta"]["extract"], "Beta extract")
            self.assertIn("Broken", errors)
        finally:
            shutil.rmtree(cache_dir, ignore_errors=True)

    def test_fetch_links_concurrent_returns_payloads_and_errors(self) -> None:
        cache_dir = Path("tests") / "tmp_cache_wiki_client_links_bulk"
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            cache = DiskCache(cache_dir)
            client = WikiClient(cache)

            def _fetch_links(title: str, *, max_links: int | None = None):  # noqa: ARG001
                if title == "Broken":
                    raise RuntimeError("boom")
                return {"seed_title": title, "seed_page_id": 1, "links": [{"title": f"{title}Link"}]}

            client.fetch_links = _fetch_links  # type: ignore[assignment]
            payloads, errors = client.fetch_links_concurrent(["Alpha", "Broken", "Beta"])

            self.assertEqual(payloads["Alpha"]["links"][0]["title"], "AlphaLink")
            self.assertEqual(payloads["Beta"]["links"][0]["title"], "BetaLink")
            self.assertIn("Broken", errors)
        finally:
            shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
