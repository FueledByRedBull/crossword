import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse

from src.cache import DiskCache
from src.wiki_client import WikiClient


class FakeWiki:
    """Stands in for urlopen: answers lead / extract batches the way the API shapes them."""

    def __init__(self):
        self.redirects: dict[str, str] = {}
        self.truncate_after: int | None = None
        self.never: set[str] = set()
        self.ids: dict[str, int] = {}
        self.requests: list[dict[str, str]] = []

    def __call__(self, request, timeout=None):
        params = {k: v[0] for k, v in parse_qs(urlparse(request.full_url).query).items()}
        self.requests.append(params)
        normalized, redirects, pages, filled = [], [], [], 0
        for requested in params["titles"].split("|"):
            title = requested.replace("_", " ")
            title = title[:1].upper() + title[1:]
            if title != requested:
                normalized.append({"from": requested, "to": title})
            while title in self.redirects:
                redirects.append({"from": title, "to": self.redirects[title]})
                title = self.redirects[title]
            if title.startswith("Nope"):
                pages.append({"ns": 0, "title": title, "missing": True})
                continue
            page_id = self.ids.setdefault(title, len(self.ids) + 1)
            page = {"pageid": page_id, "ns": 0, "title": title}
            if title not in self.never and (self.truncate_after is None or filled < self.truncate_after):
                filled += 1
                if "rvsection" in params:
                    content = f"'''{title}''' lead"
                    page["revisions"] = [{"revid": 1000 + page_id, "slots": {"main": {"content": content}}}]
                else:
                    page["extract"] = f"{title} extract"
                    if "rvprop" in params:
                        page["revisions"] = [{"revid": 1000 + page_id}]
            pages.append(page)
        query = {"normalized": normalized, "redirects": redirects, "pages": pages}
        body = {"batchcomplete": True, "query": {k: v for k, v in query.items() if v}}
        return io.BytesIO(json.dumps(body).encode("utf-8"))


class FakeClock:
    def __init__(self):
        self.now = 1000.0
        self.sleeps: list[float] = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def rate_limited(headers=None, code=429):
    return HTTPError("https://example.test", code, "limited", headers or {}, None)


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

    def test_offline_cache_miss_raises(self) -> None:
        cache_dir = Path("tests") / "tmp_cache_wiki_client_offline"
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            cache = DiskCache(cache_dir)
            client = WikiClient(cache, offline=True)
            with self.assertRaisesRegex(RuntimeError, "offline_cache_miss"):
                client.fetch_page_extract("Seed")
        finally:
            shutil.rmtree(cache_dir, ignore_errors=True)

    def test_cache_entries_are_not_shared_across_wikis(self) -> None:
        cache_dir = Path("tests") / "tmp_cache_wiki_client_lang"
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            cache = DiskCache(cache_dir)
            english = WikiClient(cache, offline=True)
            greek = WikiClient(cache, api_base="https://el.wikipedia.org/w/api.php", offline=True)
            params = {"titles": "DNA"}
            cache.set("w_api_query_links", params, {"query": {"pages": []}})

            self.assertEqual(english._query("w_api_query_links", params), {"query": {"pages": []}})
            with self.assertRaisesRegex(RuntimeError, "offline_cache_miss"):
                greek._query("w_api_query_links", params)
        finally:
            shutil.rmtree(cache_dir, ignore_errors=True)


class TransportTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cache = DiskCache(tmp.name)
        self.wiki = FakeWiki()
        self.clock = FakeClock()
        for target in (
            patch("src.wiki_client.urlopen", side_effect=self.wiki),
            patch("src.wiki_client.time", self.clock),
            patch.object(WikiClient, "_next_request_at", 0.0),
            patch.object(WikiClient, "_interval", 0.0),
        ):
            target.start()
            self.addCleanup(target.stop)

    def client(self, **kwargs) -> WikiClient:
        return WikiClient(self.cache, offline=False, **kwargs)

    def titles_requested(self) -> list[str]:
        return [request["titles"] for request in self.wiki.requests]

    def test_prefetch_batches_requests_and_single_getters_hit_the_cache(self) -> None:
        client = self.client()
        titles = [f"Page {i}" for i in range(120)]

        client.prefetch_lead_wikitext(titles)
        self.assertEqual([len(t.split("|")) for t in self.titles_requested()], [50, 50, 20])
        self.assertTrue(all("rvlimit" not in request for request in self.wiki.requests))
        client.prefetch_intro_extracts(titles[:45])
        client.prefetch_intro_extracts(titles[:45], with_revid=True)
        self.assertEqual(len(self.wiki.requests), 3 + 3 + 3)
        self.assertTrue(all(r.get("exlimit") == "max" for r in self.wiki.requests[3:]))

        requests_before = len(self.wiki.requests)
        page_id = self.wiki.ids["Page 7"]
        self.assertEqual(
            client.fetch_lead_wikitext("Page 7"),
            {"title": "Page 7", "page_id": page_id, "revid": 1000 + page_id, "wikitext": "'''Page 7''' lead"},
        )
        self.assertEqual(
            client.fetch_page_extract("Page 7", intro_only=True),
            {"title": "Page 7", "page_id": page_id, "extract": "Page 7 extract"},
        )
        self.assertEqual(
            client.fetch_page_extract_with_revid("Page 7", intro_only=True),
            {"title": "Page 7", "page_id": page_id, "extract": "Page 7 extract", "revid": 1000 + page_id},
        )
        for title in titles[:45]:
            client.fetch_lead_wikitext(title)
            client.fetch_page_extract(title, intro_only=True)
            client.fetch_page_extract_with_revid(title, intro_only=True)
        self.assertEqual(len(self.wiki.requests), requests_before)

    def test_prefetch_follows_normalized_titles_and_redirect_chains(self) -> None:
        client = self.client()
        self.wiki.redirects = {"A": "B", "B": "C"}

        client.prefetch_lead_wikitext(["quantum_mechanics", "A", "B"])

        self.assertEqual(len(self.wiki.requests), 1)
        self.assertEqual(client.fetch_lead_wikitext("quantum_mechanics")["title"], "Quantum mechanics")
        for requested in ("A", "B"):
            lead = client.fetch_lead_wikitext(requested)
            self.assertEqual((lead["title"], lead["wikitext"]), ("C", "'''C''' lead"))
        self.assertEqual(len(self.wiki.requests), 1)

    def test_prefetch_completes_truncated_responses_and_caches_missing_pages(self) -> None:
        client = self.client()
        self.wiki.truncate_after = 1

        client.prefetch_lead_wikitext(["A", "B", "C", "Nope", "D"])

        self.assertEqual(self.titles_requested(), ["A|B|C|Nope|D", "B|C|D", "C|D", "D"])
        self.assertEqual(
            client.fetch_lead_wikitext("Nope"),
            {"title": "Nope", "page_id": None, "revid": None, "wikitext": ""},
        )
        self.assertEqual(client.fetch_lead_wikitext("D")["wikitext"], "'''D''' lead")
        self.assertEqual(len(self.wiki.requests), 4)

    def test_prefetch_gives_up_on_a_title_that_never_resolves(self) -> None:
        client = self.client()
        self.wiki.never = {"Stuck"}

        client.prefetch_lead_wikitext(["A", "Stuck"])

        self.assertEqual(self.titles_requested(), ["A|Stuck", "Stuck"])
        client.fetch_lead_wikitext("A")
        self.assertEqual(len(self.wiki.requests), 2)

    def test_prefetch_skips_cached_titles_and_is_a_noop_offline(self) -> None:
        client = self.client()
        client.prefetch_intro_extracts(["X1", "X2"])
        client.prefetch_intro_extracts(["X1", "X2", "X3", ""])
        self.assertEqual(self.titles_requested(), ["X1|X2", "X3"])
        client.prefetch_intro_extracts(["X1", "X2", "X3"])
        self.assertEqual(len(self.wiki.requests), 2)

        offline = WikiClient(DiskCache(Path(self.cache.root) / "empty"), offline=True)
        offline.prefetch_lead_wikitext(["Y"])
        offline.prefetch_intro_extracts(["Y"], with_revid=True)
        self.assertEqual(len(self.wiki.requests), 2)

    def test_prefetch_uses_the_wiki_specific_cache_namespace(self) -> None:
        greek = {"api_base": "https://el.wikipedia.org/w/api.php"}
        self.client(**greek).prefetch_lead_wikitext(["Page"])

        self.assertEqual(WikiClient(self.cache, offline=True, **greek).fetch_lead_wikitext("Page")["title"], "Page")
        with self.assertRaisesRegex(RuntimeError, "offline_cache_miss"):
            WikiClient(self.cache, offline=True).fetch_lead_wikitext("Page")

    def test_rate_limit_waits_for_retry_after_within_bounds_then_retries(self) -> None:
        cases = [
            ({"Retry-After": "7"}, 429, 30.0),
            ({}, 429, 60.0),
            ({"Retry-After": "120"}, 429, 120.0),
            ({"Retry-After": "9999"}, 503, 300.0),
            ({"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}, 429, 60.0),
        ]
        for headers, code, expected in cases:
            with self.subTest(headers=headers, code=code):
                self.clock.sleeps.clear()
                WikiClient._next_request_at = 0.0
                answers = [rate_limited(headers, code), io.BytesIO(b'{"query": {}}')]
                with patch("src.wiki_client.urlopen", side_effect=answers) as urlopen:
                    self.assertEqual(self.client()._fetch("e", {"titles": "a"}), {"query": {}})
                self.assertEqual(urlopen.call_count, 2)
                self.assertEqual(self.clock.sleeps, [expected])

    def test_other_errors_keep_the_short_backoff_and_retries_are_bounded(self) -> None:
        with patch("src.wiki_client.urlopen", side_effect=[OSError("reset"), rate_limited(code=500)]) as urlopen:
            with self.assertRaisesRegex(RuntimeError, "request_failed:e"):
                self.client(max_retries=1)._fetch("e", {"titles": "a"})
        self.assertEqual(urlopen.call_count, 2)
        self.assertEqual(self.clock.sleeps, [1.0])  # the 0.5 s backoff is inside the 1 s interval

    def test_requests_are_paced(self) -> None:
        client = self.client()
        client._fetch("e", {"titles": "a"})
        client._fetch("e", {"titles": "b"})
        client._fetch("e", {"titles": "c"})
        self.assertEqual(self.clock.sleeps, [1.0, 1.0])

        self.clock.sleeps.clear()
        WikiClient._next_request_at = 0.0
        slow = self.client(min_interval_seconds=2.5)
        slow._fetch("e", {"titles": "a"})
        slow._fetch("e", {"titles": "b"})
        self.assertEqual(self.clock.sleeps, [2.5])

    def test_pacing_is_shared_by_all_clients(self) -> None:
        self.client()._fetch("e", {"titles": "a"})
        self.client()._fetch("e", {"titles": "b"})
        self.assertEqual(self.clock.sleeps, [1.0])

    def test_rate_limit_responses_double_the_interval_for_every_client_up_to_a_cap(self) -> None:
        limited = rate_limited({"Retry-After": "7"})
        with patch("src.wiki_client.urlopen", side_effect=[limited, limited, io.BytesIO(b"{}")]):
            self.client(max_retries=2)._fetch("e", {"titles": "a"})  # interval 1 -> 2 -> 4
        self.assertEqual(self.clock.sleeps, [30.0, 30.0])

        self.clock.sleeps.clear()
        self.client()._fetch("e", {"titles": "b"})
        self.assertEqual(self.clock.sleeps, [4.0])

        WikiClient._interval = 6.0
        with patch("src.wiki_client.urlopen", side_effect=[rate_limited(code=503)]):
            with self.assertRaisesRegex(RuntimeError, "request_failed"):
                self.client(max_retries=0)._fetch("e", {"titles": "c"})
        self.assertEqual(WikiClient._interval, 8.0)

    def test_long_queries_are_posted_and_short_ones_stay_in_the_url(self) -> None:
        athens = "".join(map(chr, (0x391, 0x3B8, 0x3AE, 0x3BD, 0x3B1)))  # six characters each once encoded
        greek = "|".join(f"{athens} {athens} {i}" for i in range(50))
        client = self.client()
        with patch("src.wiki_client.urlopen", return_value=io.BytesIO(b"{}")) as urlopen:
            client._fetch("e", {"action": "query", "titles": greek})
        request = urlopen.call_args.args[0]
        self.assertEqual((request.get_method(), request.full_url), ("POST", client.api_base))
        self.assertEqual(parse_qs(request.data.decode("ascii")), {"action": ["query"], "titles": [greek]})
        self.assertEqual(request.get_header("User-agent"), client.user_agent)

        with patch("src.wiki_client.urlopen", return_value=io.BytesIO(b"{}")) as urlopen:
            client._fetch("e", {"titles": "a"})
        request = urlopen.call_args.args[0]
        self.assertEqual((request.get_method(), request.data), ("GET", None))
        self.assertEqual(request.full_url, f"{client.api_base}?titles=a")

    def test_concurrent_methods_are_serial_and_paced(self) -> None:
        client = self.client()
        payloads, errors = client.fetch_page_extracts_concurrent(["Alpha", "", "Beta", "Alpha"])
        self.assertEqual(list(payloads), ["Alpha", "Beta"])
        self.assertEqual(payloads["Beta"]["extract"], "Beta extract")
        self.assertEqual(errors, {})
        self.assertEqual(self.clock.sleeps, [1.0])

    def test_user_agent_defaults_and_environment_override(self) -> None:
        with patch.dict(os.environ):
            os.environ.pop("CROSSWORD_USER_AGENT", None)
            self.assertTrue(self.client().user_agent.startswith("CrosswordPipeline/"))
            os.environ["CROSSWORD_USER_AGENT"] = "Crossword/1 (contact)"
            client = self.client()
        with patch("src.wiki_client.urlopen", return_value=io.BytesIO(b"{}")) as urlopen:
            client._fetch("e", {"titles": "a"})
        self.assertEqual(urlopen.call_args.args[0].get_header("User-agent"), "Crossword/1 (contact)")


if __name__ == "__main__":
    unittest.main()
