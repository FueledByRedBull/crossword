from __future__ import annotations

import json
import time
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .cache import DiskCache

try:
    import mwparserfromhell
except ImportError:  # pragma: no cover - optional at import time
    mwparserfromhell = None


class WikiClient:
    def __init__(
        self,
        cache: DiskCache,
        *,
        api_base: str = "https://en.wikipedia.org/w/api.php",
        timeout_seconds: int = 30,
        user_agent: str = "CrosswordPipeline/0.1 (seed-stage)",
        max_retries: int = 2,
        backoff_seconds: float = 0.5,
        backoff_factor: float = 2.0,
    ):
        self.cache = cache
        self.api_base = api_base
        self.timeout_seconds = timeout_seconds
        self.user_agent = user_agent
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.backoff_factor = backoff_factor

    def _query(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        cached = self.cache.get(endpoint, params)
        if cached is not None:
            return cached

        query = urlencode({k: v for k, v in params.items() if v is not None}, doseq=True)
        request = Request(
            f"{self.api_base}?{query}",
            headers={"User-Agent": self.user_agent},
        )
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:  # nosec B310
                    payload = json.loads(response.read().decode("utf-8"))
                self.cache.set(endpoint, params, payload)
                return payload
            except Exception as exc:  # pragma: no cover - network path
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                sleep_for = self.backoff_seconds * (self.backoff_factor**attempt)
                time.sleep(sleep_for)
        raise RuntimeError(f"request_failed:{endpoint}:{last_exc}")

    @staticmethod
    def _base_query_params() -> dict[str, Any]:
        return {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "redirects": "1",
        }

    def fetch_links(self, title: str, *, max_links: int | None = None) -> dict[str, Any]:
        endpoint = "w_api_query_links"
        params = self._base_query_params()
        params.update(
            {
                "prop": "links",
                "titles": title,
                "plnamespace": "0",
                "pllimit": "max",
            }
        )

        links: list[dict[str, Any]] = []
        seen: set[str] = set()
        resolved_title = title
        page_id: int | None = None

        while True:
            data = self._query(endpoint, params)
            pages = data.get("query", {}).get("pages", [])
            if not pages:
                break
            page = pages[0]
            page_id = page.get("pageid")
            resolved_title = page.get("title", resolved_title)
            for link in page.get("links", []):
                if link.get("ns") != 0:
                    continue
                link_title = link.get("title")
                if not link_title or link_title in seen:
                    continue
                seen.add(link_title)
                links.append(
                    {
                        "title": link_title,
                        "page_id": link.get("pageid"),
                        "depth": 1,
                    }
                )
                if max_links is not None and len(links) >= max_links:
                    return {"seed_title": resolved_title, "seed_page_id": page_id, "links": links}

            continuation = data.get("continue")
            if continuation is None:
                break
            params.update(continuation)

        return {"seed_title": resolved_title, "seed_page_id": page_id, "links": links}

    def fetch_backlinks(self, seed_title: str, *, max_backlinks: int | None = None) -> set[str]:
        endpoint = "w_api_query_backlinks"
        params = self._base_query_params()
        params.update(
            {
                "list": "backlinks",
                "bltitle": seed_title,
                "blnamespace": "0",
                "bllimit": "max",
            }
        )

        backlinks: set[str] = set()
        while True:
            data = self._query(endpoint, params)
            items = data.get("query", {}).get("backlinks", [])
            for item in items:
                title = item.get("title")
                if title:
                    backlinks.add(title)
                if max_backlinks is not None and len(backlinks) >= max_backlinks:
                    return backlinks

            continuation = data.get("continue")
            if continuation is None:
                break
            params.update(continuation)
        return backlinks

    def fetch_lead_wikitext(self, title: str) -> dict[str, Any]:
        endpoint = "w_api_query_lead_wikitext"
        params = self._base_query_params()
        params.update(
            {
                "prop": "revisions",
                "rvprop": "ids|content",
                "rvslots": "main",
                "rvsection": "0",
                "rvlimit": "1",
                "titles": title,
            }
        )
        data = self._query(endpoint, params)
        pages = data.get("query", {}).get("pages", [])
        if not pages:
            return {"title": title, "page_id": None, "revid": None, "wikitext": ""}
        page = pages[0]
        revisions = page.get("revisions", [])
        if not revisions:
            return {"title": page.get("title", title), "page_id": page.get("pageid"), "revid": None, "wikitext": ""}

        revision = revisions[0]
        slots = revision.get("slots", {})
        main_slot = slots.get("main", {})
        return {
            "title": page.get("title", title),
            "page_id": page.get("pageid"),
            "revid": revision.get("revid"),
            "wikitext": main_slot.get("content", ""),
        }

    def fetch_page_extract(self, title: str, *, intro_only: bool = True) -> dict[str, Any]:
        endpoint = "w_api_query_extracts_intro" if intro_only else "w_api_query_extracts_full"
        params = self._base_query_params()
        params.update(
            {
                "prop": "extracts",
                "explaintext": "1",
                "exsectionformat": "plain",
                "exintro": "1" if intro_only else None,
                "titles": title,
            }
        )
        data = self._query(endpoint, params)
        pages = data.get("query", {}).get("pages", [])
        if not pages:
            return {"title": title, "page_id": None, "extract": ""}
        page = pages[0]
        return {
            "title": page.get("title", title),
            "page_id": page.get("pageid"),
            "extract": page.get("extract", ""),
        }

    def fetch_page_extract_with_revid(self, title: str, *, intro_only: bool = True) -> dict[str, Any]:
        endpoint = (
            "w_api_query_extracts_intro_revid" if intro_only else "w_api_query_extracts_full_revid"
        )
        params = self._base_query_params()
        params.update(
            {
                "prop": "extracts|revisions",
                "rvprop": "ids",
                "rvslots": "main",
                "explaintext": "1",
                "exsectionformat": "plain",
                "exintro": "1" if intro_only else None,
                "titles": title,
            }
        )
        data = self._query(endpoint, params)
        pages = data.get("query", {}).get("pages", [])
        if not pages:
            return {"title": title, "page_id": None, "extract": "", "revid": None}
        page = pages[0]
        revisions = page.get("revisions", [])
        revid = None
        if revisions:
            revid = revisions[0].get("revid")
        return {
            "title": page.get("title", title),
            "page_id": page.get("pageid"),
            "extract": page.get("extract", ""),
            "revid": revid,
        }

    def extract_lead_bold_terms(self, wikitext: str) -> list[str]:
        if not wikitext or mwparserfromhell is None:
            return []

        code = mwparserfromhell.parse(wikitext)
        terms: list[str] = []
        seen: set[str] = set()

        for node in code.ifilter(recursive=True):
            class_name = node.__class__.__name__
            if class_name != "Bold":
                continue
            try:
                raw = node.strip_code().strip()
            except Exception:
                raw = mwparserfromhell.parse(str(node)).strip_code().strip()
            cleaned = " ".join(raw.split())
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                terms.append(cleaned)

        for tag in code.ifilter_tags(recursive=True):
            tag_name = str(tag.tag).strip().lower()
            if tag_name not in {"b", "strong"}:
                continue
            contents = tag.contents
            try:
                raw = contents.strip_code().strip()
            except Exception:
                raw = mwparserfromhell.parse(str(contents)).strip_code().strip()
            cleaned = " ".join(raw.split())
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                terms.append(cleaned)

        return terms
