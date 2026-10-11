from __future__ import annotations

import json
import os
import time
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from .cache import DiskCache

try:
    import mwparserfromhell
except ImportError:  # pragma: no cover - optional at import time
    mwparserfromhell = None

_RATE_LIMIT_STATUSES = (429, 503)
_RETRY_AFTER_DEFAULT = 60.0
_RETRY_AFTER_MIN = 30.0
_RETRY_AFTER_MAX = 300.0
_MAX_INTERVAL_SECONDS = 8.0
_MAX_GET_QUERY = 2000


class WikiClient:
    # One pacing clock for the whole process: every stage builds its own client, and a rate-limit wait must hold them all back.
    # A rate-limited wiki also gets a longer interval from then on, so the client settles under its limit.
    _next_request_at = 0.0
    _interval = 0.0

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
        offline: bool | None = None,
        min_interval_seconds: float = 1.0,
    ):
        self.cache = cache
        self.api_base = api_base
        # Other wikis get their own cache namespace; English keeps bare endpoints
        # so cache entries written before namespacing stay valid.
        host = urlparse(api_base).hostname
        self._cache_prefix = "" if host == "en.wikipedia.org" else f"{host}_"
        self.timeout_seconds = timeout_seconds
        # Wikimedia asks clients to identify themselves; the owner supplies contact details here.
        self.user_agent = os.getenv("CROSSWORD_USER_AGENT") or user_agent
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.backoff_factor = backoff_factor
        self.min_interval_seconds = min_interval_seconds
        if offline is None:
            offline = os.getenv("CROSSWORD_OFFLINE", "").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        self.offline = offline

    def _query(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        cache_endpoint = f"{self._cache_prefix}{endpoint}"
        cached = self.cache.get(cache_endpoint, params)
        if cached is not None:
            return cached
        if self.offline:
            raise RuntimeError(f"offline_cache_miss:{endpoint}")

        payload = self._fetch(endpoint, params)
        self.cache.set(cache_endpoint, params, payload)
        return payload

    def _fetch(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        """Every network request goes through here: paced, and patient with rate limits."""
        query = urlencode({k: v for k, v in params.items() if v is not None}, doseq=True)
        headers = {"User-Agent": self.user_agent}
        if len(query) > _MAX_GET_QUERY:  # a batch of long or non-Latin titles overflows a request line
            request = Request(self.api_base, data=query.encode("ascii"), headers=headers)
        else:
            request = Request(f"{self.api_base}?{query}", headers=headers)
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            wait = WikiClient._next_request_at - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:  # nosec B310
                    return json.loads(response.read().decode("utf-8"))
            except Exception as exc:
                last_exc = exc
                if isinstance(exc, HTTPError) and exc.code in _RATE_LIMIT_STATUSES:
                    WikiClient._interval = min(2 * self._pacing_interval(), _MAX_INTERVAL_SECONDS)
            finally:
                WikiClient._next_request_at = time.monotonic() + self._pacing_interval()
            if attempt < self.max_retries:
                delay = self._retry_delay(last_exc, attempt)
                WikiClient._next_request_at = max(WikiClient._next_request_at, time.monotonic() + delay)
        raise RuntimeError(f"request_failed:{endpoint}:{last_exc}")

    def _pacing_interval(self) -> float:
        return max(WikiClient._interval, self.min_interval_seconds)

    def _retry_delay(self, exc: Exception, attempt: int) -> float:
        if isinstance(exc, HTTPError) and exc.code in _RATE_LIMIT_STATUSES:
            try:
                retry_after = float((exc.headers or {}).get("Retry-After"))
            except (TypeError, ValueError):
                retry_after = _RETRY_AFTER_DEFAULT
            return min(max(retry_after, _RETRY_AFTER_MIN), _RETRY_AFTER_MAX)
        return self.backoff_seconds * (self.backoff_factor**attempt)

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

    def _lead_request(self, title: str) -> tuple[str, dict[str, Any]]:
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
        return "w_api_query_lead_wikitext", params

    def _extract_request(
        self, title: str, *, intro_only: bool, with_revid: bool
    ) -> tuple[str, dict[str, Any]]:
        endpoint = f"w_api_query_extracts_{'intro' if intro_only else 'full'}{'_revid' if with_revid else ''}"
        params = self._base_query_params()
        params.update(
            {
                "prop": "extracts|revisions" if with_revid else "extracts",
                "explaintext": "1",
                "exsectionformat": "plain",
                "exintro": "1" if intro_only else None,
                "titles": title,
            }
        )
        if with_revid:
            params.update({"rvprop": "ids", "rvslots": "main"})
        return endpoint, params

    def prefetch_lead_wikitext(self, titles: list[str]) -> None:
        """Cache lead wikitext for many titles at once (50 per request)."""
        self._prefetch(titles, self._lead_request, {"rvlimit": None}, 50, "revisions")

    def prefetch_intro_extracts(self, titles: list[str], *, with_revid: bool = False) -> None:
        """Cache intro extracts, with or without revision ids, for many titles at once (20 per request)."""
        self._prefetch(
            titles,
            lambda title: self._extract_request(title, intro_only=True, with_revid=with_revid),
            {"exlimit": "max"},
            20,
            "extract",
        )

    def _prefetch(self, titles, request_for, batch_params, size, page_key) -> None:
        """Store one cache entry per title under the key its single-title getter reads."""
        if self.offline:
            return
        pending = []
        for title in dict.fromkeys(titles):
            endpoint, params = request_for(title)
            if title and self.cache.get(f"{self._cache_prefix}{endpoint}", params) is None:
                pending.append(title)
        while pending:
            unresolved: list[str] = []
            for start in range(0, len(pending), size):
                unresolved += self._prefetch_chunk(
                    pending[start : start + size], request_for, batch_params, page_key
                )
            if len(unresolved) == len(pending):
                break  # no progress; the single-title getters will ask for these themselves
            pending = unresolved

    def _prefetch_chunk(self, titles, request_for, batch_params, page_key) -> list[str]:
        endpoint, params = request_for("|".join(titles))
        query = self._fetch(endpoint, {**params, **batch_params}).get("query", {})
        hops = {
            item["from"]: item["to"]
            for step in ("normalized", "redirects")
            for item in query.get(step, [])
        }
        pages = {page.get("title"): page for page in query.get("pages", [])}
        unresolved = []
        for title in titles:
            target = title
            for _ in hops:  # follows redirect chains; a cycle cannot loop forever
                target = hops.get(target, target)
            page = pages.get(target)
            # A page cut off by the response size limit lacks page_key; ask again for it.
            if page is None or not (page_key in page or page.get("missing") or page.get("invalid")):
                unresolved.append(title)
                continue
            _, single_params = request_for(title)
            self.cache.set(
                f"{self._cache_prefix}{endpoint}",
                single_params,
                {"batchcomplete": True, "query": {"pages": [page]}},
            )
        return unresolved

    def fetch_lead_wikitext(self, title: str) -> dict[str, Any]:
        endpoint, params = self._lead_request(title)
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
        endpoint, params = self._extract_request(title, intro_only=intro_only, with_revid=False)
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
        endpoint, params = self._extract_request(title, intro_only=intro_only, with_revid=True)
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

    @staticmethod
    def _fetch_each(titles, fetch, empty) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
        payloads: dict[str, dict[str, Any]] = {}
        errors: dict[str, str] = {}
        for title in dict.fromkeys(title for title in titles if title):
            try:
                payloads[title] = fetch(title)
            except Exception as exc:
                payloads[title] = empty(title)
                errors[title] = str(exc)
        return payloads, errors

    def fetch_page_extracts_concurrent(
        self,
        titles: list[str],
        *,
        intro_only: bool = True,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
        return self._fetch_each(
            titles,
            lambda title: self.fetch_page_extract(title, intro_only=intro_only),
            lambda title: {"title": title, "page_id": None, "extract": ""},
        )

    def fetch_links_concurrent(
        self,
        titles: list[str],
        *,
        max_links: int | None = None,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
        return self._fetch_each(
            titles,
            lambda title: self.fetch_links(title, max_links=max_links),
            lambda title: {"seed_title": title, "seed_page_id": None, "links": []},
        )

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
