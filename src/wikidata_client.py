from __future__ import annotations

import json
import time
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .cache import DiskCache


class WikidataClient:
    def __init__(
        self,
        cache: DiskCache,
        *,
        api_base: str = "https://www.wikidata.org/w/api.php",
        timeout_seconds: int = 30,
        user_agent: str = "CrosswordPipeline/0.1 (wikidata)",
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
        raise RuntimeError(f"wikidata_request_failed:{endpoint}:{last_exc}")

    def search_entity(self, term: str, *, lang: str = "en") -> str | None:
        params = {
            "action": "wbsearchentities",
            "search": term,
            "language": lang,
            "format": "json",
            "limit": 1,
        }
        data = self._query("wbsearchentities", params)
        results = data.get("search", [])
        if not results:
            return None
        return results[0].get("id")

    def fetch_instance_of(self, entity_id: str) -> set[str]:
        params = {
            "action": "wbgetentities",
            "ids": entity_id,
            "props": "claims",
            "format": "json",
        }
        data = self._query("wbgetentities", params)
        entities = data.get("entities", {})
        entity = entities.get(entity_id, {})
        claims = entity.get("claims", {})
        p31_claims = claims.get("P31", [])
        types: set[str] = set()
        for claim in p31_claims:
            mainsnak = claim.get("mainsnak", {})
            datavalue = mainsnak.get("datavalue", {})
            value = datavalue.get("value", {})
            qid = value.get("id")
            if qid:
                types.add(qid)
        return types

