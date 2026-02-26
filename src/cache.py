from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Mapping


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalize(value[key]) for key in sorted(value.keys(), key=str)}
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize(item) for item in value]
    if isinstance(value, set):
        normalized = [_normalize(item) for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def canonical_params_json(params: Mapping[str, Any]) -> str:
    normalized = _normalize(params)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@dataclass(frozen=True)
class CacheKey:
    endpoint: str
    params_hash: str


def make_cache_key(endpoint: str, params: Mapping[str, Any]) -> CacheKey:
    canonical = canonical_params_json(params)
    digest = hashlib.sha256(f"{endpoint}\n{canonical}".encode("utf-8")).hexdigest()
    return CacheKey(endpoint=endpoint, params_hash=digest)


class DiskCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._stats = {"gets": 0, "hits": 0, "misses": 0, "writes": 0}

    @staticmethod
    def _safe_endpoint(endpoint: str) -> str:
        return endpoint.replace("/", "_").replace("\\", "_").replace(":", "_")

    def _path_for_key(self, key: CacheKey) -> Path:
        endpoint_dir = self.root / self._safe_endpoint(key.endpoint)
        endpoint_dir.mkdir(parents=True, exist_ok=True)
        return endpoint_dir / f"{key.params_hash}.json"

    def get(self, endpoint: str, params: Mapping[str, Any]) -> Any | None:
        self._stats["gets"] += 1
        key = make_cache_key(endpoint, params)
        path = self._path_for_key(key)
        if not path.exists():
            self._stats["misses"] += 1
            return None

        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        self._stats["hits"] += 1
        return data.get("payload")

    def set(self, endpoint: str, params: Mapping[str, Any], payload: Any) -> None:
        key = make_cache_key(endpoint, params)
        path = self._path_for_key(key)
        envelope = {
            "endpoint": endpoint,
            "params_hash": key.params_hash,
            "cached_at": _utc_now_iso(),
            "payload": payload,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent) as tmp_file:
            json.dump(envelope, tmp_file, ensure_ascii=True)
            tmp_path = Path(tmp_file.name)
        tmp_path.replace(path)
        self._stats["writes"] += 1

    def stats(self) -> dict[str, int]:
        return dict(self._stats)

