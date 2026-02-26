from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_seed_stage_diagnostics(
    *,
    lang: str,
    seed_requested: str,
    seed_resolved: str,
    seed_page_id: int | None,
    candidates: list[dict[str, Any]],
    cache_stats: dict[str, int],
    include_backlinks: bool,
    errors: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "0.1.0",
        "stage": "seed_ingestion",
        "generated_at": _utc_now_iso(),
        "lang": lang,
        "seed": {
            "requested_title": seed_requested,
            "resolved_title": seed_resolved,
            "page_id": seed_page_id,
        },
        "counts": {
            "candidate_count": len(candidates),
            "backlink_annotation_enabled": include_backlinks,
        },
        "cache": cache_stats,
        "candidates": candidates,
        "errors": errors or [],
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
    return output_path
