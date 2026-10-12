from __future__ import annotations

from pathlib import Path

from ..cache import DiskCache
from ..diagnostics import build_seed_stage_diagnostics, write_json
from ..semantic import build_tfidf_vectors, cosine_similarity
from ..wiki_client import WikiClient

from .common import SeedStageResult, _prefetch


def run_seed_ingestion_stage(
    *,
    seed_title: str,
    lang: str = "en",
    diagnostics_path: str | Path = "outputs/diagnostics.json",
    cache_dir: str | Path = "data/cache/wiki",
    include_backlinks: bool = True,
    max_links: int | None = None,
    max_backlinks: int | None = None,
    expansion: str = "one_hop_only",
    max_two_hop_parents: int | None = None,
    max_two_hop_links: int | None = None,
    max_candidates: int | None = None,
) -> SeedStageResult:
    errors: list[str] = []
    cache = DiskCache(cache_dir)
    allowed_langs = {"en", "el"}
    if lang not in allowed_langs:
        errors.append(f"unsupported_lang: {lang}")
        diagnostics = build_seed_stage_diagnostics(
            lang=lang,
            seed_requested=seed_title,
            seed_resolved=seed_title,
            seed_page_id=None,
            candidates=[],
            cache_stats=cache.stats(),
            include_backlinks=include_backlinks,
            errors=errors,
        )
        final_path = write_json(diagnostics_path, diagnostics)
        return SeedStageResult(diagnostics_path=final_path, diagnostics=diagnostics)

    api_base = f"https://{lang}.wikipedia.org/w/api.php"
    client = WikiClient(cache, api_base=api_base)

    links = []
    resolved_seed_title = seed_title
    resolved_seed_page_id: int | None = None
    try:
        links_payload = client.fetch_links(seed_title)
        links = links_payload["links"]
        if max_links is not None and len(links) > max_links:
            # Evenly spaced, not the first N: the API lists links alphabetically.
            links = [links[i * len(links) // max_links] for i in range(max_links)]
        resolved_seed_title = links_payload["seed_title"]
        resolved_seed_page_id = links_payload["seed_page_id"]
    except RuntimeError as exc:  # pragma: no cover - runtime/network path
        errors.append(f"link_fetch_failed: {exc}")

    backlinks: set[str] = set()
    if include_backlinks and links:
        try:
            backlinks = client.fetch_backlinks(resolved_seed_title, max_backlinks=max_backlinks)
        except RuntimeError as exc:  # pragma: no cover - runtime/network path
            errors.append(f"backlink_fetch_failed: {exc}")

    expansion_modes = {"one_hop_only", "one_hop_plus_bounded_two_hop"}
    if expansion not in expansion_modes:
        errors.append(f"unsupported_expansion:{expansion}")
        expansion = "one_hop_only"

    candidate_map: dict[str, dict] = {}
    for candidate in links:
        title = candidate["title"]
        candidate_map[title] = {
            "title": title,
            "depth": 1,
            "links_back_to_seed": title in backlinks if include_backlinks else None,
            "status": "unscored",
        }

    two_hop_parents: list[str] = []
    two_hop_added = 0
    if expansion == "one_hop_plus_bounded_two_hop" and links:
        try:
            seed_extract = client.fetch_page_extract(resolved_seed_title, intro_only=True).get(
                "extract", ""
            )
        except RuntimeError as exc:  # pragma: no cover - runtime/network path
            seed_extract = ""
            errors.append(f"two_hop_seed_extract_failed:{exc}")

        parent_titles = [candidate["title"] for candidate in links]
        parent_extracts: list[str] = []
        if seed_extract:
            _prefetch(errors, client.prefetch_intro_extracts, parent_titles)
            extract_payloads, extract_errors = client.fetch_page_extracts_concurrent(
                parent_titles,
                intro_only=True,
            )
            for title in parent_titles:
                payload = extract_payloads.get(title, {})
                parent_extracts.append(payload.get("extract", ""))
            for title, exc_text in extract_errors.items():
                errors.append(f"two_hop_parent_extract_failed:{title}:{exc_text}")

            vectors = build_tfidf_vectors([seed_extract] + parent_extracts, lang=lang)
            seed_vec = vectors[0]
            rel_scores = [cosine_similarity(seed_vec, vec) for vec in vectors[1:]]
            ranked = sorted(
                zip(parent_titles, rel_scores), key=lambda item: item[1], reverse=True
            )
            limit = max_two_hop_parents or len(ranked)
            two_hop_parents = [title for title, _ in ranked[:limit]]

        parent_link_payloads: dict[str, dict] = {}
        parent_link_errors: dict[str, str] = {}
        if two_hop_parents:
            parent_link_payloads, parent_link_errors = client.fetch_links_concurrent(
                two_hop_parents,
                max_links=max_two_hop_links,
            )
            for title, exc_text in parent_link_errors.items():
                errors.append(f"two_hop_links_failed:{title}:{exc_text}")

        for parent_title in two_hop_parents:
            payload = parent_link_payloads.get(parent_title, {"links": []})
            for link in payload.get("links", []):
                title = link.get("title")
                if not title or title in candidate_map:
                    continue
                candidate_map[title] = {
                    "title": title,
                    "depth": 2,
                    "links_back_to_seed": title in backlinks if include_backlinks else None,
                    "status": "unscored",
                }
                two_hop_added += 1
                if max_candidates is not None and len(candidate_map) >= max_candidates:
                    break
            if max_candidates is not None and len(candidate_map) >= max_candidates:
                break

    candidates = list(candidate_map.values())
    if max_candidates is not None and len(candidates) > max_candidates:
        candidates = candidates[:max_candidates]

    diagnostics = build_seed_stage_diagnostics(
        lang=lang,
        seed_requested=seed_title,
        seed_resolved=resolved_seed_title,
        seed_page_id=resolved_seed_page_id,
        candidates=candidates,
        cache_stats=cache.stats(),
        include_backlinks=include_backlinks,
        errors=errors,
    )
    diagnostics["expansion"] = {
        "mode": expansion,
        "max_two_hop_parents": max_two_hop_parents,
        "max_two_hop_links": max_two_hop_links,
        "max_candidates": max_candidates,
        "two_hop_parents_used": len(two_hop_parents),
        "two_hop_candidates_added": two_hop_added,
    }
    diagnostics["counts"]["depth1_count"] = len([c for c in candidates if c.get("depth") == 1])
    diagnostics["counts"]["depth2_count"] = len([c for c in candidates if c.get("depth") == 2])
    final_path = write_json(diagnostics_path, diagnostics)
    return SeedStageResult(diagnostics_path=final_path, diagnostics=diagnostics)
