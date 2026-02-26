from __future__ import annotations

from urllib.parse import quote


def build_oldid_url(title: str, revid: int | None, *, lang: str = "en") -> str:
    if not title or not revid:
        return ""
    safe_title = quote(title.replace(" ", "_"))
    return f"https://{lang}.wikipedia.org/w/index.php?title={safe_title}&oldid={revid}"


def validate_clue_provenance(clues: list[dict]) -> list[str]:
    missing: list[str] = []
    required_fields = [
        "source_method",
        "source_page",
        "revid",
        "sentence_offset",
        "oldid_url",
    ]
    for idx, clue in enumerate(clues):
        for field in required_fields:
            value = clue.get(field)
            if value is None or value == "":
                missing.append(f"{idx}:{field}")
    return missing

