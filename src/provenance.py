from __future__ import annotations

from urllib.parse import quote


def build_oldid_url(title: str, revid: int | None, *, lang: str = "en") -> str:
    if not title or not revid:
        return ""
    safe_title = quote(title.replace(" ", "_"))
    return f"https://{lang}.wikipedia.org/w/index.php?title={safe_title}&oldid={revid}"


def clue_provenance_missing_fields(clue: dict) -> list[str]:
    required_fields = [
        "source_method",
        "source_page",
        "revid",
        "oldid_url",
        "sentence_offset",
    ]
    missing: list[str] = []
    for field in required_fields:
        value = clue.get(field)
        if value is None or value == "":
            missing.append(field)
    return missing


def has_packageable_clue_provenance(clue: dict) -> bool:
    return not clue_provenance_missing_fields(clue)


def validate_clue_provenance(clues: list[dict]) -> list[str]:
    missing: list[str] = []
    for idx, clue in enumerate(clues):
        for field in clue_provenance_missing_fields(clue):
            missing.append(f"{idx}:{field}")
    return missing
