from __future__ import annotations

from urllib.parse import quote


def build_oldid_url(title: str, revid: int | None, *, lang: str = "en") -> str:
    if not title or not revid:
        return ""
    safe_title = quote(title.replace(" ", "_"))
    return f"https://{lang}.wikipedia.org/w/index.php?title={safe_title}&oldid={revid}"


def infer_clue_class(clue: dict) -> str:
    clue_class = str(clue.get("clue_class") or "").strip().lower()
    if clue_class in {"source_backed", "template_fallback", "synthetic_filler"}:
        return clue_class
    source_method = str(clue.get("source_method") or "").strip().lower()
    if source_method == "package_filler_fallback":
        return "synthetic_filler"
    sentence_offset = clue.get("sentence_offset")
    if sentence_offset not in (None, "", -1, "-1"):
        return "source_backed"
    return "template_fallback"


def clue_provenance_missing_fields(clue: dict) -> list[str]:
    clue_class = infer_clue_class(clue)
    if clue_class == "synthetic_filler":
        return ["synthetic_filler_not_packageable"]
    required_fields = [
        "source_method",
        "source_page",
        "revid",
        "oldid_url",
    ]
    if clue_class == "source_backed":
        required_fields.append("sentence_offset")
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
