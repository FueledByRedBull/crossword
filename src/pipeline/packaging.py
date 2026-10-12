from __future__ import annotations

from pathlib import Path

from ..artifact_models import GridArtifact, PuzzleArtifact, SelectedCandidatesArtifact
from ..diagnostics import write_json
from ..layout import Placement, build_layout, layout_metrics, number_placements, validate_layout
from ..text_normalize import enumeration
from ..provenance import validate_clue_provenance

from .common import (
    LayoutStageResult,
    PackagingStageResult,
    _safe_float,
    _is_packageable_clue,
    _clued_answer_count,
    _answer_key,
    _read_csv_rows,
    _shared_words,
)


# People may fill at most this share of the layout candidates, unless too few other answers are clued: short
# surnames place easily, and a grid of 23 painters out of 26 entries is a list, not a themed puzzle. With a
# third of the candidates they still make up to 60% of a grid; with half, 70%.
_MAX_PEOPLE_SHARE = 1 / 3


def _layout_candidates(
    term_rows: list[dict],
    clue_rows: list[dict],
    *,
    lang: str,
    limit: int,
) -> list[tuple[str, float]]:
    """Packageable answers by descending answer_score, skipping any that repeats a word already taken.

    People take at most `_MAX_PEOPLE_SHARE` of the `limit`; when the other answers run out, they fill what is left.
    """
    clued = {
        _answer_key(row.get("answer", ""), row.get("normalized_answer", ""), lang=lang)
        for row in clue_rows
        if _is_packageable_clue(row)
    }
    ranked = sorted(term_rows, key=lambda row: -_safe_float(row.get("answer_score")))

    def pick(max_people: int) -> tuple[list[tuple[str, float]], int]:
        picked: list[tuple[str, float]] = []
        taken: set[str] = set()
        words: set[str] = set()
        people = 0
        for row in ranked:
            key = _answer_key(row.get("answer", ""), row.get("normalized_answer", ""), lang=lang)
            person = row.get("page_is_person") == "True"
            if key not in clued or key in taken or (person and people >= max_people):
                continue
            row_words = _shared_words(row.get("answer", ""))
            if row_words & words:
                continue
            taken.add(key)
            words |= row_words
            people += person
            picked.append((key, _safe_float(row.get("answer_score"))))
            if len(picked) >= limit:
                break
        return picked, people

    picked, people = pick(int(limit * _MAX_PEOPLE_SHARE))
    return picked if len(picked) >= limit else pick(limit - (len(picked) - people))[0]


def run_layout_stage(
    *,
    seed_title: str,
    lang: str = "en",
    terms_path: str | Path = "outputs/answer_candidates.csv",
    clues_path: str | Path = "outputs/clues.csv",
    grid_path: str | Path = "outputs/grid.json",
    diagnostics_path: str | Path = "outputs/diagnostics_layout.json",
    size: int = 15,
    random_seed: int = 13,
    restarts: int = 96,
    candidates: int = 80,
    min_words: int = 12,
    gate_min: int = 30,
) -> LayoutStageResult:
    errors: list[str] = []
    term_rows = _read_csv_rows(terms_path)
    clue_rows = _read_csv_rows(clues_path)
    if _clued_answer_count(clue_rows, lang=lang) < gate_min:
        errors.append("vocab_gate_failed")

    picked = [] if errors else _layout_candidates(term_rows, clue_rows, lang=lang, limit=candidates)
    layout = build_layout(picked, size=size, random_seed=random_seed, restarts=restarts)
    problems = validate_layout(layout.grid, layout.placements)
    complete = not problems and layout.metrics["word_count"] >= min_words
    fill_status = "complete" if complete else "failed"

    grid_model = GridArtifact(
        seed_title=seed_title,
        lang=lang,
        size=size,
        grid=layout.grid,
        placements=[
            {"answer": p.answer, "row": p.row, "col": p.col, "direction": p.direction}
            for p in layout.placements
        ],
        fill_status=fill_status,
    )
    final_grid = write_json(grid_path, grid_model.to_dict())
    diagnostics = {
        "stage": "layout",
        "lang": lang,
        "seed_title": seed_title,
        "size": size,
        "random_seed": random_seed,
        "restarts": restarts,
        "min_words": min_words,
        "candidate_count": len(picked),
        "fill_status": fill_status,
        "validity_errors": problems,
        "errors": errors,
        **layout.metrics,
    }
    final_diagnostics = write_json(diagnostics_path, diagnostics)
    return LayoutStageResult(
        grid_path=final_grid,
        diagnostics_path=final_diagnostics,
        diagnostics=diagnostics,
    )


def run_packaging_stage(
    *,
    seed_title: str,
    lang: str = "en",
    selected_path: str | Path = "outputs/selected_candidates.json",
    grid_path: str | Path = "outputs/grid.json",
    clues_path: str | Path = "outputs/clues.csv",
    puzzle_path: str | Path = "outputs/puzzle.json",
    attribution_path: str | Path = "outputs/attribution.json",
    diagnostics_path: str | Path = "outputs/diagnostics_package.json",
    gate_min: int = 30,
) -> PackagingStageResult:
    import json

    errors: list[str] = []

    selected = SelectedCandidatesArtifact(seed_title=seed_title, lang=lang, selected_k=0, selected_titles=[])
    if Path(selected_path).exists():
        try:
            selected = SelectedCandidatesArtifact.from_dict(
                json.loads(Path(selected_path).read_text(encoding="utf-8"))
            )
        except (OSError, ValueError) as exc:  # pragma: no cover
            errors.append(f"selected_parse_failed:{exc}")
    else:
        errors.append("selected_missing")

    grid = GridArtifact(seed_title=seed_title, lang=lang, size=15, grid=[], placements=[])
    if Path(grid_path).exists():
        try:
            grid = GridArtifact.from_dict(json.loads(Path(grid_path).read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:  # pragma: no cover
            errors.append(f"grid_parse_failed:{exc}")
    else:
        errors.append("grid_missing")

    clues = _read_csv_rows(clues_path)
    if not Path(clues_path).exists():
        errors.append("clues_missing")
    clue_by_key: dict[str, dict] = {}
    for clue in clues:
        if _is_packageable_clue(clue):
            key = _answer_key(clue.get("answer", ""), clue.get("normalized_answer", ""), lang=lang)
            clue_by_key.setdefault(key, clue)

    placements = [Placement(p["answer"], p["row"], p["col"], p["direction"]) for p in grid.placements]
    numbers = number_placements(placements)
    entries: dict[str, list[dict]] = {"across": [], "down": []}
    used_clues: list[dict] = []
    unclued: list[str] = []
    for placement in sorted(placements, key=lambda p: numbers[(p.row, p.col)]):
        clue = clue_by_key.get(placement.answer)
        if clue is None:
            unclued.append(placement.answer)
            continue
        display = clue.get("answer") or placement.answer
        entries[placement.direction].append(
            {
                "number": numbers[(placement.row, placement.col)],
                "answer": placement.answer,
                "display_answer": display,
                "enumeration": enumeration(display),
                "clue": clue["clue"],
                "clue_style": clue.get("clue_style", ""),
                "clue_class": "source_backed",
                "row": placement.row,
                "col": placement.col,
                "length": len(placement.answer),
            }
        )
        used_clues.append(clue)

    provenance_missing = validate_clue_provenance(used_clues)
    if provenance_missing:
        errors.append("provenance_incomplete")
    if unclued:
        errors.append("clue_coverage_failed")
    if _clued_answer_count(clues, lang=lang) < gate_min:
        puzzle_status = "insufficient_vocabulary"
    elif unclued:
        puzzle_status = "insufficient_clues"
    elif grid.fill_status != "complete":
        puzzle_status = "insufficient_quality"
    else:
        puzzle_status = "ok"

    placed = [entry for rows in entries.values() for entry in rows]
    clue_texts = [entry["clue"].lower() for entry in placed]
    definition_count = sum(entry["clue_style"] == "definition" for entry in placed)
    diagnostics = {
        "stage": "packaging",
        "lang": lang,
        "seed_title": seed_title,
        "selected_k": selected.selected_k,
        "clue_count": len(clues),
        "entry_count": len(placed),
        "fill_status": grid.fill_status,
        "puzzle_status": puzzle_status,
        "definition_clue_count": definition_count,
        "fill_in_blank_clue_count": len(placed) - definition_count,
        "duplicate_clue_text_count": len(clue_texts) - len(set(clue_texts)),
        "used_clue_provenance_missing_count": len(provenance_missing),
        "errors": errors,
    }
    puzzle_model = PuzzleArtifact(
        seed_title=seed_title,
        lang=lang,
        selected_articles=selected.selected_titles,
        selected_k=selected.selected_k,
        size=grid.size,
        grid_cells=grid.grid,
        across_entries=entries["across"],
        down_entries=entries["down"],
        fill_status=grid.fill_status,
        puzzle_status=puzzle_status,
        layout=layout_metrics(grid.grid, placements),
        diagnostics=diagnostics,
        attribution=used_clues,
    )
    final_puzzle = write_json(puzzle_path, puzzle_model.to_dict())
    final_attribution = write_json(attribution_path, {"clues": used_clues})
    final_diagnostics = write_json(diagnostics_path, diagnostics)
    return PackagingStageResult(
        puzzle_path=final_puzzle,
        attribution_path=final_attribution,
        diagnostics_path=final_diagnostics,
        diagnostics=diagnostics,
    )
