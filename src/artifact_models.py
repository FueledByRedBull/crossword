from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _as_str(value: Any, *, field_name: str) -> str:
    if isinstance(value, str):
        return value
    raise ValueError(f"invalid_{field_name}:expected_str")


def _as_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"invalid_{field_name}:expected_int")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(f"invalid_{field_name}:expected_int") from exc
    raise ValueError(f"invalid_{field_name}:expected_int")


@dataclass(slots=True)
class SelectedCandidatesArtifact:
    seed_title: str
    lang: str
    selected_k: int
    selected_titles: list[str]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SelectedCandidatesArtifact":
        selected_titles_raw = payload.get("selected_titles", [])
        if not isinstance(selected_titles_raw, list):
            raise ValueError("invalid_selected_titles:expected_list")
        selected_titles = [_as_str(title, field_name="selected_title") for title in selected_titles_raw]
        return cls(
            seed_title=_as_str(payload.get("seed_title", ""), field_name="seed_title"),
            lang=_as_str(payload.get("lang", ""), field_name="lang"),
            selected_k=_as_int(payload.get("selected_k", len(selected_titles)), field_name="selected_k"),
            selected_titles=selected_titles,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed_title": self.seed_title,
            "lang": self.lang,
            "selected_k": self.selected_k,
            "selected_titles": self.selected_titles,
        }


@dataclass(slots=True)
class GridArtifact:
    seed_title: str
    lang: str
    size: int
    grid: list[list[str]]
    placements: list[dict[str, Any]]
    fill_status: str = "failed"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GridArtifact":
        placements_raw = payload.get("placements", [])
        if not isinstance(placements_raw, list):
            raise ValueError("invalid_placements:expected_list")
        placements = [
            {
                "answer": _as_str(row.get("answer"), field_name="placement_answer"),
                "row": _as_int(row.get("row"), field_name="placement_row"),
                "col": _as_int(row.get("col"), field_name="placement_col"),
                "direction": _as_str(row.get("direction"), field_name="placement_direction"),
            }
            for row in placements_raw
        ]
        return cls(
            seed_title=_as_str(payload.get("seed_title", ""), field_name="seed_title"),
            lang=_as_str(payload.get("lang", ""), field_name="lang"),
            size=_as_int(payload.get("size", 15), field_name="size"),
            grid=payload.get("grid", []),
            placements=placements,
            fill_status=_as_str(payload.get("fill_status", "failed"), field_name="fill_status"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed_title": self.seed_title,
            "lang": self.lang,
            "size": self.size,
            "grid": self.grid,
            "placements": self.placements,
            "fill_status": self.fill_status,
        }


@dataclass(slots=True)
class PuzzleArtifact:
    seed_title: str
    lang: str
    selected_articles: list[str]
    selected_k: int
    size: int
    grid_cells: list[list[str]]
    across_entries: list[dict[str, Any]]
    down_entries: list[dict[str, Any]]
    fill_status: str
    puzzle_status: str
    layout: dict[str, Any]
    diagnostics: dict[str, Any]
    attribution: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed_title": self.seed_title,
            "lang": self.lang,
            "selected_articles": self.selected_articles,
            "selected_k": self.selected_k,
            "size": self.size,
            "grid_cells": self.grid_cells,
            "across_entries": self.across_entries,
            "down_entries": self.down_entries,
            "fill_status": self.fill_status,
            "puzzle_status": self.puzzle_status,
            "layout": self.layout,
            "diagnostics": self.diagnostics,
            "attribution": self.attribution,
        }
