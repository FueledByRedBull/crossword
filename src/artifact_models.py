from __future__ import annotations

from dataclasses import dataclass, field
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


def _as_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"invalid_{field_name}:expected_float")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"invalid_{field_name}:expected_float") from exc
    raise ValueError(f"invalid_{field_name}:expected_float")


def _as_cells(value: Any) -> list[tuple[int, int]]:
    if not isinstance(value, list):
        raise ValueError("invalid_cells:expected_list")
    cells: list[tuple[int, int]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("invalid_cells:item_must_be_pair")
        cells.append((_as_int(item[0], field_name="cell_row"), _as_int(item[1], field_name="cell_col")))
    return cells


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
class GridSlotRecord:
    id: int
    direction: str
    length: int
    cells: list[tuple[int, int]]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GridSlotRecord":
        return cls(
            id=_as_int(payload.get("id"), field_name="slot_id"),
            direction=_as_str(payload.get("direction", ""), field_name="slot_direction"),
            length=_as_int(payload.get("length"), field_name="slot_length"),
            cells=_as_cells(payload.get("cells", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "direction": self.direction,
            "length": self.length,
            "cells": self.cells,
        }


@dataclass(slots=True)
class GridArtifact:
    seed_title: str
    lang: str
    template: str | None
    size: int
    min_slot_len: int
    effective_min_slot_len: int
    grid: list[list[str]]
    assignments: dict[int, str]
    slots: list[GridSlotRecord]
    auto_block: dict[str, Any] = field(default_factory=dict)
    filler: dict[str, Any] = field(default_factory=dict)
    invalid_slots: list[dict[str, Any]] = field(default_factory=list)
    fill_status: str = "failed"
    fill_percent: float = 0.0
    unfilled_slots: list[dict[str, Any]] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GridArtifact":
        assignments_raw = payload.get("assignments", {})
        if not isinstance(assignments_raw, dict):
            raise ValueError("invalid_assignments:expected_dict")
        assignments: dict[int, str] = {}
        for key, value in assignments_raw.items():
            slot_id = _as_int(key, field_name="assignment_slot_id")
            assignments[slot_id] = _as_str(value, field_name="assignment_word")

        slots_raw = payload.get("slots", [])
        if not isinstance(slots_raw, list):
            raise ValueError("invalid_slots:expected_list")
        slots = [GridSlotRecord.from_dict(row) for row in slots_raw if isinstance(row, dict)]

        known_fields = {
            "seed_title",
            "lang",
            "template",
            "size",
            "min_slot_len",
            "effective_min_slot_len",
            "grid",
            "assignments",
            "slots",
            "auto_block",
            "filler",
            "invalid_slots",
            "fill_status",
            "fill_percent",
            "unfilled_slots",
        }
        extras = {key: value for key, value in payload.items() if key not in known_fields}

        template_value = payload.get("template")
        if template_value is not None and not isinstance(template_value, str):
            raise ValueError("invalid_template:expected_str_or_none")

        return cls(
            seed_title=_as_str(payload.get("seed_title", ""), field_name="seed_title"),
            lang=_as_str(payload.get("lang", ""), field_name="lang"),
            template=template_value,
            size=_as_int(payload.get("size", 15), field_name="size"),
            min_slot_len=_as_int(payload.get("min_slot_len", 3), field_name="min_slot_len"),
            effective_min_slot_len=_as_int(
                payload.get("effective_min_slot_len", payload.get("min_slot_len", 3)),
                field_name="effective_min_slot_len",
            ),
            grid=payload.get("grid", []),
            assignments=assignments,
            slots=slots,
            auto_block=payload.get("auto_block", {}) if isinstance(payload.get("auto_block", {}), dict) else {},
            filler=payload.get("filler", {}) if isinstance(payload.get("filler", {}), dict) else {},
            invalid_slots=payload.get("invalid_slots", [])
            if isinstance(payload.get("invalid_slots", []), list)
            else [],
            fill_status=_as_str(payload.get("fill_status", "failed"), field_name="fill_status"),
            fill_percent=_as_float(payload.get("fill_percent", 0.0), field_name="fill_percent"),
            unfilled_slots=payload.get("unfilled_slots", [])
            if isinstance(payload.get("unfilled_slots", []), list)
            else [],
            extras=extras,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "seed_title": self.seed_title,
            "lang": self.lang,
            "template": self.template,
            "size": self.size,
            "min_slot_len": self.min_slot_len,
            "effective_min_slot_len": self.effective_min_slot_len,
            "grid": self.grid,
            "assignments": self.assignments,
            "slots": [slot.to_dict() for slot in self.slots],
            "auto_block": self.auto_block,
            "filler": self.filler,
            "invalid_slots": self.invalid_slots,
            "fill_status": self.fill_status,
            "fill_percent": self.fill_percent,
            "unfilled_slots": self.unfilled_slots,
        }
        payload.update(self.extras)
        return payload


@dataclass(slots=True)
class PuzzleArtifact:
    seed_title: str
    lang: str
    selected_articles: list[str]
    selected_k: int
    grid_template_id: str | None
    grid_cells: list[list[str]]
    across_entries: list[dict[str, Any]]
    down_entries: list[dict[str, Any]]
    fill_status: str
    fill_percent: float
    unfilled_slots: list[dict[str, Any]]
    puzzle_status: str
    diagnostics: dict[str, Any]
    attribution: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed_title": self.seed_title,
            "lang": self.lang,
            "selected_articles": self.selected_articles,
            "selected_k": self.selected_k,
            "grid_template_id": self.grid_template_id,
            "grid_cells": self.grid_cells,
            "across_entries": self.across_entries,
            "down_entries": self.down_entries,
            "fill_status": self.fill_status,
            "fill_percent": self.fill_percent,
            "unfilled_slots": self.unfilled_slots,
            "puzzle_status": self.puzzle_status,
            "diagnostics": self.diagnostics,
            "attribution": self.attribution,
        }
