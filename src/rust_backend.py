from __future__ import annotations

from importlib import import_module
from types import ModuleType


def load_rust_csp() -> tuple[ModuleType | None, str | None]:
    try:
        module = import_module("rust_csp")
    except Exception as exc:
        return None, f"rust_solver_import_failed:{exc}"

    solver = getattr(module, "solve_crossword", None)
    if not callable(solver):
        return None, "rust_solver_missing_entrypoint"

    return module, None
