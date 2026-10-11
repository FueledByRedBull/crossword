from __future__ import annotations

from .layout import ACROSS, DOWN, Placement, number_placements, validate_puzzle

_BLANKS = ("#", ".")


def render_puzzle_text(puzzle: dict) -> str:
    grid = puzzle["grid_cells"]
    entries = [
        (direction, entry)
        for direction, key in ((ACROSS, "across_entries"), (DOWN, "down_entries"))
        for entry in puzzle.get(key, [])
    ]
    computed = number_placements(
        [Placement(str(e["answer"]), e["row"], e["col"], d) for d, e in entries]
    )
    numbers = {
        (e["row"], e["col"]): e.get("number") or computed[(e["row"], e["col"])]
        for _, e in entries
    }

    letters = [
        (r, c)
        for r, row in enumerate(grid)
        for c, ch in enumerate(row)
        if ch not in _BLANKS
    ]
    rows = range(min((r for r, _ in letters), default=0), max((r for r, _ in letters), default=-1) + 1)
    cols = range(min((c for _, c in letters), default=0), max((c for _, c in letters), default=-1) + 1)
    width = len(str(max(numbers.values(), default=0))) + 1

    def draw(cell) -> str:
        return "\n".join(
            "".join(cell(r, c).rjust(width) for c in cols).rstrip() for r in rows
        )

    def blank_cell(r: int, c: int) -> str:
        if grid[r][c] in _BLANKS:
            return ""
        return str(numbers.get((r, c), "_"))

    def solved_cell(r: int, c: int) -> str:
        return "" if grid[r][c] in _BLANKS else grid[r][c]

    lines = [str(puzzle.get("seed_title", "Crossword")), "", draw(blank_cell)]
    for title, direction in (("Across", ACROSS), ("Down", DOWN)):
        listed = sorted(
            (e for d, e in entries if d == direction),
            key=lambda e: numbers[(e["row"], e["col"])],
        )
        lines += ["", title]
        lines += [
            f"{numbers[(e['row'], e['col'])]}. {e.get('clue', '')} "
            f"({e.get('enumeration') or e.get('length') or len(e['answer'])})"
            for e in listed
        ]
    problems = validate_puzzle(puzzle)
    lines += ["", "Solution", draw(solved_cell), ""]
    lines.append("Validity: OK" if not problems else f"Validity: {len(problems)} problem(s)")
    lines += [f"- {problem}" for problem in problems]
    return "\n".join(lines)
