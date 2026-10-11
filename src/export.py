from __future__ import annotations

import json
from html import escape

_BLANKS = ("#", ".")
_LICENSE = "https://creativecommons.org/licenses/by-sa/4.0/"
_STYLE = """
body{font-family:Georgia,serif;margin:1.5cm;color:#000}
h1{font-size:20pt;margin:0 0 .5cm}h2{font-size:13pt;margin:.6cm 0 .2cm}
table.grid{border-collapse:collapse;margin:0 auto}
.grid td{width:.9cm;height:.9cm;border:1px solid #000;padding:0;position:relative;text-align:center;
font:bold 12pt sans-serif;vertical-align:middle}
.grid td.b{border:0;background:#ddd;-webkit-print-color-adjust:exact;print-color-adjust:exact}
.grid sup{position:absolute;top:1px;left:2px;font:7pt sans-serif}
.clues{display:grid;grid-template-columns:1fr 1fr;gap:1cm;font-size:10.5pt}
.clues ol{margin:0;padding:0;list-style:none}.clues li{margin:0 0 .15cm;break-inside:avoid}
.sources{font-size:8.5pt;margin-top:.6cm}.sources ul{columns:2;margin:.1cm 0;padding-left:.5cm}
.solution{break-before:page}
@page{margin:1.2cm}
@media print{body{margin:0}}
"""


def _grid(puzzle: dict) -> tuple[list[list[str]], dict[tuple[int, int], int]]:
    """Letters cropped to their bounding box (blanks as '') and start-cell numbers in cropped coordinates."""
    cells = puzzle["grid_cells"]
    letters = [(r, c) for r, row in enumerate(cells) for c, ch in enumerate(row) if ch not in _BLANKS]
    r0, r1 = min(r for r, _ in letters), max(r for r, _ in letters)
    c0, c1 = min(c for _, c in letters), max(c for _, c in letters)
    grid = [["" if ch in _BLANKS else ch for ch in row[c0 : c1 + 1]] for row in cells[r0 : r1 + 1]]
    entries = puzzle.get("across_entries", []) + puzzle.get("down_entries", [])
    return grid, {(e["row"] - r0, e["col"] - c0): e["number"] for e in entries}


def _clues(puzzle: dict, key: str) -> list[dict]:
    return sorted(puzzle.get(key, []), key=lambda e: e["number"])


def _enumeration(entry: dict) -> str:
    return str(entry.get("enumeration") or entry.get("length") or len(entry["answer"]))


def _sources(puzzle: dict) -> dict[str, str]:
    """Wikipedia page title -> permanent (oldid) URL, one per page."""
    return {a["source_page"]: a["oldid_url"] for a in puzzle.get("attribution", []) if a.get("source_page")}


def _text(value: str) -> str:
    """ipuz display strings are HTML, and this text comes from Wikipedia: it must show as written."""
    return escape(value, quote=False)


def export_ipuz(puzzle: dict) -> str:
    # Blocks are "#" (not null): every solving app understands them, null cells are patchy.
    # Cropped to the letters' bounding box so a 15x15 grid does not carry empty margins.
    grid, numbers = _grid(puzzle)
    sources = _sources(puzzle)
    ipuz = {
        "version": "http://ipuz.org/v2",
        "kind": ["http://ipuz.org/crossword#1"],
        "title": _text(str(puzzle.get("seed_title", "Crossword"))),
        "copyright": f"Clues adapted from Wikipedia, licensed CC BY-SA 4.0 ({_LICENSE})",
        "notes": "Sources: " + "; ".join(f"{_text(page)} ({_text(url)})" for page, url in sources.items()),
        "dimensions": {"width": len(grid[0]), "height": len(grid)},
        "block": "#",
        "empty": 0,
        "puzzle": [
            [(numbers.get((r, c), 0) if ch else "#") for c, ch in enumerate(row)]
            for r, row in enumerate(grid)
        ],
        "solution": [[ch or "#" for ch in row] for row in grid],
        "clues": {
            name: [
                {"number": e["number"], "clue": _text(e.get("clue", "")), "enumeration": _enumeration(e)}
                for e in _clues(puzzle, key)
            ]
            for name, key in (("Across", "across_entries"), ("Down", "down_entries"))
        },
    }
    return json.dumps(ipuz, ensure_ascii=False, indent=2) + "\n"


def export_html(puzzle: dict, solution: bool = False) -> str:
    grid, numbers = _grid(puzzle)
    title = escape(str(puzzle.get("seed_title", "Crossword")))

    def table(show_letters: bool) -> str:
        rows = []
        for r, row in enumerate(grid):
            tds = []
            for c, ch in enumerate(row):
                if not ch:
                    tds.append('<td class="b"></td>')
                    continue
                num = f"<sup>{numbers[r, c]}</sup>" if (r, c) in numbers else ""
                tds.append(f"<td>{num}{escape(ch) if show_letters else ''}</td>")
            rows.append("<tr>" + "".join(tds) + "</tr>")
        return '<table class="grid">' + "".join(rows) + "</table>"

    def clue_list(name: str, key: str) -> str:
        items = "".join(
            f"<li><b>{e['number']}.</b> {escape(e.get('clue', ''))} ({escape(_enumeration(e))})</li>"
            for e in _clues(puzzle, key)
        )
        return f"<section><h2>{name}</h2><ol>{items}</ol></section>"

    sources = "".join(
        f'<li><a href="{escape(url, quote=True)}">{escape(page)}</a></li>' for page, url in _sources(puzzle).items()
    )
    parts = [
        "<!DOCTYPE html>",
        f'<html lang="{escape(str(puzzle.get("lang", "en")), quote=True)}"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{title}</title><style>{_STYLE}</style></head><body>",
        f"<h1>{title}</h1>",
        table(False),
        '<div class="clues">' + clue_list("Across", "across_entries") + clue_list("Down", "down_entries") + "</div>",
        '<div class="sources"><b>Sources</b>: clues adapted from Wikipedia articles, licensed '
        f'<a href="{_LICENSE}">CC BY-SA 4.0</a>.<ul>{sources}</ul></div>',
    ]
    if solution:
        parts.append(f'<section class="solution"><h2>Solution: {title}</h2>{table(True)}</section>')
    parts.append("</body></html>")
    return "\n".join(parts) + "\n"
