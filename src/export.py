from __future__ import annotations

import base64
import hashlib
import json
import re
from html import escape
from pathlib import Path

_BLANKS = ("#", ".")
_LICENSE = "https://creativecommons.org/licenses/by-sa/4.0/"
_SOLVABLE_PAGE = Path(__file__).with_name("solvable_page.html")
_SOLVABLE_LABELS = {
    "en": {
        "across": "Across",
        "down": "Down",
        "check": "Check",
        "reveal": "Reveal",
        "clear": "Clear",
        "letter": "Letter",
        "word": "Word",
        "grid": "Grid",
        "pause": "Pause",
        "resume": "Resume",
        "time": "Time",
        "paused": "Paused. The grid and clues are hidden.",
        "sources": "Sources (the page titles may give away answers)",
        "attribution": "Clues are adapted from Wikipedia articles, licensed {license}.",
        "cell": "Row {row}, column {column}",
        "position": "letter {position} of {length}",
        "done": "Complete and correct in {time}. Check: {check}. Reveal: {reveal}.",
        "used": "used",
        "unused": "not used",
        "ask_reveal": "Reveal the whole grid?",
        "ask_clear": "Clear the whole grid? Your progress will be lost.",
        "help": "Arrow keys move, Space turns, Tab or Enter goes to the next clue, Escape leaves the grid. "
        "Nothing is checked until you ask.",
    },
    "el": {
        "across": "Οριζόντια",
        "down": "Κάθετα",
        "check": "Έλεγχος",
        "reveal": "Αποκάλυψη",
        "clear": "Καθαρισμός",
        "letter": "Γράμμα",
        "word": "Λέξη",
        "grid": "Πλέγμα",
        "pause": "Παύση",
        "resume": "Συνέχεια",
        "time": "Χρόνος",
        "paused": "Σε παύση. Το πλέγμα και οι ορισμοί είναι κρυμμένα.",
        "sources": "Πηγές (οι τίτλοι των άρθρων μπορεί να αποκαλύπτουν απαντήσεις)",
        "attribution": "Οι ορισμοί προσαρμόζονται από άρθρα της Βικιπαίδειας, με άδεια {license}.",
        "cell": "Γραμμή {row}, στήλη {column}",
        "position": "γράμμα {position} από {length}",
        "done": "Ολοκληρώθηκε σωστά σε {time}. Έλεγχος: {check}. Αποκάλυψη: {reveal}.",
        "used": "χρησιμοποιήθηκε",
        "unused": "δεν χρησιμοποιήθηκε",
        "ask_reveal": "Να αποκαλυφθεί όλο το πλέγμα;",
        "ask_clear": "Να καθαριστεί όλο το πλέγμα; Η πρόοδός σας θα χαθεί.",
        "help": "Τα βέλη μετακινούν, το Space αλλάζει κατεύθυνση, το Tab ή το Enter πάει στον επόμενο ορισμό, "
        "το Escape βγάζει από το πλέγμα. Τίποτα δεν ελέγχεται αν δεν το ζητήσετε.",
    },
}
# The labels the page script uses; the others are written into the HTML here.
_SOLVABLE_SCRIPT_LABELS = ("across", "down", "pause", "resume", "paused", "done", "used", "unused", "ask_reveal", "ask_clear")
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


def _solvable_words(puzzle: dict, grid: list[list[str]], numbers: dict[tuple[int, int], int]) -> list[dict]:
    """Across words, then down words, in clue-list order, each with its cells as row * width + column indices."""
    width = len(grid[0])
    starts = {number: cell for cell, number in numbers.items()}
    words = []
    for direction, key in enumerate(("across_entries", "down_entries")):
        for entry in _clues(puzzle, key):
            row, column = starts[entry["number"]]
            cells = [(row + i * direction) * width + column + i * (1 - direction) for i in range(len(entry["answer"]))]
            if "".join(grid[cell // width][cell % width] for cell in cells) != entry["answer"]:
                raise ValueError(f"entry {entry['number']} ({entry['answer']}) does not match the grid")
            words.append({"direction": direction, "number": entry["number"], "cells": cells, "entry": entry})
    return words


def export_solvable_html(puzzle: dict) -> str:
    """One self-contained page to solve the puzzle in a browser: src/solvable_page.html filled with this puzzle.

    The page reads the solution from base64 in its data block (a guard against spoilers in the page source, not
    against anyone who decodes it) and keeps progress in localStorage under a key made of the title and a hash of
    the solution.
    """
    grid, numbers = _grid(puzzle)
    width = len(grid[0])
    lang = str(puzzle.get("lang", "en"))
    labels = _SOLVABLE_LABELS.get(lang, _SOLVABLE_LABELS["en"])
    title = escape(str(puzzle.get("seed_title", "Crossword")))
    words = _solvable_words(puzzle, grid, numbers)

    # The accessible name of a cell: where it is and each word through it, e.g. "1 across (letter 1 of 3)".
    through: dict[int, list[str]] = {}
    for word in words:
        name = labels["down" if word["direction"] else "across"].lower()
        for position, cell in enumerate(word["cells"], 1):
            letter = labels["position"].format(position=position, length=len(word["cells"]))
            through.setdefault(cell, []).append(f"{word['number']} {name} ({letter})")

    cells = []
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            if not ch:
                cells.append("<div></div>")  # blanks stay empty space: the grid is free-form
                continue
            num = f'<i class="number" aria-hidden="true">{numbers[r, c]}</i>' if (r, c) in numbers else ""
            name = escape(labels["cell"].format(row=r + 1, column=c + 1) + ", " + ", ".join(through[r * width + c]))
            cells.append(
                f'<div class="cell">{num}<input aria-label="{name}" tabindex="-1" autocomplete="off" '
                'autocorrect="off" autocapitalize="characters" spellcheck="false" enterkeyhint="next"></div>'
            )

    def clue_list(direction: int) -> str:
        items = "".join(
            f'<li><button type="button" class="clue" data-word="{index}"><b class="clue-number">{word["number"]}</b>'
            f'<span><span class="clue-text">{_text(word["entry"].get("clue", ""))}</span> '
            f'<span class="enumeration">({_text(_enumeration(word["entry"]))})</span></span></button></li>'
            for index, word in enumerate(words)
            if word["direction"] == direction
        )
        heading = labels["down" if direction else "across"]
        return f'<section><h2>{heading}</h2><ol class="clue-list">{items}</ol></section>'

    def tool(action: str, scopes: tuple[str, ...]) -> str:
        buttons = "".join(
            f'<button type="button" data-action="{action}" data-scope="{scope}">{labels[scope]}</button>'
            for scope in scopes
        )
        return f'<span class="group" role="group" aria-label="{labels[action]}"><b>{labels[action]}</b>{buttons}</span>'

    solution = ["".join(ch or "." for ch in row) for row in grid]
    data = {
        "columns": width,
        "rows": len(grid),
        "words": [{key: word[key] for key in ("direction", "number", "cells")} for word in words],
        "solution": base64.b64encode(json.dumps(solution).encode("ascii")).decode("ascii"),
        # Saved progress is found under this key: changing how it is made orphans progress saved by earlier pages.
        "storageKey": f"crossword:{puzzle.get('seed_title', '')}:"
        + hashlib.sha256("".join(solution).encode("utf-8")).hexdigest()[:12],
        "labels": {name: labels[name] for name in _SOLVABLE_SCRIPT_LABELS},
    }
    sources = "".join(f'<li><a href="{escape(url)}">{_text(page)}</a></li>' for page, url in _sources(puzzle).items())
    values = {
        "LANG": escape(lang),
        "TITLE": title,
        "COLUMNS": str(width),
        "TIMER": f'<span id="time" role="timer" aria-label="{labels["time"]}">0:00</span>'
        f'<button type="button" id="pause">{labels["pause"]}</button>',
        "TOOLS": tool("check", ("letter", "word", "grid"))
        + tool("reveal", ("letter", "word", "grid"))
        + tool("clear", ("word", "grid")),
        "HELP": labels["help"],
        "CELLS": "".join(cells),
        "CLUES": clue_list(0) + clue_list(1),
        "ATTRIBUTION": labels["attribution"].format(license=f'<a href="{_LICENSE}">CC BY-SA 4.0</a>'),
        # Closed: the page titles give away answers.
        "SOURCES": f'<details id="sources"><summary>{labels["sources"]}</summary><ul>{sources}</ul></details>',
        # "<" is escaped so the data cannot close its script block (or open a comment); json.dumps already
        # writes U+2028 and U+2029 as escapes.
        "DATA": json.dumps(data).replace("<", "\\u003c"),
    }
    # One pass: text inserted for one placeholder is never searched for another.
    return re.sub(r"@([A-Z]+)@", lambda match: values[match.group(1)], _SOLVABLE_PAGE.read_text(encoding="utf-8"))
