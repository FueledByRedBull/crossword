from __future__ import annotations

import argparse
from pathlib import Path
import re


_SPLIT_RE = re.compile(r"[\s,\t]+")


def normalize_token(token: str) -> str:
    return "".join(ch for ch in token.upper() if ch.isalpha())


def load_scowl(path: Path) -> dict[str, float]:
    entries: dict[str, float] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            parts = [part for part in _SPLIT_RE.split(text) if part]
            if not parts:
                continue
            token = normalize_token(parts[0])
            if not token:
                continue
            score = 1.0
            if len(parts) > 1:
                try:
                    score = float(parts[1])
                except ValueError:
                    score = 1.0
            current = entries.get(token)
            if current is None or score > current:
                entries[token] = score
    return entries


def load_filler(path: Path) -> set[str]:
    words: set[str] = set()
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            token = normalize_token(text.split()[0])
            if token:
                words.add(token)
    return words


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine SCOWL wordfreq with filler list.")
    parser.add_argument("--scowl", default="data/lexicon/scowl_wordfreq.txt")
    parser.add_argument("--filler", default="data/lexicon/filler_words.txt")
    parser.add_argument("--output", default="data/lexicon/combined_wordfreq.txt")
    parser.add_argument("--filler-score", type=float, default=0.1)
    args = parser.parse_args()

    scowl_path = Path(args.scowl)
    filler_path = Path(args.filler)
    output_path = Path(args.output)

    scowl = load_scowl(scowl_path)
    filler = load_filler(filler_path)

    combined = dict(scowl)
    for word in filler:
        if word not in combined:
            combined[word] = float(args.filler_score)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "# Combined lexicon (SCOWL + filler)",
        f"# scowl={scowl_path.name} filler={filler_path.name}",
        f"# filler_score={args.filler_score}",
        f"# total_entries={len(combined)}",
    ]
    lines = header + [f"{word}\t{combined[word]}" for word in sorted(combined.keys())]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
