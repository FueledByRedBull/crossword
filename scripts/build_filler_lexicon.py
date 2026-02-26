from __future__ import annotations

import argparse
from pathlib import Path


def normalize_token(token: str) -> str:
    return "".join(ch for ch in token.upper() if ch.isalpha())


def parse_hunspell_dic(
    dic_path: Path,
    *,
    min_len: int,
    max_len: int,
    max_per_length: int | None,
) -> list[str]:
    if not dic_path.exists():
        raise FileNotFoundError(dic_path)
    buckets: dict[int, set[str]] = {}
    with dic_path.open("r", encoding="utf-8", errors="ignore") as handle:
        first_line = True
        for line in handle:
            text = line.strip()
            if not text:
                continue
            if first_line and text.isdigit():
                first_line = False
                continue
            first_line = False
            base = text.split()[0]
            if not base:
                continue
            base = base.split("/", 1)[0]
            normalized = normalize_token(base)
            if not normalized:
                continue
            length = len(normalized)
            if length < min_len or length > max_len:
                continue
            bucket = buckets.setdefault(length, set())
            bucket.add(normalized)

    words: list[str] = []
    for length in sorted(buckets.keys()):
        candidates = sorted(buckets[length])
        if max_per_length is not None:
            candidates = candidates[:max_per_length]
        words.extend(candidates)
    return words


def main() -> None:
    parser = argparse.ArgumentParser(description="Build filler word list from Hunspell dic file.")
    parser.add_argument("--dic", required=True, help="Path to Hunspell .dic file")
    parser.add_argument("--output", required=True, help="Output filler word list")
    parser.add_argument("--min-len", type=int, default=3)
    parser.add_argument("--max-len", type=int, default=12)
    parser.add_argument("--max-per-length", type=int, default=4000)
    args = parser.parse_args()

    dic_path = Path(args.dic)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    words = parse_hunspell_dic(
        dic_path,
        min_len=args.min_len,
        max_len=args.max_len,
        max_per_length=args.max_per_length,
    )
    header = [
        "# Filler word list generated from Hunspell",
        f"# source: {dic_path.name}",
        f"# min_len={args.min_len} max_len={args.max_len} max_per_length={args.max_per_length}",
        f"# total_words={len(words)}",
    ]
    output_path.write_text("\n".join(header + words) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
