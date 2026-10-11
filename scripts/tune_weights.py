from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _parse_weight_list(value: str, size: int) -> list[tuple[float, ...]]:
    groups = []
    for group in value.split(";"):
        group = group.strip()
        if not group:
            continue
        parts = [float(item.strip()) for item in group.split(",") if item.strip()]
        if len(parts) != size:
            raise ValueError(f"Expected {size} values per group, got {len(parts)} in '{group}'")
        groups.append(tuple(parts))
    if not groups:
        raise ValueError("No weight groups provided")
    return groups


def main() -> None:
    from src.pipeline import run_generate_pipeline

    parser = argparse.ArgumentParser(description="Grid-search candidate and k-selection weights.")
    parser.add_argument("--seeds", default="Thermodynamics,Jazz")
    parser.add_argument("--output-dir", default="outputs/weight_tuning")
    parser.add_argument("--cache-dir", default="data/cache/wiki")
    parser.add_argument("--wikidata-cache-dir", default="data/cache/wikidata")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--candidate-weights",
        default="1.5,0.3,0.3,0.2",
        help="Semicolon-separated groups a,b,c,d",
    )
    parser.add_argument(
        "--k-weights",
        default="1.0,1.0,1.0",
        help="Semicolon-separated groups w1,w2,w3",
    )
    args = parser.parse_args()

    if args.offline:
        os.environ["CROSSWORD_OFFLINE"] = "1"

    seeds = [seed.strip() for seed in args.seeds.split(",") if seed.strip()]
    candidate_groups = _parse_weight_list(args.candidate_weights, 4)
    k_groups = _parse_weight_list(args.k_weights, 3)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_lines = [
        "candidate_weights,k_weights,seed,fill_status,word_count,crossing_count,output_dir",
    ]

    for c_idx, c_weights in enumerate(candidate_groups):
        candidate_weights = {"a": c_weights[0], "b": c_weights[1], "c": c_weights[2], "d": c_weights[3]}
        for k_idx, k_weights_tuple in enumerate(k_groups):
            k_weights = {
                "w1": k_weights_tuple[0],
                "w2": k_weights_tuple[1],
                "w3": k_weights_tuple[2],
            }
            for seed in seeds:
                combo_dir = output_dir / f"cand_{c_idx}" / f"k_{k_idx}" / seed.replace(" ", "_")
                result = run_generate_pipeline(
                    seed_title=seed,
                    lang="en",
                    output_dir=combo_dir,
                    cache_dir=args.cache_dir,
                    wikidata_cache_dir=args.wikidata_cache_dir,
                    candidate_weights=candidate_weights,
                    k_weights=k_weights,
                )
                layout = result["layout"].diagnostics
                fill_status = layout.get("fill_status", "unknown")
                word_count = layout.get("word_count", 0)
                crossing_count = layout.get("crossing_count", 0)
                summary_lines.append(
                    f"\"{candidate_weights}\",\"{k_weights}\",{seed},{fill_status},{word_count},{crossing_count},{combo_dir}"
                )

    summary_path = output_dir / "summary.csv"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
