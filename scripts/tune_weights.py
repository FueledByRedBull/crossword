from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import run_generate_pipeline


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
    parser = argparse.ArgumentParser(description="Grid-search candidate and k-selection weights.")
    parser.add_argument("--seeds", default="Thermodynamics,Jazz")
    parser.add_argument("--output-dir", default="outputs/weight_tuning")
    parser.add_argument("--cache-dir", default="data/cache/wiki")
    parser.add_argument("--wikidata-cache-dir", default="data/cache/wikidata")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--use-rust", action="store_true")
    parser.add_argument(
        "--candidate-weights",
        default="1.5,0.3,0.3,0.2",
        help="Semicolon-separated groups a,b,c,d",
    )
    parser.add_argument(
        "--k-weights",
        default="1.0,1.0,1.0,0.5,0.5",
        help="Semicolon-separated groups w1,w2,w3,w4,w5",
    )
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--max-restarts", type=int, default=2)
    parser.add_argument("--template-trials", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=24)
    parser.add_argument("--filler-max-per-length", type=int, default=1200)
    parser.add_argument("--filler-weight", type=float, default=0.01)
    args = parser.parse_args()

    if args.offline:
        os.environ["CROSSWORD_OFFLINE"] = "1"

    seeds = [seed.strip() for seed in args.seeds.split(",") if seed.strip()]
    candidate_groups = _parse_weight_list(args.candidate_weights, 4)
    k_groups = _parse_weight_list(args.k_weights, 5)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_lines = [
        "candidate_weights,k_weights,seed,fill_status,fill_percent,output_dir",
    ]

    for c_idx, c_weights in enumerate(candidate_groups):
        candidate_weights = {"a": c_weights[0], "b": c_weights[1], "c": c_weights[2], "d": c_weights[3]}
        for k_idx, k_weights_tuple in enumerate(k_groups):
            k_weights = {
                "w1": k_weights_tuple[0],
                "w2": k_weights_tuple[1],
                "w3": k_weights_tuple[2],
                "w4": k_weights_tuple[3],
                "w5": k_weights_tuple[4],
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
                    max_steps=args.max_steps,
                    max_restarts=args.max_restarts,
                    template_trials=args.template_trials,
                    beam_width=args.beam_width,
                    filler_max_per_length=args.filler_max_per_length,
                    filler_weight=args.filler_weight,
                    use_rust=args.use_rust,
                )
                fill_status = result["solve"].diagnostics.get("fill_status", "unknown")
                fill_percent = result["solve"].diagnostics.get("fill_percent", 0.0)
                summary_lines.append(
                    f"\"{candidate_weights}\",\"{k_weights}\",{seed},{fill_status},{fill_percent},{combo_dir}"
                )

    summary_path = output_dir / "summary.csv"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
