from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _gate_a_scores(fixture: dict, pool_path: Path) -> dict:
    """Central recall and off-theme count of one seed's answer pool; ungraded without fixture or pool."""
    if not fixture or not pool_path.exists():
        return {"gate_a_graded": False}
    with pool_path.open(encoding="utf-8", newline="") as handle:
        pool = {row["source_titles"].split("|")[0] for row in csv.DictReader(handle)}
    central = fixture["central"]
    return {
        "gate_a_graded": True,
        "central_recall": sum(title in pool for title in central) / len(central),
        "off_theme_count": sum(title in pool for title in fixture["off_theme"]),
    }


def _load_nlp(lang: str):
    try:
        import spacy

        return spacy.load("en_core_web_sm" if lang == "en" else "el_core_news_sm")
    except (ImportError, OSError):  # the pipeline run records the same failure in its terms diagnostics
        return None


def _score_entries(puzzle: dict, lang: str) -> dict:
    """Leakage, blank, duplicate and clue-style counts over the entries actually placed."""
    from src.clue_builder import BLANK, answer_pattern, has_morphological_leakage

    entries = puzzle.get("across_entries", []) + puzzle.get("down_entries", [])
    nlp = _load_nlp(lang)
    leakage = 0
    blank_missing = 0
    definitions = 0
    seen: set[str] = set()
    duplicates = 0
    for entry in entries:
        clue = entry.get("clue", "")
        answer = entry.get("display_answer") or entry.get("answer", "")
        if answer_pattern(answer).search(clue) or has_morphological_leakage(clue, answer, nlp, lang=lang):
            leakage += 1
        if entry.get("clue_style") == "definition":
            definitions += 1
        elif BLANK not in clue:
            blank_missing += 1
        key = " ".join(clue.lower().split())
        duplicates += key in seen
        seen.add(key)
    total = len(entries)
    return {
        "leakage_count": leakage,
        "leakage_rate": leakage / total if total else 0.0,
        "blank_missing_count": blank_missing,
        "duplicate_clue_count": duplicates,
        "definition_clue_ratio": definitions / total if total else 0.0,
    }


def run_benchmark(seed: str, args: argparse.Namespace, fixtures: dict) -> dict:
    from src.evaluation import summarize_benchmark
    from src.pipeline import run_generate_pipeline

    output_dir = Path(args.output_dir) / seed.replace(" ", "_")

    run_generate_pipeline(
        seed_title=seed,
        lang=args.lang,
        output_dir=output_dir,
        cache_dir=args.cache_dir,
        wikidata_cache_dir=args.wikidata_cache_dir,
        include_backlinks=not args.no_backlinks,
        # A fixture written for a sample of the seed's links (Ancient Rome) is only valid on that sample.
        max_links=fixtures.get(seed, {}).get("max_links", args.max_links),
        max_backlinks=args.max_backlinks,
        expansion=args.expansion,
        max_two_hop_parents=args.max_two_hop_parents,
        max_two_hop_links=args.max_two_hop_links,
        max_candidates=args.max_candidates,
        keep_threshold=args.keep_threshold,
        borderline_threshold=args.borderline_threshold,
        lexicon_path=args.lexicon_path,
        candidate_lexicon_weight=args.candidate_lexicon_weight,
        term_lexicon_weight=args.term_lexicon_weight,
        min_k=args.min_k,
        max_k=args.max_k,
        epsilon=args.epsilon,
        m=args.m,
        min_len=args.min_len,
        max_len=args.max_len,
        min_alpha_ratio=args.min_alpha_ratio,
        min_df=args.min_df,
        nlp_backend=args.nlp_backend,
        entity_type_scoring=args.entity_type_scoring,
        gate_min=args.gate_min,
        clue_min_words=args.clue_min_words,
        clue_max_words=args.clue_max_words,
        diversity_cap=args.diversity_cap,
        size=args.grid_size,
        random_seed=args.random_seed,
        layout_restarts=args.layout_restarts,
        layout_candidates=args.layout_candidates,
        min_words=args.min_words,
    )

    scores = _load_json(output_dir / "diagnostics_scores.json")
    terms = _load_json(output_dir / "diagnostics_terms.json")
    layout = _load_json(output_dir / "diagnostics_layout.json")
    package = _load_json(output_dir / "diagnostics_package.json")
    puzzle = _load_json(output_dir / "puzzle.json")

    payload = {
        "seed": seed,
        "puzzle_status": package.get("puzzle_status", "unknown"),
        "pipeline_errors": (
            scores.get("errors", []) + terms.get("errors", []) + layout.get("errors", []) + package.get("errors", [])
        ),
        "candidate_count": scores.get("candidate_count", 0),
        "selected_k": package.get("selected_k", 0),
        "term_count": terms.get("term_count", 0),
        "clue_count": package.get("clue_count", 0),
        "fill_status": layout.get("fill_status", "failed"),
        "valid": bool(layout) and not layout.get("validity_errors", ["missing"]),
        "used_clue_provenance_missing_count": package.get("used_clue_provenance_missing_count", 0),
        **{
            key: layout[key]
            for key in (
                "word_count",
                "crossing_count",
                "mean_crossings_per_word",
                "two_plus_crossings_ratio",
                "checked_letter_ratio",
            )
            if key in layout
        },
        **_score_entries(puzzle, args.lang),
        **_gate_a_scores(fixtures.get(seed, {}), output_dir / "answer_candidates.csv"),
    }
    summary = summarize_benchmark(payload)
    summary_path = output_dir / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
    return summary.to_dict()


def main() -> None:
    from src.evaluation import summarize_benchmark_collection

    parser = argparse.ArgumentParser(description="Benchmark runner for crossword pipeline")
    parser.add_argument(
        "--seeds",
        default="Thermodynamics,Jazz",
        help="Comma-separated list of seed titles",
    )
    parser.add_argument("--gate-a-fixture", default="data/benchmarks/gate_a.json", help="Gate A central/off-theme pages")
    parser.add_argument("--lang", default="en", help="Wikipedia language code")
    parser.add_argument("--cache-dir", default="data/cache/wiki", help="Disk cache directory")
    parser.add_argument("--wikidata-cache-dir", default="data/cache/wikidata", help="Wikidata cache")
    parser.add_argument("--output-dir", default="outputs/benchmarks", help="Output root directory")
    parser.add_argument("--max-links", type=int, default=None)
    parser.add_argument("--max-backlinks", type=int, default=None)
    parser.add_argument("--no-backlinks", action="store_true")
    parser.add_argument(
        "--expansion",
        default="one_hop_only",
        choices=["one_hop_only", "one_hop_plus_bounded_two_hop"],
    )
    parser.add_argument("--max-two-hop-parents", type=int, default=None)
    parser.add_argument("--max-two-hop-links", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--keep-threshold", type=float, default=0.2)
    parser.add_argument("--borderline-threshold", type=float, default=0.1)
    parser.add_argument(
        "--lexicon-path",
        default="data/lexicon/combined_wordfreq.txt",
        help="Optional external lexicon file (word[,score])",
    )
    parser.add_argument("--candidate-lexicon-weight", type=float, default=0.08)
    parser.add_argument("--term-lexicon-weight", type=float, default=0.15)
    parser.add_argument("--min-k", type=int, default=5)
    parser.add_argument("--max-k", type=int, default=None)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--grid-size", type=int, default=15)
    parser.add_argument("--min-len", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=15)
    parser.add_argument("--min-alpha-ratio", type=float, default=0.8)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--nlp-backend", choices=["auto", "spacy"], default="auto")
    parser.add_argument("--entity-type-scoring", action="store_true")
    parser.add_argument("--gate-min", type=int, default=30)
    parser.add_argument("--clue-min-words", type=int, default=6)
    parser.add_argument("--clue-max-words", type=int, default=12)
    parser.add_argument("--diversity-cap", type=int, default=2)
    parser.add_argument("--random-seed", type=int, default=13)
    parser.add_argument("--layout-restarts", type=int, default=96)
    parser.add_argument("--layout-candidates", type=int, default=80)
    parser.add_argument("--min-words", type=int, default=12)
    args = parser.parse_args()

    seeds = [seed.strip() for seed in args.seeds.split(",") if seed.strip()]
    fixtures = _load_json(Path(args.gate_a_fixture)).get("seeds", {})
    results = [run_benchmark(seed, args, fixtures) for seed in seeds]
    aggregate = summarize_benchmark_collection(results)
    summary_path = Path(args.output_dir) / "benchmarks_summary.json"
    summary_path.write_text(
        json.dumps({"results": results, "aggregate": aggregate.to_dict()}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
