from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json

from src.clue_builder import clue_pass_validate
from src.evaluation import summarize_benchmark
from src.pipeline import (
    run_candidate_scoring_stage,
    run_clue_extraction_stage,
    run_csp_solve_stage,
    run_k_selection_stage,
    run_packaging_stage,
    run_term_extraction_stage,
    run_vocab_gate_stage,
)


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _count_leakage(clues_path: Path, lang: str) -> tuple[int, int]:
    if not clues_path.exists():
        return 0, 0
    import csv

    try:
        import spacy

        model_name = "en_core_web_sm" if lang == "en" else "el_core_news_sm"
        nlp = spacy.load(model_name)
    except Exception:
        nlp = None

    total = 0
    leakage = 0
    with clues_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            total += 1
            clue = row.get("clue", "")
            answer = row.get("answer", "")
            if not clue_pass_validate(clue, answer, min_words=2, nlp=nlp, original_sentence=None):
                leakage += 1
    return leakage, total


def run_benchmark(seed: str, args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir) / seed.replace(" ", "_")
    output_dir.mkdir(parents=True, exist_ok=True)

    scores_path = output_dir / "candidate_scores.csv"
    diagnostics_scores = output_dir / "diagnostics_scores.json"
    selected_path = output_dir / "selected_candidates.json"
    trace_path = output_dir / "k_selection_trace.csv"
    diagnostics_k = output_dir / "diagnostics_k.json"
    terms_path = output_dir / "answer_candidates.csv"
    diagnostics_terms = output_dir / "diagnostics_terms.json"
    diagnostics_gate = output_dir / "diagnostics_vocab_gate.json"
    clues_path = output_dir / "clues.csv"
    diagnostics_clues = output_dir / "diagnostics_clues.json"
    grid_path = output_dir / "grid.json"
    diagnostics_csp = output_dir / "diagnostics_csp.json"
    puzzle_path = output_dir / "puzzle.json"
    attribution_path = output_dir / "attribution.json"
    diagnostics_package = output_dir / "diagnostics_package.json"

    scoring = run_candidate_scoring_stage(
        seed_title=seed,
        lang=args.lang,
        cache_dir=args.cache_dir,
        diagnostics_path=diagnostics_scores,
        scores_path=scores_path,
        include_backlinks=not args.no_backlinks,
        max_links=args.max_links,
        max_backlinks=args.max_backlinks,
        expansion=args.expansion,
        max_two_hop_parents=args.max_two_hop_parents,
        max_two_hop_links=args.max_two_hop_links,
        max_candidates=args.max_candidates,
        keep_threshold=args.keep_threshold,
        borderline_threshold=args.borderline_threshold,
        lexicon_path=args.lexicon_path,
        lexicon_weight=args.candidate_lexicon_weight,
    )

    selection = run_k_selection_stage(
        seed_title=seed,
        lang=args.lang,
        cache_dir=args.cache_dir,
        diagnostics_path=diagnostics_k,
        scores_path=scores_path,
        trace_path=trace_path,
        selected_path=selected_path,
        include_backlinks=not args.no_backlinks,
        max_links=args.max_links,
        max_backlinks=args.max_backlinks,
        expansion=args.expansion,
        max_two_hop_parents=args.max_two_hop_parents,
        max_two_hop_links=args.max_two_hop_links,
        max_candidates=args.max_candidates,
        keep_threshold=args.keep_threshold,
        borderline_threshold=args.borderline_threshold,
        min_k=args.min_k,
        max_k=args.max_k,
        epsilon=args.epsilon,
        m=args.m,
        size=args.grid_size,
        min_slot_len=args.min_slot_len,
        lexicon_path=args.lexicon_path,
        lexicon_weight=args.candidate_lexicon_weight,
    )

    terms = run_term_extraction_stage(
        seed_title=seed,
        lang=args.lang,
        cache_dir=args.cache_dir,
        diagnostics_path=diagnostics_terms,
        selected_path=selected_path,
        terms_path=terms_path,
        min_len=args.min_len,
        max_len=args.max_len,
        min_alpha_ratio=args.min_alpha_ratio,
        min_df=args.min_df,
        nlp_backend=args.nlp_backend,
        entity_type_scoring=args.entity_type_scoring,
        wikidata_cache_dir=args.wikidata_cache_dir,
        lexicon_path=args.lexicon_path,
        lexicon_weight=args.term_lexicon_weight,
    )

    gate = run_vocab_gate_stage(
        seed_title=seed,
        lang=args.lang,
        diagnostics_path=diagnostics_gate,
        terms_path=terms_path,
        min_required=args.gate_min,
        max_allowed=args.gate_max,
    )

    clues = run_clue_extraction_stage(
        seed_title=seed,
        lang=args.lang,
        cache_dir=args.cache_dir,
        diagnostics_path=diagnostics_clues,
        terms_path=terms_path,
        clues_path=clues_path,
        min_words=args.clue_min_words,
        max_words=args.clue_max_words,
        diversity_cap=args.diversity_cap,
    )

    solve = run_csp_solve_stage(
        seed_title=seed,
        lang=args.lang,
        terms_path=terms_path,
        diagnostics_path=diagnostics_csp,
        grid_path=grid_path,
        size=args.grid_size,
        min_slot_len=args.min_slot_len,
        template_name=args.template,
        max_steps=args.max_steps,
        min_domain=args.min_domain,
        max_restarts=args.max_restarts,
        random_seed=args.random_seed,
        use_ac3=not args.no_ac3,
        beam_width=args.beam_width,
        enable_local_repair=not args.no_local_repair,
        repair_steps=args.repair_steps,
        template_trials=args.template_trials,
        filler_path=None if args.no_filler else args.filler_path,
        filler_min_len=args.filler_min_len,
        filler_max_len=args.filler_max_len,
        filler_max_per_length=args.filler_max_per_length,
        filler_weight=args.filler_weight,
        require_gate=not args.skip_gate,
        gate_min=args.gate_min,
        gate_max=args.gate_max,
    )

    package = run_packaging_stage(
        seed_title=seed,
        lang=args.lang,
        selected_path=selected_path,
        grid_path=grid_path,
        clues_path=clues_path,
        puzzle_path=puzzle_path,
        attribution_path=attribution_path,
        diagnostics_path=diagnostics_package,
    )

    leakage, total_clues = _count_leakage(clues_path, args.lang)
    leakage_rate = 0.0 if total_clues == 0 else leakage / total_clues

    diagnostics_scores_payload = _load_json(diagnostics_scores)
    diagnostics_terms_payload = _load_json(diagnostics_terms)
    diagnostics_clues_payload = _load_json(diagnostics_clues)
    diagnostics_csp_payload = _load_json(diagnostics_csp)
    diagnostics_package_payload = _load_json(diagnostics_package)

    payload = {
        "seed": seed,
        "pipeline_errors": (
            diagnostics_scores_payload.get("errors", [])
            + diagnostics_terms_payload.get("errors", [])
            + diagnostics_csp_payload.get("errors", [])
            + diagnostics_package_payload.get("errors", [])
        ),
        "candidate_count": diagnostics_scores_payload.get("candidate_count", 0),
        "selected_k": selection.diagnostics.get("selected_k", 0),
        "term_count": diagnostics_terms_payload.get("term_count", 0),
        "clue_count": diagnostics_clues_payload.get("clue_count", 0),
        "leakage_count": leakage,
        "leakage_rate": leakage_rate,
        "fill_status": diagnostics_csp_payload.get("fill_status", "failed"),
        "fill_percent": diagnostics_csp_payload.get("fill_percent", 0.0),
        "provenance_missing_count": diagnostics_package_payload.get("provenance_missing_count", 0),
    }
    summary = summarize_benchmark(payload)
    summary_path = output_dir / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary.__dict__, indent=2), encoding="utf-8")
    return summary.__dict__


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark runner for crossword pipeline")
    parser.add_argument(
        "--seeds",
        default="Thermodynamics,Jazz",
        help="Comma-separated list of seed titles",
    )
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
    parser.add_argument("--min-slot-len", type=int, default=3)
    parser.add_argument("--min-len", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=12)
    parser.add_argument("--min-alpha-ratio", type=float, default=0.8)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--nlp-backend", choices=["auto", "spacy", "nltk"], default="auto")
    parser.add_argument("--entity-type-scoring", action="store_true")
    parser.add_argument("--gate-min", type=int, default=40)
    parser.add_argument("--gate-max", type=int, default=250)
    parser.add_argument("--clue-min-words", type=int, default=6)
    parser.add_argument("--clue-max-words", type=int, default=12)
    parser.add_argument("--diversity-cap", type=int, default=2)
    parser.add_argument("--template", default=None)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--min-domain", type=int, default=1)
    parser.add_argument("--max-restarts", type=int, default=2)
    parser.add_argument("--template-trials", type=int, default=3)
    parser.add_argument("--random-seed", type=int, default=13)
    parser.add_argument("--no-ac3", action="store_true")
    parser.add_argument("--beam-width", type=int, default=32)
    parser.add_argument("--no-local-repair", action="store_true")
    parser.add_argument("--repair-steps", type=int, default=300)
    parser.add_argument("--filler-path", default="data/lexicon/filler_words.txt")
    parser.add_argument("--no-filler", action="store_true")
    parser.add_argument("--filler-min-len", type=int, default=3)
    parser.add_argument("--filler-max-len", type=int, default=12)
    parser.add_argument("--filler-max-per-length", type=int, default=4000)
    parser.add_argument("--filler-weight", type=float, default=0.05)
    parser.add_argument("--skip-gate", action="store_true")
    args = parser.parse_args()

    seeds = [seed.strip() for seed in args.seeds.split(",") if seed.strip()]
    results = [run_benchmark(seed, args) for seed in seeds]
    summary_path = Path(args.output_dir) / "benchmarks_summary.json"
    summary_path.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
