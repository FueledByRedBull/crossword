from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from src.pipeline import (
    run_candidate_scoring_stage,
    run_k_selection_stage,
    run_seed_ingestion_stage,
    run_term_extraction_stage,
    run_vocab_gate_stage,
    run_clue_extraction_stage,
    run_rescue_ladder,
    run_layout_stage,
    run_packaging_stage,
    run_generate_pipeline,
)
from src.export import export_html, export_ipuz
from src.render import render_puzzle_text


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wikipedia-seeded crossword pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    seed_parser = subparsers.add_parser(
        "seed",
        help="Run seed ingestion stage and emit diagnostics.json",
    )
    seed_parser.add_argument("--seed", required=True, help="Seed Wikipedia article title")
    seed_parser.add_argument(
        "--lang",
        default="en",
        help="Language code for Wikipedia API (default: en, also supports el)",
    )
    seed_parser.add_argument(
        "--output",
        default="outputs/diagnostics.json",
        help="Path to diagnostics JSON output",
    )
    seed_parser.add_argument(
        "--cache-dir",
        default="data/cache/wiki",
        help="Disk cache directory",
    )
    seed_parser.add_argument(
        "--max-links",
        type=int,
        default=None,
        help="Optional cap on outgoing links fetched",
    )
    seed_parser.add_argument(
        "--max-backlinks",
        type=int,
        default=None,
        help="Optional cap on backlinks fetched",
    )
    seed_parser.add_argument(
        "--no-backlinks",
        action="store_true",
        help="Skip backlink annotation for candidates",
    )
    seed_parser.add_argument(
        "--expansion",
        default="one_hop_only",
        choices=["one_hop_only", "one_hop_plus_bounded_two_hop"],
        help="Expansion policy for candidate graph",
    )
    seed_parser.add_argument(
        "--max-two-hop-parents",
        type=int,
        default=None,
        help="Max number of 1-hop parents to expand for 2-hop links",
    )
    seed_parser.add_argument(
        "--max-two-hop-links",
        type=int,
        default=None,
        help="Max number of links per 2-hop parent",
    )
    seed_parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Optional cap on total candidates (1-hop + 2-hop)",
    )

    score_parser = subparsers.add_parser(
        "rank",
        help="Run candidate scoring stage and emit candidate_scores.csv",
    )
    score_parser.add_argument("--seed", required=True, help="Seed Wikipedia article title")
    score_parser.add_argument(
        "--lang",
        default="en",
        help="Language code for Wikipedia API (default: en, also supports el)",
    )
    score_parser.add_argument(
        "--output",
        default="outputs/candidate_scores.csv",
        help="Path to candidate scores CSV output",
    )
    score_parser.add_argument(
        "--diagnostics",
        default="outputs/diagnostics_scores.json",
        help="Path to scoring diagnostics JSON output",
    )
    score_parser.add_argument(
        "--cache-dir",
        default="data/cache/wiki",
        help="Disk cache directory",
    )
    score_parser.add_argument(
        "--max-links",
        type=int,
        default=None,
        help="Optional cap on outgoing links fetched",
    )
    score_parser.add_argument(
        "--max-backlinks",
        type=int,
        default=None,
        help="Optional cap on backlinks fetched",
    )
    score_parser.add_argument(
        "--no-backlinks",
        action="store_true",
        help="Skip backlink annotation for candidates",
    )
    score_parser.add_argument(
        "--keep-threshold",
        type=float,
        default=0.2,
        help="Relevance from which an on-topic page is KEEP instead of BORDERLINE",
    )
    score_parser.add_argument(
        "--lexicon-path",
        default="data/lexicon/combined_wordfreq.txt",
        help="Optional external lexicon file (word[,score])",
    )
    score_parser.add_argument(
        "--lexicon-weight",
        type=float,
        default=0.08,
        help="Weight of lexicon score in article candidate ranking",
    )
    score_parser.add_argument(
        "--expansion",
        default="one_hop_only",
        choices=["one_hop_only", "one_hop_plus_bounded_two_hop"],
        help="Expansion policy for candidate graph",
    )
    score_parser.add_argument(
        "--max-two-hop-parents",
        type=int,
        default=None,
        help="Max number of 1-hop parents to expand for 2-hop links",
    )
    score_parser.add_argument(
        "--max-two-hop-links",
        type=int,
        default=None,
        help="Max number of links per 2-hop parent",
    )
    score_parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Optional cap on total candidates (1-hop + 2-hop)",
    )

    k_parser = subparsers.add_parser(
        "select-k",
        help="Select K candidates and emit k_selection_trace.csv",
    )
    k_parser.add_argument("--seed", required=True, help="Seed Wikipedia article title")
    k_parser.add_argument(
        "--lang",
        default="en",
        help="Language code for Wikipedia API (default: en, also supports el)",
    )
    k_parser.add_argument(
        "--scores",
        default="outputs/candidate_scores.csv",
        help="Path to candidate scores CSV output",
    )
    k_parser.add_argument(
        "--output",
        default="outputs/selected_candidates.json",
        help="Path to selected candidates JSON output",
    )
    k_parser.add_argument(
        "--trace",
        default="outputs/k_selection_trace.csv",
        help="Path to K selection trace CSV output",
    )
    k_parser.add_argument(
        "--diagnostics",
        default="outputs/diagnostics_k.json",
        help="Path to K selection diagnostics JSON output",
    )
    k_parser.add_argument(
        "--cache-dir",
        default="data/cache/wiki",
        help="Disk cache directory",
    )
    k_parser.add_argument(
        "--max-links",
        type=int,
        default=None,
        help="Optional cap on outgoing links fetched",
    )
    k_parser.add_argument(
        "--max-backlinks",
        type=int,
        default=None,
        help="Optional cap on backlinks fetched",
    )
    k_parser.add_argument(
        "--no-backlinks",
        action="store_true",
        help="Skip backlink annotation for candidates",
    )
    k_parser.add_argument(
        "--keep-threshold",
        type=float,
        default=0.2,
        help="Relevance from which an on-topic page is KEEP instead of BORDERLINE",
    )
    k_parser.add_argument(
        "--lexicon-path",
        default="data/lexicon/combined_wordfreq.txt",
        help="Optional external lexicon file (word[,score])",
    )
    k_parser.add_argument(
        "--lexicon-weight",
        type=float,
        default=0.08,
        help="Weight of lexicon score in article candidate ranking",
    )
    k_parser.add_argument(
        "--expansion",
        default="one_hop_only",
        choices=["one_hop_only", "one_hop_plus_bounded_two_hop"],
        help="Expansion policy for candidate graph",
    )
    k_parser.add_argument(
        "--max-two-hop-parents",
        type=int,
        default=None,
        help="Max number of 1-hop parents to expand for 2-hop links",
    )
    k_parser.add_argument(
        "--max-two-hop-links",
        type=int,
        default=None,
        help="Max number of links per 2-hop parent",
    )
    k_parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Optional cap on total candidates (1-hop + 2-hop)",
    )
    k_parser.add_argument("--min-k", type=int, default=5, help="Minimum K to select")
    k_parser.add_argument("--max-k", type=int, default=None, help="Maximum K to select")
    k_parser.add_argument("--epsilon", type=float, default=0.01, help="Marginal gain epsilon")
    k_parser.add_argument("--m", type=int, default=2, help="Consecutive epsilon count to stop")
    k_parser.add_argument("--grid-size", type=int, default=15, help="Grid size")

    term_parser = subparsers.add_parser(
        "terms",
        help="Extract term candidates from selected articles",
    )
    term_parser.add_argument("--seed", required=True, help="Seed Wikipedia article title")
    term_parser.add_argument(
        "--lang",
        default="en",
        help="Language code for Wikipedia API (default: en, also supports el)",
    )
    term_parser.add_argument(
        "--selected",
        default="outputs/selected_candidates.json",
        help="Path to selected candidates JSON",
    )
    term_parser.add_argument(
        "--output",
        default="outputs/answer_candidates.csv",
        help="Path to answer candidates CSV output",
    )
    term_parser.add_argument(
        "--diagnostics",
        default="outputs/diagnostics_terms.json",
        help="Path to term extraction diagnostics JSON output",
    )
    term_parser.add_argument(
        "--cache-dir",
        default="data/cache/wiki",
        help="Disk cache directory",
    )
    term_parser.add_argument(
        "--scores-path",
        default=None,
        help="Candidate scores CSV; adds the selected articles' titles to the answer pool",
    )
    term_parser.add_argument("--min-len", type=int, default=4, help="Minimum answer length")
    term_parser.add_argument("--max-len", type=int, default=15, help="Maximum answer length")
    term_parser.add_argument(
        "--min-alpha-ratio",
        type=float,
        default=0.8,
        help="Minimum alphabetic ratio for candidate term",
    )
    term_parser.add_argument("--min-df", type=int, default=2, help="Minimum doc frequency")
    term_parser.add_argument(
        "--nlp-backend",
        choices=["auto", "spacy"],
        default="auto",
        help="NLP backend for term extraction",
    )
    term_parser.add_argument(
        "--entity-type-scoring",
        action="store_true",
        help="Enable Wikidata entity-type scoring",
    )
    term_parser.add_argument(
        "--wikidata-cache-dir",
        default="data/cache/wikidata",
        help="Disk cache directory for Wikidata API calls",
    )
    term_parser.add_argument(
        "--lexicon-path",
        default="data/lexicon/combined_wordfreq.txt",
        help="Optional external lexicon file (word[,score])",
    )
    term_parser.add_argument(
        "--lexicon-weight",
        type=float,
        default=0.15,
        help="Weight of lexicon score in answer ranking",
    )

    gate_parser = subparsers.add_parser(
        "vocab-gate",
        help="Evaluate vocabulary readiness gate",
    )
    gate_parser.add_argument("--seed", required=True, help="Seed Wikipedia article title")
    gate_parser.add_argument(
        "--lang",
        default="en",
        help="Language code for Wikipedia API (default: en, also supports el)",
    )
    gate_parser.add_argument(
        "--terms",
        default="outputs/answer_candidates.csv",
        help="Path to answer candidates CSV input",
    )
    gate_parser.add_argument(
        "--clues-path",
        default=None,
        help="Clues CSV; when given, count distinct packageable clued answers instead of terms",
    )
    gate_parser.add_argument(
        "--diagnostics",
        default="outputs/diagnostics_vocab_gate.json",
        help="Path to vocab gate diagnostics JSON output",
    )
    gate_parser.add_argument("--min-required", type=int, default=30, help="Minimum terms required")
    gate_parser.add_argument("--max-allowed", type=int, default=None, help="Maximum terms allowed")

    clue_parser = subparsers.add_parser(
        "clues",
        help="Generate clues from term candidates",
    )
    clue_parser.add_argument("--seed", required=True, help="Seed Wikipedia article title")
    clue_parser.add_argument(
        "--lang",
        default="en",
        help="Language code for Wikipedia API (default: en, also supports el)",
    )
    clue_parser.add_argument(
        "--terms",
        default="outputs/answer_candidates.csv",
        help="Path to answer candidates CSV input",
    )
    clue_parser.add_argument(
        "--selected",
        default="outputs/selected_candidates.json",
        help="Path to selected candidates JSON; these articles are cited from their full text",
    )
    clue_parser.add_argument(
        "--output",
        default="outputs/clues.csv",
        help="Path to clue CSV output",
    )
    clue_parser.add_argument(
        "--diagnostics",
        default="outputs/diagnostics_clues.json",
        help="Path to clue diagnostics JSON output",
    )
    clue_parser.add_argument(
        "--cache-dir",
        default="data/cache/wiki",
        help="Disk cache directory",
    )
    clue_parser.add_argument("--min-words", type=int, default=6, help="Minimum words per clue")
    clue_parser.add_argument("--max-words", type=int, default=12, help="Maximum words per clue")
    clue_parser.add_argument(
        "--diversity-cap",
        type=int,
        default=2,
        help="Maximum clues per 3-token prefix bucket",
    )

    rescue_parser = subparsers.add_parser(
        "rescue",
        help="Run thin-pool rescue ladder for vocabulary readiness",
    )
    rescue_parser.add_argument("--seed", required=True, help="Seed Wikipedia article title")
    rescue_parser.add_argument(
        "--lang",
        default="en",
        help="Language code for Wikipedia API (default: en, also supports el)",
    )
    rescue_parser.add_argument(
        "--selected",
        default="outputs/selected_candidates.json",
        help="Path to selected candidates JSON",
    )
    rescue_parser.add_argument(
        "--cache-dir",
        default="data/cache/wiki",
        help="Disk cache directory",
    )
    rescue_parser.add_argument(
        "--terms",
        default="outputs/answer_candidates.csv",
        help="Path to answer candidates CSV output",
    )
    rescue_parser.add_argument(
        "--diagnostics",
        default="outputs/diagnostics_rescue.json",
        help="Path to rescue diagnostics JSON output",
    )
    rescue_parser.add_argument("--gate-min", type=int, default=30, help="Minimum clued answers required")
    rescue_parser.add_argument("--min-len", type=int, default=4, help="Minimum answer length")
    rescue_parser.add_argument("--max-len", type=int, default=15, help="Maximum answer length")
    rescue_parser.add_argument("--min-df", type=int, default=2, help="Minimum doc frequency")
    rescue_parser.add_argument(
        "--min-alpha-ratio",
        type=float,
        default=0.8,
        help="Minimum alphabetic ratio for candidate term",
    )
    rescue_parser.add_argument(
        "--nlp-backend",
        choices=["auto", "spacy"],
        default="auto",
        help="NLP backend for rescue term extraction",
    )
    rescue_parser.add_argument(
        "--entity-type-scoring",
        action="store_true",
        help="Enable Wikidata entity-type scoring during rescue term extraction",
    )
    rescue_parser.add_argument(
        "--wikidata-cache-dir",
        default="data/cache/wikidata",
        help="Disk cache directory for Wikidata API calls during rescue",
    )
    rescue_parser.add_argument(
        "--lexicon-path",
        default="data/lexicon/combined_wordfreq.txt",
        help="Optional external lexicon file (word[,score]) for rescue term extraction",
    )
    rescue_parser.add_argument(
        "--lexicon-weight",
        type=float,
        default=0.15,
        help="Weight of lexicon score in rescue answer ranking",
    )

    layout_parser = subparsers.add_parser(
        "layout",
        help="Build a free-form crossword layout from clued answers",
    )
    layout_parser.add_argument("--seed", required=True, help="Seed Wikipedia article title")
    layout_parser.add_argument(
        "--lang",
        default="en",
        help="Language code for Wikipedia API (default: en, also supports el)",
    )
    layout_parser.add_argument(
        "--terms",
        default="outputs/answer_candidates.csv",
        help="Path to answer candidates CSV input",
    )
    layout_parser.add_argument(
        "--clues",
        default="outputs/clues.csv",
        help="Path to clues CSV input",
    )
    layout_parser.add_argument(
        "--output",
        default="outputs/grid.json",
        help="Path to grid JSON output",
    )
    layout_parser.add_argument(
        "--diagnostics",
        default="outputs/diagnostics_layout.json",
        help="Path to layout diagnostics JSON output",
    )
    layout_parser.add_argument("--size", type=int, default=15, help="Grid size")
    layout_parser.add_argument("--random-seed", type=int, default=13, help="Random seed for the layout search")
    layout_parser.add_argument("--restarts", type=int, default=96, help="Layout search effort: 25 steps each")
    layout_parser.add_argument("--candidates", type=int, default=80, help="Maximum answers offered to the layout")
    layout_parser.add_argument("--min-words", type=int, default=12, help="Words needed for a complete layout")
    layout_parser.add_argument("--gate-min", type=int, default=30, help="Minimum clued answers required")

    package_parser = subparsers.add_parser(
        "package",
        help="Package puzzle output and attribution bundle",
    )
    package_parser.add_argument("--seed", required=True, help="Seed Wikipedia article title")
    package_parser.add_argument(
        "--lang",
        default="en",
        help="Language code for Wikipedia API (default: en, also supports el)",
    )
    package_parser.add_argument(
        "--selected",
        default="outputs/selected_candidates.json",
        help="Path to selected candidates JSON input",
    )
    package_parser.add_argument(
        "--grid",
        default="outputs/grid.json",
        help="Path to grid JSON input",
    )
    package_parser.add_argument(
        "--clues",
        default="outputs/clues.csv",
        help="Path to clues CSV input",
    )
    package_parser.add_argument(
        "--output",
        default="outputs/puzzle.json",
        help="Path to puzzle JSON output",
    )
    package_parser.add_argument(
        "--attribution",
        default="outputs/attribution.json",
        help="Path to attribution JSON output",
    )
    package_parser.add_argument(
        "--diagnostics",
        default="outputs/diagnostics_package.json",
        help="Path to packaging diagnostics JSON output",
    )
    package_parser.add_argument("--gate-min", type=int, default=30, help="Minimum clued answers required")

    generate_parser = subparsers.add_parser(
        "generate",
        help="Run full pipeline end-to-end",
    )
    generate_parser.add_argument("--seed", required=True, help="Seed Wikipedia article title")
    generate_parser.add_argument(
        "--lang",
        default="en",
        help="Language code for Wikipedia API (default: en, also supports el)",
    )
    generate_parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Output directory for all stage artifacts",
    )
    generate_parser.add_argument(
        "--cache-dir",
        default="data/cache/wiki",
        help="Disk cache directory",
    )
    generate_parser.add_argument(
        "--wikidata-cache-dir",
        default="data/cache/wikidata",
        help="Wikidata cache directory",
    )
    generate_parser.add_argument(
        "--offline",
        action="store_true",
        help="Use cache only (skip network requests)",
    )
    generate_parser.add_argument("--max-links", type=int, default=None)
    generate_parser.add_argument("--max-backlinks", type=int, default=None)
    generate_parser.add_argument("--no-backlinks", action="store_true")
    generate_parser.add_argument(
        "--expansion",
        default="one_hop_only",
        choices=["one_hop_only", "one_hop_plus_bounded_two_hop"],
    )
    generate_parser.add_argument("--max-two-hop-parents", type=int, default=None)
    generate_parser.add_argument("--max-two-hop-links", type=int, default=None)
    generate_parser.add_argument("--max-candidates", type=int, default=None)
    generate_parser.add_argument("--keep-threshold", type=float, default=0.2)
    generate_parser.add_argument(
        "--lexicon-path",
        default="data/lexicon/combined_wordfreq.txt",
        help="Optional external lexicon file (word[,score])",
    )
    generate_parser.add_argument("--candidate-lexicon-weight", type=float, default=0.08)
    generate_parser.add_argument("--term-lexicon-weight", type=float, default=0.15)
    generate_parser.add_argument("--min-k", type=int, default=5)
    generate_parser.add_argument("--max-k", type=int, default=None)
    generate_parser.add_argument("--epsilon", type=float, default=0.01)
    generate_parser.add_argument("--m", type=int, default=2)
    generate_parser.add_argument("--grid-size", type=int, default=15)
    generate_parser.add_argument("--min-len", type=int, default=4)
    generate_parser.add_argument("--max-len", type=int, default=15)
    generate_parser.add_argument("--min-alpha-ratio", type=float, default=0.8)
    generate_parser.add_argument("--min-df", type=int, default=2)
    generate_parser.add_argument("--nlp-backend", choices=["auto", "spacy"], default="auto")
    generate_parser.add_argument("--entity-type-scoring", action="store_true")
    generate_parser.add_argument("--gate-min", type=int, default=30)
    generate_parser.add_argument("--clue-min-words", type=int, default=6)
    generate_parser.add_argument("--clue-max-words", type=int, default=12)
    generate_parser.add_argument("--diversity-cap", type=int, default=2)
    generate_parser.add_argument("--random-seed", type=int, default=13)
    generate_parser.add_argument("--layout-restarts", type=int, default=96, help="Layout search effort: 25 steps each")
    generate_parser.add_argument("--layout-candidates", type=int, default=80)
    generate_parser.add_argument("--min-words", type=int, default=12)
    generate_parser.add_argument("--no-rescue", action="store_true")

    render_parser = subparsers.add_parser(
        "render",
        help="Print a puzzle.json as a text grid with clues and validity report",
    )
    render_parser.add_argument("--puzzle", required=True, help="Path to puzzle JSON")

    export_parser = subparsers.add_parser(
        "export",
        help="Export a puzzle.json as .ipuz or a printable HTML page",
    )
    export_parser.add_argument("--puzzle", required=True, help="Path to puzzle JSON")
    export_parser.add_argument("--format", required=True, choices=["ipuz", "html"], help="Output format")
    export_parser.add_argument("--output", required=True, help="Path of the file to write")
    export_parser.add_argument(
        "--solution",
        action="store_true",
        help="HTML only: add the solved grid on its own page (ipuz always carries it)",
    )

    return parser


def _report_problems(result: dict, started: float) -> bool:
    """Say on stderr why a run came out failed or thin: its statuses and the first errors the stages recorded.

    True when the puzzle is complete and ok; errors a stage recorded on the way do not make it false.
    """
    package = result["package"].diagnostics
    recorded: dict[str, list[str]] = {}
    for path in sorted(Path(result["output_dir"]).glob("diagnostics_*.json")):
        if path.stat().st_mtime >= started:  # an earlier run in the same directory may have left others
            recorded[path.name] = json.loads(path.read_text(encoding="utf-8")).get("errors", [])
    messages = list(dict.fromkeys(message for found in recorded.values() for message in found))
    usable = package["fill_status"] == "complete" and package["puzzle_status"] == "ok"
    if usable and not messages:
        return True
    print(
        f"Problems: fill_status={package['fill_status']}, puzzle_status={package['puzzle_status']}, "
        f"{len(messages)} distinct errors recorded",
        file=sys.stderr,
    )
    if messages:
        print(f"Recorded in: {', '.join(name for name, found in recorded.items() if found)}", file=sys.stderr)
    for message in messages[:5]:
        print(f"  {message[:160]}", file=sys.stderr)
    return usable


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "seed":
        result = run_seed_ingestion_stage(
            seed_title=args.seed,
            lang=args.lang,
            diagnostics_path=args.output,
            cache_dir=args.cache_dir,
            include_backlinks=not args.no_backlinks,
            max_links=args.max_links,
            max_backlinks=args.max_backlinks,
            expansion=args.expansion,
            max_two_hop_parents=args.max_two_hop_parents,
            max_two_hop_links=args.max_two_hop_links,
            max_candidates=args.max_candidates,
        )
        print(f"Diagnostics written: {result.diagnostics_path}")
        print(f"Candidates: {result.diagnostics['counts']['candidate_count']}")
        print(f"Cache stats: {result.diagnostics['cache']}")
    elif args.command == "rank":
        result = run_candidate_scoring_stage(
            seed_title=args.seed,
            lang=args.lang,
            cache_dir=args.cache_dir,
            diagnostics_path=args.diagnostics,
            scores_path=args.output,
            include_backlinks=not args.no_backlinks,
            max_links=args.max_links,
            max_backlinks=args.max_backlinks,
            expansion=args.expansion,
            max_two_hop_parents=args.max_two_hop_parents,
            max_two_hop_links=args.max_two_hop_links,
            max_candidates=args.max_candidates,
            keep_threshold=args.keep_threshold,
            lexicon_path=args.lexicon_path,
            lexicon_weight=args.lexicon_weight,
        )
        print(f"Scores written: {result.scores_path}")
        print(f"Diagnostics written: {result.diagnostics_path}")
        print(f"Candidates scored: {result.diagnostics.get('candidate_count', 0)}")
    elif args.command == "select-k":
        result = run_k_selection_stage(
            seed_title=args.seed,
            lang=args.lang,
            cache_dir=args.cache_dir,
            diagnostics_path=args.diagnostics,
            scores_path=args.scores,
            trace_path=args.trace,
            selected_path=args.output,
            include_backlinks=not args.no_backlinks,
            max_links=args.max_links,
            max_backlinks=args.max_backlinks,
            expansion=args.expansion,
            max_two_hop_parents=args.max_two_hop_parents,
            max_two_hop_links=args.max_two_hop_links,
            max_candidates=args.max_candidates,
            keep_threshold=args.keep_threshold,
            min_k=args.min_k,
            max_k=args.max_k,
            epsilon=args.epsilon,
            m=args.m,
            size=args.grid_size,
            lexicon_path=args.lexicon_path,
            lexicon_weight=args.lexicon_weight,
        )
        print(f"Selected: {result.selected_path}")
        print(f"Trace: {result.trace_path}")
        print(f"Diagnostics: {result.diagnostics_path}")
    elif args.command == "terms":
        result = run_term_extraction_stage(
            seed_title=args.seed,
            lang=args.lang,
            cache_dir=args.cache_dir,
            diagnostics_path=args.diagnostics,
            selected_path=args.selected,
            terms_path=args.output,
            min_len=args.min_len,
            max_len=args.max_len,
            min_alpha_ratio=args.min_alpha_ratio,
            min_df=args.min_df,
            nlp_backend=args.nlp_backend,
            entity_type_scoring=args.entity_type_scoring,
            wikidata_cache_dir=args.wikidata_cache_dir,
            lexicon_path=args.lexicon_path,
            lexicon_weight=args.lexicon_weight,
            scores_path=args.scores_path,
        )
        print(f"Terms written: {result.terms_path}")
        print(f"Diagnostics: {result.diagnostics_path}")
    elif args.command == "vocab-gate":
        result = run_vocab_gate_stage(
            seed_title=args.seed,
            lang=args.lang,
            diagnostics_path=args.diagnostics,
            terms_path=args.terms,
            clues_path=args.clues_path,
            min_required=args.min_required,
            max_allowed=args.max_allowed,
        )
        print(f"Gate diagnostics: {result.diagnostics_path}")
    elif args.command == "clues":
        result = run_clue_extraction_stage(
            seed_title=args.seed,
            lang=args.lang,
            cache_dir=args.cache_dir,
            diagnostics_path=args.diagnostics,
            terms_path=args.terms,
            clues_path=args.output,
            selected_path=args.selected,
            min_words=args.min_words,
            max_words=args.max_words,
            diversity_cap=args.diversity_cap,
        )
        print(f"Clues written: {result.clues_path}")
        print(f"Diagnostics: {result.diagnostics_path}")
    elif args.command == "rescue":
        result = run_rescue_ladder(
            seed_title=args.seed,
            lang=args.lang,
            cache_dir=args.cache_dir,
            selected_path=args.selected,
            terms_path=args.terms,
            diagnostics_path=args.diagnostics,
            gate_min=args.gate_min,
            min_len=args.min_len,
            max_len=args.max_len,
            min_alpha_ratio=args.min_alpha_ratio,
            min_df=args.min_df,
            nlp_backend=args.nlp_backend,
            entity_type_scoring=args.entity_type_scoring,
            wikidata_cache_dir=args.wikidata_cache_dir,
            lexicon_path=args.lexicon_path,
            lexicon_weight=args.lexicon_weight,
        )
        print(f"Rescue diagnostics: {args.diagnostics}")
    elif args.command == "layout":
        result = run_layout_stage(
            seed_title=args.seed,
            lang=args.lang,
            terms_path=args.terms,
            clues_path=args.clues,
            grid_path=args.output,
            diagnostics_path=args.diagnostics,
            size=args.size,
            random_seed=args.random_seed,
            restarts=args.restarts,
            candidates=args.candidates,
            min_words=args.min_words,
            gate_min=args.gate_min,
        )
        print(f"Grid: {result.grid_path}")
        print(f"Diagnostics: {result.diagnostics_path}")
    elif args.command == "package":
        result = run_packaging_stage(
            seed_title=args.seed,
            lang=args.lang,
            selected_path=args.selected,
            grid_path=args.grid,
            clues_path=args.clues,
            puzzle_path=args.output,
            attribution_path=args.attribution,
            diagnostics_path=args.diagnostics,
            gate_min=args.gate_min,
        )
        print(f"Puzzle: {result.puzzle_path}")
        print(f"Attribution: {result.attribution_path}")
        print(f"Diagnostics: {result.diagnostics_path}")
    elif args.command == "generate":
        if args.offline:
            os.environ["CROSSWORD_OFFLINE"] = "1"
        started = time.time()
        result = run_generate_pipeline(
            seed_title=args.seed,
            lang=args.lang,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            wikidata_cache_dir=args.wikidata_cache_dir,
            include_backlinks=not args.no_backlinks,
            max_links=args.max_links,
            max_backlinks=args.max_backlinks,
            expansion=args.expansion,
            max_two_hop_parents=args.max_two_hop_parents,
            max_two_hop_links=args.max_two_hop_links,
            max_candidates=args.max_candidates,
            keep_threshold=args.keep_threshold,
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
            rescue=not args.no_rescue,
            clue_min_words=args.clue_min_words,
            clue_max_words=args.clue_max_words,
            diversity_cap=args.diversity_cap,
            size=args.grid_size,
            random_seed=args.random_seed,
            layout_restarts=args.layout_restarts,
            layout_candidates=args.layout_candidates,
            min_words=args.min_words,
        )
        print(f"Output dir: {result['output_dir']}")
        print(f"Puzzle: {result['package'].puzzle_path}")
        if not _report_problems(result, started):
            sys.exit(1)
    elif args.command == "render":
        with open(args.puzzle, encoding="utf-8") as handle:
            puzzle = json.load(handle)
        # Piped stdout on Windows defaults to a legacy code page that cannot encode e.g. Greek.
        sys.stdout.reconfigure(encoding="utf-8")
        print(render_puzzle_text(puzzle))
    elif args.command == "export":
        with open(args.puzzle, encoding="utf-8") as handle:
            puzzle = json.load(handle)
        text = export_ipuz(puzzle) if args.format == "ipuz" else export_html(puzzle, args.solution)
        with open(args.output, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
