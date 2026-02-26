from __future__ import annotations

import argparse
import os

from src.pipeline import (
    run_candidate_scoring_stage,
    run_k_selection_stage,
    run_seed_ingestion_stage,
    run_term_extraction_stage,
    run_vocab_gate_stage,
    run_clue_extraction_stage,
    run_rescue_ladder,
    run_topology_selection_stage,
    run_csp_solve_stage,
    run_packaging_stage,
    run_generate_pipeline,
)


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
        help="Relevance threshold for KEEP status",
    )
    score_parser.add_argument(
        "--borderline-threshold",
        type=float,
        default=0.1,
        help="Relevance threshold for BORDERLINE status",
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
        help="Relevance threshold for KEEP status",
    )
    k_parser.add_argument(
        "--borderline-threshold",
        type=float,
        default=0.1,
        help="Relevance threshold for BORDERLINE status",
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
    k_parser.add_argument("--grid-size", type=int, default=15, help="Grid size for template fit")
    k_parser.add_argument(
        "--min-slot-len",
        type=int,
        default=3,
        help="Minimum slot length for template fit scoring",
    )

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
    term_parser.add_argument("--min-len", type=int, default=4, help="Minimum answer length")
    term_parser.add_argument("--max-len", type=int, default=12, help="Maximum answer length")
    term_parser.add_argument(
        "--min-alpha-ratio",
        type=float,
        default=0.8,
        help="Minimum alphabetic ratio for candidate term",
    )
    term_parser.add_argument("--min-df", type=int, default=2, help="Minimum doc frequency")
    term_parser.add_argument(
        "--nlp-backend",
        choices=["auto", "spacy", "nltk"],
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
        "--diagnostics",
        default="outputs/diagnostics_vocab_gate.json",
        help="Path to vocab gate diagnostics JSON output",
    )
    gate_parser.add_argument("--min-required", type=int, default=40, help="Minimum terms required")
    gate_parser.add_argument("--max-allowed", type=int, default=250, help="Maximum terms allowed")

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
    rescue_parser.add_argument("--gate-min", type=int, default=40, help="Minimum terms required")
    rescue_parser.add_argument("--gate-max", type=int, default=250, help="Maximum terms allowed")
    rescue_parser.add_argument("--min-len", type=int, default=4, help="Minimum answer length")
    rescue_parser.add_argument("--max-len", type=int, default=12, help="Maximum answer length")
    rescue_parser.add_argument(
        "--min-alpha-ratio",
        type=float,
        default=0.8,
        help="Minimum alphabetic ratio for candidate term",
    )

    topo_parser = subparsers.add_parser(
        "topology",
        help="Select best grid template based on term lengths",
    )
    topo_parser.add_argument("--seed", required=True, help="Seed Wikipedia article title")
    topo_parser.add_argument(
        "--lang",
        default="en",
        help="Language code for Wikipedia API (default: en, also supports el)",
    )
    topo_parser.add_argument(
        "--terms",
        default="outputs/answer_candidates.csv",
        help="Path to answer candidates CSV input",
    )
    topo_parser.add_argument(
        "--diagnostics",
        default="outputs/diagnostics_topology.json",
        help="Path to topology diagnostics JSON output",
    )
    topo_parser.add_argument("--size", type=int, default=15, help="Grid size")
    topo_parser.add_argument("--min-slot-len", type=int, default=3, help="Minimum slot length")
    topo_parser.add_argument(
        "--skip-gate",
        action="store_true",
        help="Skip vocabulary readiness gate",
    )
    topo_parser.add_argument("--gate-min", type=int, default=40, help="Minimum terms required")
    topo_parser.add_argument("--gate-max", type=int, default=250, help="Maximum terms allowed")

    csp_parser = subparsers.add_parser(
        "solve",
        help="Solve crossword grid with CSP",
    )
    csp_parser.add_argument("--seed", required=True, help="Seed Wikipedia article title")
    csp_parser.add_argument(
        "--lang",
        default="en",
        help="Language code for Wikipedia API (default: en, also supports el)",
    )
    csp_parser.add_argument(
        "--terms",
        default="outputs/answer_candidates.csv",
        help="Path to answer candidates CSV input",
    )
    csp_parser.add_argument(
        "--output",
        default="outputs/grid.json",
        help="Path to grid JSON output",
    )
    csp_parser.add_argument(
        "--diagnostics",
        default="outputs/diagnostics_csp.json",
        help="Path to CSP diagnostics JSON output",
    )
    csp_parser.add_argument("--size", type=int, default=15, help="Grid size")
    csp_parser.add_argument("--min-slot-len", type=int, default=3, help="Minimum slot length")
    csp_parser.add_argument(
        "--template",
        default=None,
        help="Template name to force (e.g., open or symmetric_sparse)",
    )
    csp_parser.add_argument(
        "--max-steps",
        type=int,
        default=20000,
        help="Maximum backtracking steps",
    )
    csp_parser.add_argument(
        "--max-restarts",
        type=int,
        default=2,
        help="Maximum solver restarts",
    )
    csp_parser.add_argument(
        "--template-trials",
        type=int,
        default=3,
        help="Number of top templates to try before selecting best fill",
    )
    csp_parser.add_argument(
        "--filler-path",
        default="data/lexicon/filler_words.txt",
        help="Optional filler word list for CSP (one word per line)",
    )
    csp_parser.add_argument(
        "--no-filler",
        action="store_true",
        help="Disable filler word list for CSP",
    )
    csp_parser.add_argument(
        "--filler-min-len",
        type=int,
        default=3,
        help="Minimum filler word length",
    )
    csp_parser.add_argument(
        "--filler-max-len",
        type=int,
        default=12,
        help="Maximum filler word length",
    )
    csp_parser.add_argument(
        "--filler-max-per-length",
        type=int,
        default=4000,
        help="Max filler words to keep per length bucket",
    )
    csp_parser.add_argument(
        "--filler-weight",
        type=float,
        default=0.05,
        help="Score weight for filler words (lower favors thematic words)",
    )
    csp_parser.add_argument(
        "--random-seed",
        type=int,
        default=13,
        help="Random seed for solver restarts",
    )
    csp_parser.add_argument(
        "--no-ac3",
        action="store_true",
        help="Disable AC-3 preprocessing",
    )
    csp_parser.add_argument(
        "--beam-width",
        type=int,
        default=32,
        help="Beam width for CSP search",
    )
    csp_parser.add_argument(
        "--no-local-repair",
        action="store_true",
        help="Disable local repair pass after beam search",
    )
    csp_parser.add_argument(
        "--repair-steps",
        type=int,
        default=300,
        help="Step budget for local repair",
    )
    csp_parser.add_argument(
        "--min-domain",
        type=int,
        default=1,
        help="Minimum domain size per slot; slots below are dropped",
    )
    csp_parser.add_argument(
        "--skip-gate",
        action="store_true",
        help="Skip vocabulary readiness gate",
    )
    csp_parser.add_argument("--gate-min", type=int, default=40, help="Minimum terms required")
    csp_parser.add_argument("--gate-max", type=int, default=250, help="Maximum terms allowed")

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
    generate_parser.add_argument("--borderline-threshold", type=float, default=0.1)
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
    generate_parser.add_argument("--min-slot-len", type=int, default=3)
    generate_parser.add_argument("--min-len", type=int, default=4)
    generate_parser.add_argument("--max-len", type=int, default=12)
    generate_parser.add_argument("--min-alpha-ratio", type=float, default=0.8)
    generate_parser.add_argument("--min-df", type=int, default=2)
    generate_parser.add_argument("--nlp-backend", choices=["auto", "spacy", "nltk"], default="auto")
    generate_parser.add_argument("--entity-type-scoring", action="store_true")
    generate_parser.add_argument("--gate-min", type=int, default=40)
    generate_parser.add_argument("--gate-max", type=int, default=250)
    generate_parser.add_argument("--clue-min-words", type=int, default=6)
    generate_parser.add_argument("--clue-max-words", type=int, default=12)
    generate_parser.add_argument("--diversity-cap", type=int, default=2)
    generate_parser.add_argument("--template", default=None)
    generate_parser.add_argument("--use-topology", action="store_true")
    generate_parser.add_argument("--max-steps", type=int, default=20000)
    generate_parser.add_argument("--min-domain", type=int, default=1)
    generate_parser.add_argument("--max-restarts", type=int, default=2)
    generate_parser.add_argument("--template-trials", type=int, default=3)
    generate_parser.add_argument("--random-seed", type=int, default=13)
    generate_parser.add_argument("--no-ac3", action="store_true")
    generate_parser.add_argument("--beam-width", type=int, default=32)
    generate_parser.add_argument("--no-local-repair", action="store_true")
    generate_parser.add_argument("--repair-steps", type=int, default=300)
    generate_parser.add_argument("--filler-path", default="data/lexicon/filler_words.txt")
    generate_parser.add_argument("--no-filler", action="store_true")
    generate_parser.add_argument("--filler-min-len", type=int, default=3)
    generate_parser.add_argument("--filler-max-len", type=int, default=12)
    generate_parser.add_argument("--filler-max-per-length", type=int, default=4000)
    generate_parser.add_argument("--filler-weight", type=float, default=0.05)
    generate_parser.add_argument("--skip-gate", action="store_true")
    generate_parser.add_argument("--no-rescue", action="store_true")

    return parser


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
            borderline_threshold=args.borderline_threshold,
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
            borderline_threshold=args.borderline_threshold,
            min_k=args.min_k,
            max_k=args.max_k,
            epsilon=args.epsilon,
            m=args.m,
            size=args.grid_size,
            min_slot_len=args.min_slot_len,
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
        )
        print(f"Terms written: {result.terms_path}")
        print(f"Diagnostics: {result.diagnostics_path}")
    elif args.command == "vocab-gate":
        result = run_vocab_gate_stage(
            seed_title=args.seed,
            lang=args.lang,
            diagnostics_path=args.diagnostics,
            terms_path=args.terms,
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
            gate_max=args.gate_max,
            min_len=args.min_len,
            max_len=args.max_len,
            min_alpha_ratio=args.min_alpha_ratio,
        )
        print(f"Rescue diagnostics: {args.diagnostics}")
    elif args.command == "topology":
        result = run_topology_selection_stage(
            seed_title=args.seed,
            lang=args.lang,
            terms_path=args.terms,
            diagnostics_path=args.diagnostics,
            size=args.size,
            min_slot_len=args.min_slot_len,
            require_gate=not args.skip_gate,
            gate_min=args.gate_min,
            gate_max=args.gate_max,
        )
        print(f"Topology diagnostics: {result.diagnostics_path}")
    elif args.command == "solve":
        result = run_csp_solve_stage(
            seed_title=args.seed,
            lang=args.lang,
            terms_path=args.terms,
            diagnostics_path=args.diagnostics,
            grid_path=args.output,
            size=args.size,
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
        )
        print(f"Puzzle: {result.puzzle_path}")
        print(f"Attribution: {result.attribution_path}")
        print(f"Diagnostics: {result.diagnostics_path}")
    elif args.command == "generate":
        if args.offline:
            os.environ["CROSSWORD_OFFLINE"] = "1"
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
            gate_max=args.gate_max,
            rescue=not args.no_rescue,
            clue_min_words=args.clue_min_words,
            clue_max_words=args.clue_max_words,
            diversity_cap=args.diversity_cap,
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
            skip_gate=args.skip_gate,
            use_topology=args.use_topology,
        )
        print(f"Output dir: {result['output_dir']}")
        print(f"Puzzle: {result['package'].puzzle_path}")


if __name__ == "__main__":
    main()
