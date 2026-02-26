# 🧩 Wikipedia-Seeded Thematic Crossword Generator

Generate coherent, educational crossword puzzles from any Wikipedia article. Give it a seed topic like *Thermodynamics* or *Jazz*, and it builds a complete puzzle — grid, clues, and all.

## How It Works

The system runs a multi-stage pipeline:

1. **Seed Graph Expansion** — Fetches outgoing links from the seed Wikipedia article (optional 2-hop expansion)
2. **Semantic Scoring** — Ranks candidates using TF-IDF cosine similarity, redundancy penalty, depth penalty, and backlink bonus (MMR-style)
3. **K Optimization** — Selects the optimal number of articles using a crossword-aware objective with diminishing-returns stopping
4. **Term Extraction** — Extracts crossword answer candidates via spaCy NLP (with nltk fallback), lead-section boldface signals, and quality filters
5. **Clue Generation** — Four-pass pipeline: extract → mask/trim → validate (leakage check) → diversity deduplication
6. **Grid Topology Selection** — Scores candidate grid templates against the word-length distribution
7. **CSP Fill** — Fills the grid using arc consistency (AC-3), MRV/degree variable ordering, and forward checking with restarts
8. **Provenance & Packaging** — Bundles the puzzle with full CC BY-SA attribution metadata

## Quick Start

### Prerequisites

- Python 3.10+

### Installation

```bash
pip install -r requirements.txt
```

### Usage

```bash
# Generate an English crossword from "Thermodynamics"
python cli.py generate \
  --seed "Thermodynamics" \
  --lang en \
  --grid-size 15 \
  --output outputs/thermo_15.json

# Generate a Greek crossword
python cli.py generate \
  --seed "Θερμοδυναμική" \
  --lang el \
  --grid-size 13 \
  --output outputs/thermo_el_13.json
```

**Key flags:**
| Flag | Description | Default |
|------|-------------|---------|
| `--seed` | Wikipedia article title | *(required)* |
| `--lang` | Language (`en` or `el`) | `en` |
| `--grid-size` | Grid dimension | `15` |
| `--expansion` | `one_hop_only` or `one_hop_plus_bounded_two_hop` | `one_hop_only` |
| `--output` | Output path | `outputs/puzzle.json` |

## Project Structure

```
src/
├── pipeline.py          # Stage orchestration
├── wiki_client.py       # MediaWiki API client with caching
├── wikidata_client.py   # Wikidata entity lookups
├── semantic.py          # TF-IDF vectorization & MMR scoring
├── k_selector.py        # Crossword-aware K optimization
├── term_extractor.py    # NLP-based answer extraction
├── clue_builder.py      # Four-pass clue pipeline
├── topology.py          # Grid template generation & scoring
├── crossword_csp.py     # Constraint solver (AC-3 + backtracking)
├── provenance.py        # Attribution capture
├── text_normalize.py    # Text cleaning & normalization
├── cache.py             # Disk cache layer
├── diagnostics.py       # Diagnostics emission
└── __init__.py

tests/                   # Unit & integration tests
scripts/
└── bench.py             # Benchmarking utility
cli.py                   # CLI entry point
PLAN.md                  # Detailed design document
requirements.txt         # Pinned dependencies
```

## Output

Each run produces:
- **`puzzle.json`** — Grid, clues, and fill status
- **`diagnostics.json`** — Scores, selection decisions, and solver trace
- **`candidate_scores.csv`** — Ranked article candidates
- **`k_selection_trace.csv`** — Marginal utility trace
- **`attribution.json`** — Per-clue Wikipedia revision provenance

## Dependencies

- [spaCy](https://spacy.io/) — NLP backbone for term extraction
- [mwparserfromhell](https://github.com/earwig/mwparserfromhell) — Wikipedia markup parsing
- MediaWiki API — Content and metadata (no scraping)

## License

Content derived from Wikipedia is used under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/). Attribution metadata is bundled with every generated puzzle.
