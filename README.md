# Wikipedia-Seeded Crossword Generator

Generate thematic crosswords from a Wikipedia seed article with provenance-backed clues and diagnostics.

## Requirements

- Python 3.10+
- Rust toolchain (optional, only for Rust CSP backend)

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Optional: Build Rust CSP Backend

```powershell
cd rust_csp
python -m pip install maturin
python -m maturin develop
cd ..
```

After this, `--use-rust` is available in `generate`, `solve`, and tuning/benchmark scripts.

## Quick Start

```powershell
python cli.py generate `
  --seed "Thermodynamics" `
  --lang en `
  --output-dir outputs\run_thermo `
  --offline
```

Rust backend (optional):

```powershell
python cli.py generate `
  --seed "Thermodynamics" `
  --lang en `
  --output-dir outputs\run_thermo_rust `
  --offline `
  --use-rust
```

## Current Quality Controls

The CSP stage accepts complete or strong partial fills. Current quality gates are:

- `fill_percent >= 0.70`
- `invalid_slots == 0`
- `filler_used_ratio <= 0.25`
- `clued_entry_ratio >= 0.90` (when clue set is available)
- no non-themed filler in long slots

Passing puzzles can still be `fill_status = "partial"` if they clear the gate and package cleanly as `puzzle_status = "ok"`.

The solve stage also carries a separate `preferred_fill_target = 0.85`. That is not a hard acceptance gate; it is the retry/rescue target that drives selection expansion, rescue-only `min_df` relaxation, and short-slot completion before packaging.

## Clue Quality

Clues are tagged into three classes throughout the pipeline:

- `source_backed`: extracted from a real cached article sentence with provenance
- `template_fallback`: article-backed fallback clue text when sentence extraction misses
- `synthetic_filler`: debug/search-only short-fill fallback class that is not packageable

Packaged puzzles may use bounded `template_fallback` clues, but they must carry page-level provenance. `synthetic_filler` clues do not count as source-backed coverage and are rejected from final packaging.

## Solver Behavior

- Two-phase solve:
  Phase A is a thematic prepass for long slots.
  Phase B is the full solve with quality-gated selection.
- `--use-topology` is a ranking hint for template order, not a hard template lock.
- Python and Rust CSP backends are kept in parity by direct fixture tests and the offline seed corpus.
- English filler vocabulary is ASCII A-Z only, filtered aggressively, and capped to short bridge lengths when clue answers are available.

## Tuning Weights

Run grid-search tuning across benchmark seeds:

```powershell
python scripts/tune_weights.py `
  --offline `
  --use-rust `
  --seeds "Thermodynamics,Jazz,Ancient Rome,Quantum mechanics" `
  --output-dir outputs/weight_tuning_v2 `
  --candidate-weights "1.5,0.3,0.3,0.2;1.8,0.25,0.35,0.15;1.3,0.35,0.25,0.25;1.6,0.4,0.2,0.1" `
  --k-weights "1.0,1.0,1.0,0.5,0.5;1.2,1.1,1.0,0.4,0.3;0.9,1.2,1.1,0.6,0.4;1.1,0.9,1.2,0.3,0.6" `
  --max-steps 80000 `
  --max-restarts 12 `
  --template-trials 5 `
  --beam-width 64 `
  --filler-max-per-length 1200 `
  --filler-weight 0.01
```

Expected runtime for that full matrix is typically 6-14 hours.

## Regression Coverage

The offline integration corpus currently covers:

- `Thermodynamics`
- `Quantum mechanics`
- `Jazz`
- `Ancient Rome`

The test suite runs that corpus on the Python backend and, when installed, the Rust backend. It checks per-seed gate clearance plus aggregate quality:

- average `fill_percent >= 0.73`
- average `filler_used_ratio <= 0.10`
- average `long_slot_theme_ratio >= 0.95`
- average `used_source_backed_entry_ratio >= 0.90`
- average `used_template_fallback_entry_ratio <= 0.10`
- average `packaged_synthetic_filler_count == 0`
- average `used_clue_provenance_missing_count == 0`

The benchmark runner writes these quality metrics into `benchmarks_summary.json` per seed and in the top-level `aggregate` block. That summary now tracks fill, filler pressure, long-slot theme usage, used-clue source-backed coverage, used template-fallback usage, packaged synthetic filler count, used-clue provenance completeness, and pass rates across a seed set.

## Key Outputs

For `generate --output-dir <dir>`:

- `<dir>\puzzle.json`
- `<dir>\grid.json`
- `<dir>\clues.csv`
- `<dir>\diagnostics_terms.json`
- `<dir>\diagnostics_vocab_gate.json`
- `<dir>\diagnostics_csp.json`
- `<dir>\diagnostics_package.json`
- `<dir>\attribution.json`

## Main Commands

- `python cli.py generate ...` full end-to-end pipeline
- `python cli.py solve ...` CSP stage only
- `python scripts/bench.py ...` multi-seed benchmark runner
- `python scripts/tune_weights.py ...` weight grid search
