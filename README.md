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
  --use-topology `
  --use-rust
```

Offline (cache-only):

```powershell
python cli.py generate `
  --seed "Thermodynamics" `
  --output-dir outputs\run_thermo_offline `
  --offline `
  --use-rust
```

## Current Quality Controls

The CSP stage now enforces quality gates before accepting a puzzle:

- `fill_percent >= 0.98`
- `invalid_slots == 0`
- `filler_used_ratio <= 0.25`
- `clued_entry_ratio >= 0.90` (when clue set is available)
- no non-themed filler in long slots

If any gate fails, solve status is marked as failed and packaging reports `insufficient_quality` or `insufficient_clues`.

## Solver Behavior

- Two-phase solve:
- Phase A: thematic prepass focused on long slots.
- Phase B: full solve with strict filler penalties, then quality-gated selection.
- Strict filler filtering:
- English solver vocabulary is ASCII A-Z only.
- Acronym-like and low-quality filler is filtered out.
- Defaults are conservative: `filler_max_per_length=1200`, `filler_weight=0.01`.

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
