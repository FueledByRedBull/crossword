# Wikipedia-Seeded Crossword Generator

Generates a free-form thematic crossword in English or Greek from one Wikipedia seed article. The answers are mostly the titles of the pages the seed links to, and each clue is a definition or a fill-in-the-blank sentence taken from article text, with the article, revision id and `oldid` link it came from.

It is a command-line tool, developed against a fixed set of cached seeds. There is no CI and there are no tagged releases.

## Requirements

- Python 3.10 to 3.13 (tested on 3.12 and 3.13). The pinned `mwparserfromhell==0.7.2` has no wheels for Python 3.14, which is untested.
- A Rust toolchain (`cargo`, any recent stable; built and tested with 1.94) for the layout search. It is built automatically on first use; to build it by hand run `cargo build --release --manifest-path layout-rs/Cargo.toml`.

The commands below are for Windows PowerShell.

## Install

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -3 -m pip install -r requirements.txt
```

`requirements.txt` also installs the two spaCy models (`en_core_web_sm`, `el_core_news_sm`) from GitHub releases, so the install needs network access.

## Quick Start

```powershell
py -3 cli.py generate `
  --seed "Thermodynamics" `
  --lang en `
  --output-dir outputs\run_thermo

py -3 cli.py render --puzzle outputs\run_thermo\puzzle.json
```

The first run fetches the seed's pages from the Wikipedia API, which can take several minutes (see Online Runs), and caches them under `data/cache/wiki` (not in git). Add `--offline` to run from the cache alone. `render` prints the grid as text with the numbered clues and a validity report. The start of its output; the words depend on the Wikipedia revision and on the code:

```text
Thermodynamics

  1  _  _  _  _  _  _  _     2  3  _  _  _
  _                       4     _
  5  _  _  _  _  _  6     _     _     7
  _                 _     8  _  _  _  _  _  _
  _           9  _  _  _  _           _
...
Across
1. ...
```

## Export

```powershell
py -3 cli.py export --puzzle outputs\run_thermo\puzzle.json --format html --output outputs\run_thermo\puzzle.html --solution
py -3 cli.py export --puzzle outputs\run_thermo\puzzle.json --format ipuz --output outputs\run_thermo\puzzle.ipuz
```

- `html` writes one self-contained printable page: the numbered grid, the clues with enumerations, and a Sources list linking each article revision (`oldid`) under CC BY-SA 4.0. `--solution` adds the solved grid on its own page.
- `ipuz` writes an [ipuz](http://www.ipuz.org/) crossword cropped to the letters' bounding box, with the solution, enumerations and the Wikipedia sources in `copyright` and `notes`.

## How It Works

1. Score the articles linked from the seed, keep the ones that are on topic, and choose how many to use (`K`).
2. Take the answers from the titles of those pages (a person by the surname), then from terms in the selected articles. A vocabulary gate checks there are enough; when there are not, a rescue ladder loosens the term filters and widens the pool.
3. Build one clue per answer: a definition from the article's first sentence with the answer masked, or a fill-in-the-blank sentence as fallback.
4. Place the clued answers in a free-form layout, where words cross wherever their letters match. The search is a small Rust program (`layout-rs`) that fills the grid greedily and then keeps removing a few words and refilling (simulated annealing), about 72,000 steps by default. For the same answers and `--random-seed` it gives the same layout on every platform, because it uses only exact arithmetic (its own random numbers and its own logarithm, no math library). The answers and their scores come from the stages before it, which use the platform's math library and the language model, so a whole run is only sure to repeat on one machine.
5. Package the puzzle with attribution.

The rules in full (what makes a page on topic, how a person is answered by the surname, how a definition is cut, which clues are rejected, what a valid layout is) are in [docs/how-it-works.md](docs/how-it-works.md). The design and the backlog are in [PLAN.md](PLAN.md).

## Key Outputs

For `generate --output-dir <dir>`:

- `<dir>\candidate_scores.csv`, `diagnostics_scores.json`
- `<dir>\selected_candidates.json`, `k_selection_trace.csv`, `diagnostics_k.json`
- `<dir>\answer_candidates.csv`, `diagnostics_terms.json`, `diagnostics_vocab_gate.json`
- `<dir>\clues.csv`, `diagnostics_clues.json`
- `<dir>\grid.json`, `diagnostics_layout.json`
- `<dir>\puzzle.json`, `attribution.json`, `diagnostics_package.json`

`puzzle.json` fields: `seed_title`, `lang`, `selected_articles`, `selected_k`, `size`, `grid_cells` (`size` x `size`, `#` marks an empty cell), `across_entries`, `down_entries`, `fill_status`, `puzzle_status`, `layout` (the layout metrics), `diagnostics`, `attribution`.

Each entry: `number`, `answer` (normalised uppercase), `display_answer`, `enumeration` (for example `5,5`), `clue`, `clue_style` (`definition` or `fill_in_blank`), `clue_class` (always `source_backed`), `row`, `col`, `length`.

`fill_status` is `complete` when the layout is valid and has at least `--min-words` words, otherwise `failed`. A layout is valid only if every word crosses at least one other word, all words form one connected piece, and every run of two or more letters in a row or column is exactly one placed word.

`layout` holds `word_count`, `across_count`, `down_count`, `crossing_count`, `mean_crossings_per_word`, `words_with_two_plus_crossings`, `two_plus_crossings_ratio`, `letter_count`, `checked_letter_ratio`, `bbox_rows`, `bbox_cols` and `density`. `--layout-candidates` caps how many clued answers are offered to the layout and `--layout-restarts` scales the search effort (750 search steps each).

`puzzle_status` is one of `ok` (valid layout and every entry has a packageable clue), `insufficient_vocabulary`, `insufficient_clues`, `insufficient_quality` (layout failed).

Each stage records what went wrong in the `errors` list of its diagnostics file: a page that could not be fetched, a cache miss in offline mode, a missing spaCy model. When the puzzle is not `complete` and `ok`, or any stage recorded an error, `generate` prints the two statuses, the number of distinct errors, the files that hold them and the first five messages to stderr. The exit code is 1 when the puzzle is not `complete` and `ok`, and 0 otherwise: errors recorded on the way to a good puzzle do not fail the run.

## Commands

- `py -3 cli.py generate ...` full end-to-end pipeline
- `py -3 cli.py render --puzzle <path>` print a puzzle as text
- `py -3 cli.py export --puzzle <path> --format ipuz|html --output <path>` write an `.ipuz` file or a printable HTML page
- `py -3 cli.py <stage> ...` single pipeline stages (`seed`, `rank`, `select-k`, `terms`, `vocab-gate`, `clues`, `rescue`, `layout`, `package`; see `--help`)
- `py -3 scripts/bench.py ...` multi-seed benchmark runner
- `py -3 scripts/tune_weights.py ...` weight grid search

`--offline` is an option of `generate`. The stage commands read the environment variable `CROSSWORD_OFFLINE=1` instead.

## Online Runs

Without `--offline`, pages missing from the cache are fetched from the Wikipedia API:

- in batches: 50 titles per lead-wikitext request and 20 per intro request (full article text is one page per request, for the seed and the selected articles only);
- paced: at most one request per second across the whole run;
- patiently: on HTTP 429 or 503 the client waits for the `Retry-After` time (between 30 and 300 seconds), tries again, and from then on doubles the gap between requests (up to 8 seconds).

On 2026-10-11 Wikipedia allowed this client about 10 requests a minute, so the first run of a seed with several hundred links takes several minutes and hits a few 429 waits before the pacing settles. Responses are cached under `data/cache/wiki`, so a second run of the same seed makes no requests. Set the `CROSSWORD_USER_AGENT` environment variable to a User-Agent that carries your contact details, as the Wikimedia User-Agent policy asks; the default identifies only the program.

## Greek

`--lang el` reads el.wikipedia.org. Answers are Greek capital letters without accents (Θερμότητα becomes `ΘΕΡΜΟΤΗΤΑ`); titles with letters of another alphabet are not answers. Definitions are cut from Greek lead sentences (`είναι`, `ήταν`, `ονομάζεται` ...), and the count of how often the seed names a page tolerates inflection endings.

```powershell
py -3 cli.py generate --seed "Αστρονομία" --lang el --output-dir outputs\run_astro
```

Greek articles are shorter and link less than English ones, so a thin seed does not reach the 30 clued answers the vocabulary gate asks for from its own links, and the rescue ladder widens the pool. el.wikipedia.org rate-limits sooner than en.wikipedia.org, so a first online run waits on HTTP 429 a few times. What is different in Greek, and the known gaps, are in [docs/how-it-works.md](docs/how-it-works.md#greek); the results on the thin Greek seeds are in [docs/evaluation.md](docs/evaluation.md#greek-seeds).

## Tuning Weights

Grid-search the candidate and K-selection weights across English seeds. `summary.csv` records `fill_status`, `word_count` and `crossing_count` per combination.

```powershell
py -3 scripts/tune_weights.py `
  --offline `
  --seeds "Thermodynamics,Jazz,Ancient Rome,Quantum mechanics" `
  --output-dir outputs/weight_tuning `
  --candidate-weights "1.5,0.3,0.3,0.2;1.8,0.25,0.35,0.15" `
  --k-weights "1.0,1.0,1.0;1.2,1.1,1.0"
```

## Tests and Benchmarks

```powershell
py -3 -m pip install -r requirements-dev.txt
py -3 -m pytest tests -q -k "not corpus"
cargo test --manifest-path layout-rs/Cargo.toml
```

The deselected test, the offline integration corpus, runs four cached seeds end to end and takes several minutes; it is skipped when the cache is missing. It and the benchmark runner (`scripts\benchmark_suite.ps1`), the release gates and the held-out seeds are described in [docs/evaluation.md](docs/evaluation.md).

## License

[MIT](LICENSE)

The word lists under `data/lexicon` derive from SCOWL; where each comes from and SCOWL's notice are in [data/lexicon/README.md](data/lexicon/README.md).
