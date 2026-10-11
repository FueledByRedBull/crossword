# Wikipedia-Seeded Crossword Generator

Generate a free-form thematic crossword from a Wikipedia seed article, with definition clues taken from article text and provenance for every clue.

## Requirements

- Python 3.10+

## Install

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -3 -m pip install -r requirements.txt
```

## Quick Start

```powershell
py -3 cli.py generate `
  --seed "Thermodynamics" `
  --lang en `
  --output-dir outputs\run_thermo `
  --offline

py -3 cli.py render --puzzle outputs\run_thermo\puzzle.json
```

`render` prints the grid as text with the numbered clues and a validity report.

## Export

```powershell
py -3 cli.py export --puzzle outputs\run_thermo\puzzle.json --format html --output outputs\run_thermo\puzzle.html --solution
py -3 cli.py export --puzzle outputs\run_thermo\puzzle.json --format ipuz --output outputs\run_thermo\puzzle.ipuz
```

- `html` writes one self-contained printable page: the numbered grid, the clues with enumerations, and a Sources list linking each article revision (`oldid`) under CC BY-SA 4.0. `--solution` adds the solved grid on its own page.
- `ipuz` writes an [ipuz](http://www.ipuz.org/) crossword cropped to the letters' bounding box, with the solution, enumerations and the Wikipedia sources in `copyright` and `notes`.

## How It Works

1. Score the articles linked from the seed and choose how many to keep (`K`). A linked article is on topic when its intro uses a word of the seed title (matched by stem) or the seed article names it at least twice. Off-topic articles are cut (`cut_off_topic` in `candidate_scores.csv`) and are selected only when no on-topic article is left.
2. Extract answers primarily from the titles of the on-topic linked articles, then from terms in the selected articles; a vocabulary gate checks there are enough.
3. Build one clue per answer: a definition clue from the article's first sentence (answer masked), or a fill-in-the-blank clue from article text as fallback, preferring a sentence of the seed article.
4. Place the clued answers in a free-form layout, where words cross wherever their letters match.
5. Package the puzzle with attribution.

## Layout Rules

A layout is valid only if:

- every word crosses at least one other word;
- all words form one connected piece;
- every run of two or more letters in a row or column is exactly one placed word.

`fill_status` is `complete` when the layout is valid and has at least `--min-words` words, otherwise `failed`. Layout quality is reported as `word_count`, `across_count`, `down_count`, `crossing_count`, `mean_crossings_per_word`, `words_with_two_plus_crossings`, `two_plus_crossings_ratio`, `letter_count`, `checked_letter_ratio`, `bbox_rows`, `bbox_cols` and `density`. `--layout-restarts` and `--layout-candidates` bound the search.

## Clue Quality

- Each entry has `clue_style` of `definition` or `fill_in_blank`; a fill-in-the-blank clue must contain the blank (`____`).
- A clue must not contain the answer or a morphological cousin of it.
- A clue has at most 18 words. A definition is cut only at a clean boundary; a fill-in-the-blank clue is a whole sentence, or the start of one cut at a clause boundary (`...` at the end only), and never opens with a word that points at an earlier sentence ("It", "This", "These", "But" ...).
- A definition every word of which is common among the pool's lead sentences ("An American physicist") is dropped as generic.
- A blank that is one part of a longer proper name is not a clue.
- Clue text must not repeat across entries.
- Every used clue carries provenance (article, revision id, sentence offset, `oldid` URL). Clue text and revision id always come from the same API response: the full text for the seed and the selected articles, the intro for every other page. A clue without such a revision id is not used.

## Key Outputs

For `generate --output-dir <dir>`:

- `<dir>\candidate_scores.csv`, `diagnostics_scores.json`
- `<dir>\selected_candidates.json`, `diagnostics_k.json`
- `<dir>\answer_candidates.csv`, `diagnostics_terms.json`, `diagnostics_vocab_gate.json`
- `<dir>\clues.csv`, `diagnostics_clues.json`
- `<dir>\grid.json`, `diagnostics_layout.json`
- `<dir>\puzzle.json`, `attribution.json`, `diagnostics_package.json`

`puzzle.json` fields: `seed_title`, `lang`, `selected_articles`, `selected_k`, `size`, `grid_cells` (`size` x `size`, `#` marks an empty cell), `across_entries`, `down_entries`, `fill_status`, `puzzle_status`, `layout` (the layout metrics above), `diagnostics`, `attribution`.

Each entry: `number`, `answer` (normalised uppercase), `display_answer`, `enumeration` (for example `5,5`), `clue`, `clue_style`, `clue_class`, `row`, `col`, `length`.

`puzzle_status` is one of `ok` (valid layout and every entry has a packageable clue), `insufficient_vocabulary`, `insufficient_clues`, `insufficient_quality` (layout failed).

## Main Commands

- `py -3 cli.py generate ...` full end-to-end pipeline
- `py -3 cli.py render --puzzle <path>` print a puzzle as text
- `py -3 cli.py export --puzzle <path> --format ipuz|html --output <path>` write an `.ipuz` file or a printable HTML page
- `py -3 cli.py <stage> ...` single pipeline stages (see `--help`)
- `py -3 scripts/bench.py ...` multi-seed benchmark runner
- `py -3 scripts/tune_weights.py ...` weight grid search

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

Greek articles are shorter and link less than English ones, so a thin seed fails the vocabulary gate: on 2026-10-11 `Αστρονομία` gave a complete 19-word puzzle, while `Θερμοδυναμική` stopped at `insufficient_vocabulary` with 9 clued answers. Known gaps: there is no Greek word-frequency list (the lexicon score is 0), a fill-in-the-blank clue only blanks the exact written form of the answer, and el.wikipedia.org rate-limits sooner than en.wikipedia.org, so a first online run waits on HTTP 429 a few times.

## Tuning Weights

Grid-search the candidate and K-selection weights across seeds. `summary.csv` records `fill_status`, `word_count` and `crossing_count` per combination.

```powershell
py -3 scripts/tune_weights.py `
  --offline `
  --seeds "Thermodynamics,Jazz,Ancient Rome,Quantum mechanics" `
  --output-dir outputs/weight_tuning `
  --candidate-weights "1.5,0.3,0.3,0.2;1.8,0.25,0.35,0.15" `
  --k-weights "1.0,1.0,1.0;1.2,1.1,1.0"
```

## Regression Coverage

The offline integration corpus covers the English cached path:

- `Thermodynamics`
- `Quantum mechanics`
- `Jazz`
- `Ancient Rome`

`Ancient Rome` links to 2,193 pages and is run with `--max-links 200`, which keeps an evenly spaced sample of the seed's links. The other seeds run uncapped. The corpus reads the local wiki cache (`data/cache/wiki`, not in git) and is skipped when the cache is missing.

Per-seed checks in `tests/test_generate_integration.py`:

- `fill_status == "complete"`, `puzzle_status == "ok"`, and the puzzle passes the validity checker
- `word_count` at least 16
- `mean_crossings_per_word` at least 1.9
- `two_plus_crossings_ratio` at least 0.5
- `checked_letter_ratio` at least 0.15
- definition clues are at least 50% of the entries
- no clue longer than 18 words, none that starts mid-sentence (`...`)
- no clue contains its answer, no duplicate clue text, no fill-in-the-blank clue missing its blank
- no used clue missing provenance

The benchmark runner writes per-seed results and a top-level `aggregate` block (with the release gates below) to `benchmarks_summary.json`. Leakage, blank and duplicate counts are computed over the entries in `puzzle.json`, so they describe the clues actually used.

```powershell
.\scripts\benchmark_suite.ps1 -Offline
```

runs the four corpus seeds from the cache through `scripts/bench.py`. A seed whose Gate A fixture carries `max_links` (Ancient Rome: 200) is run on that link sample; the other seeds use `--max-links`.

## Release Gates

- Gate A: every seed has a fixture in `data/benchmarks/gate_a.json` and an answer pool, with `central_recall >= 0.75` and `off_theme_count == 0`. Both are measured on the first source title of each row of `answer_candidates.csv`, against 15 central and 8 off-theme pages per seed that were written from subject knowledge before looking at any pool.
- Gate B: `fill_pass_rate >= 0.80` (share of seeds with `fill_status == "complete"`) and `valid_rate == 1.0` among seeds that produced a layout; an invalid layout never passes.
- Gate C: every seed graded, `average_leakage_rate <= 0.05`, total `blank_missing_count == 0`, total `duplicate_clue_count == 0`.
- Gate D: every seed graded and no used clue is missing provenance.
- `release_candidate_pass`: Gates A, B, C and D pass and `puzzle_ok_rate >= 0.80`.

A seed is graded when it has `word_count > 0` and a known `puzzle_status`; gates C and D fail closed otherwise.
