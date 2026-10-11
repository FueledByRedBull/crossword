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

1. Score the articles linked from the seed and choose how many to keep (`K`). A linked article is on topic when its intro uses a word of the seed title (matched by stem) or the seed article names it at least twice. Where the word-frequency list tells the title's words apart, only the rarest counts ("Rome", not "ancient", in "Ancient Rome"), and a derived form of a short word ("Roman") counts only in the intro's first sentence. Articles whose intro is on topic are selected first, then those the seed merely names. Off-topic articles are cut (`cut_off_topic` in `candidate_scores.csv`) and are selected only when nothing else is left.
2. Extract answers primarily from the titles of the on-topic linked articles, then from terms in the selected articles; a vocabulary gate checks there are enough. When there are not, a rescue ladder loosens the term filters and then widens the pool with pages linked from the on-topic selected articles (at most 100 new pages from at most 6 articles).
3. Build one clue per answer: a definition clue from the article's first sentence (answer masked), or a fill-in-the-blank clue as fallback. A fill-in comes only from the seed article, the answer's own page or a selected article whose intro is on topic, preferring the seed. An answer that is in the pool only because the seed names it takes a clean seed sentence before its own page's definition.
4. Place the clued answers in a free-form layout, where words cross wherever their letters match. The search fills the grid greedily, then keeps removing a few words and refilling (simulated annealing); it is deterministic for a given `--random-seed`.
5. Package the puzzle with attribution.

## Layout Rules

A layout is valid only if:

- every word crosses at least one other word;
- all words form one connected piece;
- every run of two or more letters in a row or column is exactly one placed word.

`fill_status` is `complete` when the layout is valid and has at least `--min-words` words, otherwise `failed`. Layout quality is reported as `word_count`, `across_count`, `down_count`, `crossing_count`, `mean_crossings_per_word`, `words_with_two_plus_crossings`, `two_plus_crossings_ratio`, `letter_count`, `checked_letter_ratio`, `bbox_rows`, `bbox_cols` and `density`. `--layout-candidates` caps how many clued answers are offered to the layout and `--layout-restarts` scales the search effort (25 search steps each).

Measured on 2026-10-11 over six real candidate lists and three random seeds, the search places 20.1 words on average with 21.9% of the letters crossed, in 2 to 9 seconds per layout (the earlier greedy search: 18.8 words and 20.8%). What limits density is the supply of short answers, not the grid or the search: a typical list offers 6 to 17 answers of six letters or fewer and the layouts already place about 80 to 90% of them.

## Clue Quality

- Each entry has `clue_style` of `definition` or `fill_in_blank`; a fill-in-the-blank clue must contain the blank (`____`).
- A clue must not contain the answer or a morphological cousin of it.
- A clue has at most 18 words. A definition is cut only at a clean boundary; a fill-in-the-blank clue is a whole sentence, or the start of one cut at a clause boundary (`...` at the end only) that still has its finite verb.
- A fill-in-the-blank clue never opens with a word that points at an earlier sentence ("It", "This", "But", "Hence," ...) and does not lean on one further in: a pronoun with nothing before it to refer to ("In 1870 he introduced ..."), "this pump" when no pump was named, "the former", "the following", "doing so".
- A definition is dropped as generic when it has no word that is rare among the pool's lead sentences (words of the seed title and nationalities do not count); when it opens on the answer's nationality and has fewer than two such words ("An American jazz drummer"); when its head noun names a kind of word without saying what the word means ("A term coined in the 1980s"); or when it opens on a vague quantifier ("Any of many different quantities").
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

Greek articles are shorter and link less than English ones, so a thin seed fails the vocabulary gate: on 2026-10-11 `Αστρονομία` gave a complete 18-word puzzle, while `Θερμοδυναμική` stopped at `insufficient_vocabulary`. For that seed the rescue ladder scored 82 more pages linked from the selected articles, 13 of them on topic, which raised the clued answers from 9 to 15 against the 30 the gate asks for. With `--gate-min 12 --min-words 8` the same seed gives a valid 12-word puzzle. Known gaps: the checks for a clue that points outside its sentence and for a generic definition are weaker in Greek than in English (no personal-pronoun check, no nationality or kind-of-word rule), there is no Greek word-frequency list (the lexicon score is 0), a fill-in-the-blank clue only blanks the exact written form of the answer, and el.wikipedia.org rate-limits sooner than en.wikipedia.org, so a first online run waits on HTTP 429 a few times.

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
- `word_count` at least 17
- `mean_crossings_per_word` at least 2.1
- `two_plus_crossings_ratio` at least 0.6
- `checked_letter_ratio` at least 0.18
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
