# Evaluation and measurements

How the generator is tested and benchmarked, and the dated measurements behind its choices. Figures are as measured on the date given and are not refreshed by later runs unless a sentence says so.

## Regression coverage

The offline integration corpus covers the English cached path:

- `Thermodynamics`
- `Quantum mechanics`
- `Jazz`
- `Ancient Rome`

`Ancient Rome` links to 2,193 pages and is run with `--max-links 200`, which keeps an evenly spaced sample of the seed's links. The other seeds run uncapped. The corpus reads the local wiki cache (`data/cache/wiki`, not in git) and is skipped when the cache is missing.

Per-seed checks in `tests/test_generate_integration.py`:

- `fill_status == "complete"`, `puzzle_status == "ok"`, and the puzzle passes the validity checker
- `word_count` at least 20
- `mean_crossings_per_word` at least 2.1
- `two_plus_crossings_ratio` at least 0.6 (the share of words with two or more crossings ran from 0.68 to 1.00 over five random seeds of one answer list, and the floor sits under that whole range)
- `checked_letter_ratio` at least 0.2
- definition clues are at least 50% of the entries
- no clue longer than 18 words, none that starts mid-sentence (`...`)
- no clue contains its answer, no duplicate clue text, no fill-in-the-blank clue missing its blank
- no used clue missing provenance

The benchmark runner writes per-seed results and a top-level `aggregate` block (with the release gates below) to `benchmarks_summary.json`. Leakage, blank and duplicate counts are computed over the entries in `puzzle.json`, so they describe the clues actually used.

```powershell
.\scripts\benchmark_suite.ps1 -Offline
```

runs the four corpus seeds from the cache through `scripts/bench.py`; `-Seeds`, `-Lang` and `-Fixture` choose other seeds, their language and their Gate A fixture. A seed whose Gate A fixture carries `max_links` (Ancient Rome: 200) is run on that link sample; the other seeds use `--max-links`.

## Release gates

The gates are defined in [PLAN.md](../PLAN.md) section 3. `scripts/bench.py` computes them into the `aggregate` block of `benchmarks_summary.json` (`gate_a_pass`, `gate_b_pass`, `gate_c_pass`, `gate_d_pass`, `release_candidate_pass`), from `central_recall`, `off_theme_count`, `fill_pass_rate`, `valid_rate`, `average_leakage_rate`, `blank_missing_count`, `duplicate_clue_count`, `puzzle_ok_rate` and `used_clue_provenance_missing_count`.

Gate A is measured on the first source title of each row of `answer_candidates.csv`, against 15 central (14 for Cryptography) and 8 off-theme pages per seed that were written from subject knowledge before looking at any pool. The fixture holds the four corpus seeds and six former held-out seeds, English and Greek; `scripts/bench.py` takes one `--lang`, so the seeds of one language are run together.

## Held-out seeds

The figures in this section are as measured on 2026-10-12, before the layout search moved to Rust and before people were answered by the surname without further evidence. On 2026-10-13 the 21 cached seeds (the ten of this section, the four corpus seeds, Impressionism, `Αστρονομία` and the five thin Greek seeds) average 24.3 words, 31.6 crossings and 26.2% of the letters crossed, all 21 complete; 38 people in their pools have no clue (64 before this round).

The relevance and clue rules were written while reading the output of a few seeds, so numbers measured on those seeds flatter the rules. `data/benchmarks/held_out.json` is a Gate A fixture for seeds that no rule was derived from: 15 central and 8 off-theme pages each, chosen from the seed's link list before any output for the seed existed. A seed that becomes the example for a rule moves to `gate_a.json`, and a fresh seed replaces it.

### First round

On 2026-10-12 seven seeds were run that had never been run before. Measured before anything was changed for them, they showed what carries over from the development seeds and what does not:

- Relevance and layout carried over in English: Evolution, Chess, Volcano, Cryptography and Opera passed Gate A and averaged 26.6 words with 26.9% of the letters crossed.
- Relevance did not carry over in Greek: `Μαθηματικά` and `Φιλοσοφία` failed Gate A with 9 and 11 of 15 central pages in the pool.
- The clue rules carried over less well than the development grids suggest. Reading every grid clue, 17 of 175 were clearly weak, against 2 of 149 in the development grids: the wrong sense of the word (Genome clued as "A monthly peer-reviewed scientific journal ..."), a sentence that leans on one the solver does not see ("Sparing no detail, he listed a wealth of ____ examples of past life forms"), a timeline entry, a definition cut one word late ("... in cooperation"), and Greek answers that are an adverb, an infinitive and a lower-case plural instead of names. Another 26 were bland, against 10 ("An American historian, journalist, and writer" for Kahn).

Three changes came out of that reading. The fill-in ranking was designed on Evolution and Chess and checked on the other five before their clues were read. The accent rule closes a leak that the ranking had put into the Volcano grid. Together they took the clearly weak clues of those seven grids from 17 to 12 of 175. The third is to relevance: `Μαθηματικά`, `Φιλοσοφία` and Cryptography were losing on-topic pages to a fixed threshold on how close a page's text is to the seed's (Euclid, whom the seed names six times, among them), and that cut is gone. Those six seeds are development seeds now, with their fixtures in `gate_a.json`.

### Blind check of the relevance change

Three fresh seeds were chosen for it: `Φυσική`, `Μουσική` and Photography. Their fixtures were written from the link lists alone, they were run once on the old code, and the recall the change should give them was written down from the old run's candidate scores (13, 15 and 14 of 15) before the new runs were read. It came out as predicted:

| Seed | Words | Letters crossed | Definition clues | Central pages in the pool | Off-theme pages |
|---|---|---|---|---|---|
| Φυσική | 25 → 26 | 27.3% → 25.8% | 68% → 62% | 12 → 13 of 15 | 0 → 0 |
| Μουσική | 20 → 20 | 22.1% → 24.8% | 65% → 70% | 12 → 15 of 15 | 0 → 0 |
| Photography | 21 → 22 | 22.4% → 26.3% | 81% → 82% | 13 → 14 of 15 | 0 → 0 |

All three passed Gate A before the change as well. Of their eight missing central pages, five were on-topic pages under the old threshold (among them Max Planck, Stravinsky, Wagner and Louis Daguerre, whom the seed names three times) and three are cut as off topic, which the change does not touch.

The seven first-round seeds, before and after the same change:

| Seed | Words | Letters crossed | Definition clues | Central pages in the pool | Off-theme pages |
|---|---|---|---|---|---|
| Evolution | 24 → 24 | 25.0% → 25.0% | 54% → 54% | 13 → 13 of 15 | 0 → 0 |
| Chess | 26 → 26 | 25.2% → 25.2% | 54% → 54% | 14 → 14 of 15 | 0 → 0 |
| Volcano | 28 → 29 | 29.2% → 27.2% | 82% → 83% | 14 → 14 of 15 | 0 → 0 |
| Cryptography | 27 → 26 | 28.3% → 28.0% | 81% → 92% | 12 → 13 of 14 | 0 → 0 |
| Opera | 28 → 26 | 26.8% → 27.5% | 46% → 69% | 14 → 14 of 15 | 0 → 0 |
| Μαθηματικά | 17 → 23 | 21.7% → 20.0% | 71% → 61% | 9 → 15 of 15 | 1 → 1 |
| Φιλοσοφία | 25 → 24 | 24.4% → 26.6% | 72% → 83% | 11 → 14 of 15 | 0 → 0 |

### What still does not pass

- `Μαθηματικά` still fails Gate A. All 15 central pages are in its pool now, but so is `Βιολογία`, whose intro passes the seed-word test.
- Clue quality did not improve with relevance. Clearly weak grid clues before and after the relevance change: 12 of 175 and 14 of 178 on the seven first-round seeds (both new ones are Greek), 6 of 66 and 6 of 68 on the three fresh seeds, 2 of 149 and 1 of 153 on the development seeds. That is 20 of 390 and 21 of 399 together, the comparison the change was accepted on. Greek is the worst: 5 of the 26 clues of `Φυσική` (two sentences that open on "Συνεπώς" or "Παρόμοια", a broken definition, "Έργο" clued in the sense of a written work, a definition of "Χημεία" that fits any natural science).
- A small change to the pool re-lays the whole grid: one page more in the Thermodynamics pool swapped 7 of its 22 answers, and Opera's share of definition clues went from 46% to 69% with one answer different in a pool of 150. Single-grid numbers in the tables above carry that noise.
- The held-out seeds today are Opera, `Φυσική`, `Μουσική` and Photography. All four have been read once, so the next rule change needs fresh seeds for its blind check.

```powershell
.\scripts\benchmark_suite.ps1 -Offline -Fixture data/benchmarks/held_out.json -Seeds "Opera,Photography" -OutputDir outputs/held_out_en
.\scripts\benchmark_suite.ps1 -Offline -Fixture data/benchmarks/held_out.json -Lang el -Seeds "Φυσική,Μουσική" -OutputDir outputs/held_out_el
```

Without `-Offline` the first run of a seed fetches its pages (see Online Runs). The six seeds that moved to `gate_a.json` run the same way without `-Fixture`: `-Seeds "Evolution,Chess,Volcano,Cryptography"`, and `-Lang el -Seeds "Μαθηματικά,Φιλοσοφία"`, which fails Gate A on `Βιολογία`.

## Measurements

### Layout search

Measured on 2026-10-13 over the candidate lists of 21 seeds and three random seeds, the search places 22.4 words on average with 29.3 crossings and 25.0% of the letters crossed (26.6 words and 34.2 crossings on the lists of 80 candidates), in about 4 seconds of CPU time for 80 candidates. The Python search it replaced ran about 50 times fewer steps a second and placed 21.5 words with 26.9 crossings in the same time. More steps keep paying, slowly: 24,000, 72,000 and 240,000 steps give 22.2, 22.4 and 22.5 words and 28.3, 29.3 and 29.7 crossings. Other kinds of move were measured at equal steps in the Python search and not adopted, because none scored higher: removing a word together with the words that cross it, removing every word in a window, late-acceptance and threshold acceptance in place of annealing, and several short runs continued from the best one; keeping the best of two refills gained less than the extra time it took. Giving the answers' scores less weight in the objective was measured again in the Rust search, on 63 layouts: a weight of 2 instead of 3 places 0.2 more words and 0.3 more crossings for 0.1% less total score of the placed answers, and a weight of 1 places 0.2 more words and 0.6 more crossings for 1.6% less. About a third of the layouts that change lose words (10 of 29 at weight 2, 12 of 30 at weight 1), so the gain is no larger than the difference between two random seeds and the weight stays at 3. Single layouts differ by one or two words from one random seed to the next.

### Answers named by the surname

Surnames are where the short answers come from. Over six real seeds at the default settings the puzzles average 19.5 words with full titles only, 21.3 when a page is answered by the name the seed uses, and 22.8 when a person is also answered by the surname their own page uses (Thermodynamics: 19 to 22 words, Impressionism: 20 to 23), with 24.0% of the letters crossed and 72% definition clues. People are up to 57% of the entries of those grids. Since on-topic pages are no longer cut for low similarity to the seed text, the same six seeds average 23.2 words with 23.7% of the letters crossed and 74% definition clues, and people are up to 67% of a grid (Impressionism: 16 of 24). A short name of a page that is not about a person is clued by a seed sentence, never by a definition.

### Other names of a page

Other names of a page were measured on 2026-10-12 and not adopted. A bold term of a page's lead that the seed article uses alone at least twice offered 12 names over seven seeds ("bass", "drums", "pop", "punk", "gens", but also "not" for Probability and "density") and added 0.5 words and 0.6 points of crossed letters over 18 layouts, no more than the difference between two random seeds. Wikidata aliases gave 283 candidates, 44 of them names a solver would accept; the rest were initials, misspellings, plurals and other languages. Taking a bold lead term only when it is also a Wikidata alias added 1.4 words and 1.3 points, but needs Wikidata at run time and puts TIQM, RQM, MWI and QKD in the grid beside BASS, SAX and PUNK.

### First online run of a seed that needed the backlink step

The step that widens a thin seed with the pages that link to it was first run in English on 2026-10-13, on `Bryology`: its first run made 63 requests in six minutes (two 429 waits) and scored 213 linking pages. With the rules as they are now it gives a complete 30-word puzzle with 41 crossings; 27 of the 30 answers are bryologists, because few other answers have a clue.

### Greek seeds

Greek articles are shorter and link less than English ones, so a thin seed does not reach the 30 clued answers the vocabulary gate asks for from its own links. On 2026-10-13 five such seeds gave complete puzzles through the rescue ladder: `Θερμοδυναμική` 20 words, `Οπτική` 27 and `Γλυπτική` 23 with the pages that link to them, `Ακουστική` 19 after pages linked from its selected articles were added as well, and `Σεισμολογία` 13 from a thin pool of 17 clued answers that no step could widen (the run goes on because it has at least `--min-words`, 12; three on-theme pages are cut there, because words such as "σεισμός" share too little with "σεισμολογία", and matching on the part before -λογία was measured and let noise into other seeds). On those seeds between a fifth and three fifths of the answers offered to the layout come from linking pages that use the seed's word only after their first sentence (19 of 40 for `Θερμοδυναμική`, 26 of 42 for `Γλυπτική`); without them the five grids would have 16, 22, 13, 17 and 12 words. Some of those pages mention the subject only in passing (`ΠΥΡΑΝΟΖΗ` and `ΛΕΙΖΕΡ` in the `Θερμοδυναμική` grid, `ΦΑΣΗΠΑΝΙΔΑΣ` in `Σεισμολογία`). Nothing in the cached text tells them from the on-theme ones: the number of uses, the part of speech, being named by an on-topic page and the similarity of the text were each measured and dropped as many on-theme pages as off-theme ones or more; only a use that is nothing but an adverb is a safe cut. `Μαθηματικά` and `Μουσική` needed the ladder to reach 30 clued answers while on-topic pages were still cut for low similarity; they now reach 60 and 33 without it.
