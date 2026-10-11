# Wikipedia-Seeded Thematic Crossword Generator

This document is target scope and backlog, not a statement that every item below is implemented or continuously validated today.

## 1. Goal and Scope

Build an end-to-end system that generates coherent, educational crossword puzzles from a single Wikipedia seed article (example: `Thermodynamics`) by:

1. Expanding to candidate articles from Wikipedia links.
2. Measuring semantic drift while controlling thematic creep.
3. Choosing article count `K` from data (not a fixed constant).
4. Extracting high-quality answers and robust clues.
5. Laying the clued answers out in a free-form crossword grid where words cross.

The system should optimize for thematic coherence, layout quality, clue quality, and reproducibility.

## 2. Product Requirements

### Functional Requirements

1. Accept a seed Wikipedia page title and grid parameters (`size`, layout search controls).
2. Fetch outgoing links via MediaWiki API (no scraping).
3. Optionally expand to bounded 2-hop candidates.
4. Fetch article content and metadata for seed + candidates.
5. Compute relevance, redundancy, depth, and backlink-aware selection scores.
6. Select coherent-yet-diverse articles and derive optimal `K`.
7. Extract, filter, and score crossword answer candidates using an explicit NLP backend strategy (`spaCy` default, `nltk` fallback).
8. Support a `--lang` flag (`en` default, `el` for Greek) that switches the Wikipedia base URL, spaCy model, alphabet filter, and accent normalization profile.
9. Generate clues: definition clues from each article's first sentence, with fill-in-the-blank clues from article text as fallback, validated by a strict four-pass pipeline.
10. Lay out the clued answers in a free-form grid (no templates, no filler words).
11. Render a puzzle as text (`py -3 cli.py render --puzzle <path>`).
12. Emit puzzle, diagnostics, and attribution/provenance artifacts.

### Non-Functional Requirements

1. Deterministic outputs with fixed random seed.
2. Explainable intermediate scores and keep/cut decisions.
3. End-to-end latency target of 2-5 minutes per puzzle on commodity hardware.
4. Resilient API layer with caching, retries, and resumable stages.
5. Licensing-safe provenance stored at clue extraction time.
6. Cache-first architecture from day one:
   - Cache every MediaWiki response on disk keyed by `(endpoint, params_hash)`.
7. Reproducible NLP stack:
   - Pin `spaCy` and language-specific model versions (`en_core_web_*`, `el_core_news_*`) exactly in dependency lock files.

## 3. Success Metrics and Release Gates

### Core Metrics

1. Theme coherence:
   - Mean similarity of selected articles to seed.
   - Pairwise coherence of selected set.
2. Controlled diversity:
   - Novel information gain per selected article.
3. Layout quality:
   - Usable answer pool size.
   - Share of seeds with a valid layout (`fill_status == "complete"`).
   - Crossings per word, share of words with two or more crossings, checked-letter ratio.
4. Clue quality:
   - Clue leakage rate (exact term or morphological cousin appears in clue).
   - Share of definition clues versus fill-in-the-blank clues.
   - Readability proxy (6-12 word clues preferred).
5. Operational quality:
   - Runtime distribution.
   - API retry/failure rates.
6. Attribution completeness:
   - Percentage of clues with full source metadata.

### Release Gates

1. Gate A: Selection keeps thematically central pages and suppresses drift on benchmark seeds. Every seed needs a fixture in `data/benchmarks/gate_a.json` and an answer pool, with `central_recall >= 0.75` and `off_theme_count == 0`, measured on the first source title of each row of `answer_candidates.csv`.
2. Gate B: `fill_pass_rate >= 0.80` (share of seeds with `fill_status == "complete"`) and `valid_rate == 1.0` among seeds that produced a layout. An invalid layout never passes.
3. Gate C: every seed graded, average leakage rate <= 5%, no duplicate clue text, no fill-in-the-blank clue missing its blank.
4. Gate D: every seed graded and every used clue has attribution metadata.

A seed is graded when it has `word_count > 0` and a known `puzzle_status`; Gates C and D fail closed otherwise. The release candidate passes when Gates A, B, C and D pass and `puzzle_ok_rate >= 0.80`.

## 4. End-to-End Pipeline

1. `Seed Graph Expansion`:
   - Collect 1-hop outgoing links from seed.
   - Optionally sample bounded 2-hop links for additional vocabulary.
2. `Content + Metadata Ingestion`:
   - Fetch text for seed/candidates.
   - Fetch revision metadata and Wikidata IDs.
   - Capture lead-section boldface term signals for high-precision term priors.
3. `Semantic + Structural Scoring`:
   - Compute relevance, redundancy, depth penalty, backlink bonus.
4. `K Optimization`:
   - Add candidates until crossword-aware marginal utility plateaus.
5. `Term Extraction + Filtering`:
   - Take answers primarily from the titles of the most relevant linked articles, then from extracted noun phrases/entities, and apply quality filters.
6. `Clue Generation`:
   - Definition clue from the article's first sentence; fill-in-the-blank as fallback.
   - Four-pass extract/mask-trim/validate/diversity pipeline.
7. `Free-Form Layout`:
   - Place clued answers so they intersect where letters match (see 5.6).
8. `Evaluation + Packaging`:
   - Produce puzzle artifacts, diagnostics, and provenance bundle.
   - Emit `diagnostics.json` from the first runnable stage, even before full pipeline completion.

## 5. Detailed Algorithm Plan

### 5.1 Step 1: Seed Graph Expansion

#### Inputs

1. Seed title (`Thermodynamics`).
2. Expansion policy:
   - `one_hop_only` or `one_hop_plus_bounded_two_hop`.
3. Candidate cap to control runtime.

#### Process

1. Check disk cache before each API call using key `(endpoint, params_hash)`.
2. Resolve normalized seed page and page ID.
3. Fetch all outgoing links (`prop=links`, continuation handling).
4. Optional 2-hop expansion:
   - Expand only top 1-hop pages by preliminary relevance.
   - Keep strict budget to avoid topic explosion.
5. Assign link-depth feature:
   - Depth 1 for direct links.
   - Depth 2 for expanded links.
6. Fetch backlink signal (`list=backlinks`) to check whether candidate links back to seed.
7. Persist raw and normalized responses on cache miss for repeatable downstream iteration.

#### Outputs

1. Candidate table with:
   - `title`, `page_id`, `depth`, `links_back_to_seed`.

### 5.2 Step 2: Semantic Drift and Selection Scoring

#### Vectorization

1. Baseline: TF-IDF on seed + candidate corpus.
2. Optional backend: lightweight embeddings for robustness.

#### Core Scores

1. Relevance:
   - `rel(i) = cosine(seed_vec, cand_vec_i)`.
2. Redundancy for diversity:
   - `red(i) = max cosine(cand_vec_i, selected_vec_j)`.
3. Depth penalty:
   - `depth_pen(i) = 0` for depth 1, positive for depth 2.
4. Backlink bonus:
   - `back(i) = 1` if candidate links back to seed, else `0`.

#### Combined Ranking (MMR + graph structure)

`score(i) = a * rel(i) - b * red(i) - c * depth_pen(i) + d * back(i)`

#### Weight Defaults (BASELINE_V1)

| Weight | Default | Rationale |
|--------|---------|--------------------------------------------|
| `a`    | 1.0     | Relevance is the primary signal |
| `b`    | 0.5     | Redundancy penalty is half as strong |
| `c`    | 0.3     | Depth is a light penalty |
| `d`    | 0.2     | Backlink is a light bonus |

Document as `BASELINE_V1` in config. Tune by hand against benchmark seeds before running a grid search.

Notes:

1. Keep depth and backlink as soft features, not hard gates.
2. Depth penalty only differentiates if 2-hop expansion is enabled.
3. Retain borderline set for later vocabulary rescue.

#### Outputs

1. Ranked candidates with reason codes (`KEEP`, `BORDERLINE`, `CUT`).
2. Drift diagnostics table for explainability.

### 5.3 Step 3: Crossword-Aware K Optimization

#### Objective

Choose `K` with diminishing returns:

`J(K) = w1*coherence + w2*diversity + w3*term_quality`

#### J(K) Weight Defaults (BASELINE_V1)

`w1 = w2 = w3 = 1.0`. Rationale: equal weight on the three quality signals.

#### Procedure

1. Add candidates in descending selection score.
2. Recompute `J(K)` incrementally.
3. Stop when marginal gain:
   - `delta(K) = J(K) - J(K-1) < epsilon` for `m` consecutive additions.
4. Enforce min/max `K` bounds.

#### Thin Pool Rescue Ladder

If `J(K)` plateaus but the vocabulary readiness gate is not satisfied, apply the following steps in order, stopping as soon as the gate passes:

1. Loosen `min_df` to `1` (from adaptive default).
2. Promote up to 3 `BORDERLINE` candidates into the selected set.
3. Enable 2-hop expansion if not already active.
4. Reduce minimum answer length from `4` to `3`.
5. If all four steps fail: emit `puzzle_status: insufficient_vocabulary` and halt gracefully.

#### Outputs

1. Selected article set.
2. Chosen `K`.
3. Marginal-gain trace for debugging.
4. Rescue ladder steps taken (if any), for diagnostics.

### 5.4 Step 4: Term Extraction and Answer Quality Filtering

#### Extraction

1. Primary backend: `spaCy` noun chunks + entity spans (default for speed/quality balance on Wikipedia prose).
2. Fallback backend: `nltk` chunker behind a feature flag when `spaCy` models are unavailable.
3. Wikipedia structural signal:
   - Parse lead-section boldface terms (first-definition convention) and add a high-precision prior.
4. Merge NLP and structural candidates into a unified pool.
5. Normalize (case, punctuation, inflection handling).

#### Backend and Signal Policy

| `--lang` | spaCy model          | Alphabet filter      | Accent normalization          |
|----------|----------------------|----------------------|-------------------------------|
| `en`     | `en_core_web_sm`     | `[A-Za-z]`           | None                          |
| `el`     | `el_core_news_sm`    | `[Α-Ωα-ω]`           | Strip diacritics for grid fill, keep accented form for clue text |

Additional notes:

1. Optional quality profile: upgrade to `en_core_web_md` / `el_core_news_md` when higher recall is needed.
2. Greek leakage check must rely on spaCy's Greek lemmatizer rather than simple stemming, due to high inflection.
3. Structural boost: terms found in lead boldface receive a score bonus, not an auto-accept.
4. Provenance tags on each candidate: `source_method = spacy|nltk|lead_bold|hybrid`.
5. Reproducibility requirement:
   - Pin exact `spaCy` and model versions in `requirements.txt`/lock file and log active versions in diagnostics.

#### Quality Filters

1. Basic crosswordability:
   - Length range (default 4-15), high alphabetic ratio.
2. Wikidata entity type preference:
   - Prioritize scientific concept, physical quantity, unit of measurement.
3. Frequency floor across selected articles:
   - Prefer terms appearing in at least `min_df` selected pages.
   - Recommended adaptive default: `min_df = max(2, ceil(0.2*K))`.
4. Answer shape scoring:
   - Penalize low-crossability letter patterns early.
   - Treat shape as a score penalty, not a hard reject.
5. Composite answer score:
   - Combine theme relevance, entity-type preference, frequency support, lead-bold prior, and shape penalty.

#### Outputs

1. Scored answer candidate list with source mentions.

### 5.5 Step 5: Clue Quality Pipeline (Four Passes)

#### Pass 1: Extract

1. Pull sentence containing candidate answer term.
2. Keep top sentence candidates per answer.

#### Pass 2: Mask + Trim

1. Mask answer surface form in clue text.
2. Trim to a single clause where possible.
3. Target concise clues (ideal <=12 words).

#### Pass 3: Validate

1. Reject clues containing morphological cousins of answer (lemma/stem check).
2. Reject clues where masked slot is grammatical subject when this makes clue trivial.
3. Reject clues below 6 words as too vague.
4. Down-rank awkward or self-referential constructions.

#### Pass 4: Diversity Deduplication

1. Bucket all puzzle clues by their first 3 tokens.
2. Enforce at most 2 clues per bucket across the full puzzle.
3. Additionally bucket by structural pattern: definitional clues (starts with "The", contains "of") capped at 3 per puzzle.
4. For clues that fail diversity checks, re-run extraction against the next-best candidate sentence.

#### Outputs

1. One canonical clue per answer plus confidence score.

### 5.6 Step 6: Free-Form Layout

Implemented in `src/layout.py`. Words are placed on a `size` x `size` grid so that they intersect where letters match; there are no templates, no black-square patterns, and no filler words.

#### Validity Rules

1. Every word crosses at least one other word.
2. All words form one connected piece.
3. Every run of two or more letters in a row or column is exactly one placed word.

#### Search

1. Start from one of a few high-scoring long answers and greedily add the placement with the most crossings (bounding-box growth and answer score break ties).
2. Repeat for `layout_restarts` restarts with a different start word and noise; keep the layout with the most words, then most crossings. `layout_candidates` caps how many clued answers are offered to the layout.

#### Outputs

1. `grid.json` and `diagnostics_layout.json`.
2. `fill_status: complete | failed` (`complete` means valid and at least `min_words` words).
3. `validity_errors` (empty when valid) and the layout metrics: `word_count`, `across_count`, `down_count`, `crossing_count`, `mean_crossings_per_word`, `words_with_two_plus_crossings`, `two_plus_crossings_ratio`, `letter_count`, `checked_letter_ratio`, `bbox_rows`, `bbox_cols`, `density`.

### 5.7 Step 7: Rendering

`py -3 cli.py render --puzzle <path>` prints a `puzzle.json` as a text grid with numbered clues and a validity report.

### 5.8 Step 8: Attribution and Provenance

Store provenance during extraction (not post hoc):

1. `article_title`
2. `revision_id` (`revid`)
3. `sentence_offset` or span index
4. `page_id`
5. `oldid` URL for stable citation

This is required for reliable CC BY-SA attribution in published outputs.

## 6. Module Layout and Interfaces

1. `src/wiki_client.py`:
   - `fetch_links`, `fetch_backlinks`, `fetch_page_content`, `fetch_revid`, `fetch_lead_markup`.
2. `src/text_normalize.py`:
   - Cleaning, tokenization, phrase normalization.
3. `src/semantic.py`:
   - Vectorization, cosine, MMR+graph scoring.
4. `src/k_selector.py`:
   - Objective and marginal stopping logic.
5. `src/term_extractor.py`:
   - Candidate term extraction (`spaCy` primary, `nltk` fallback), lead-bold integration, and scoring.
6. `src/clue_builder.py`:
   - Four-pass clue pipeline (including diversity) and leakage checks.
7. `src/vocab_gate.py`:
   - Vocabulary readiness gate.
8. `src/layout.py`:
   - Free-form layout search, validity checks, layout metrics.
9. `src/render.py`:
   - Text rendering of `puzzle.json`.
10. `src/provenance.py`:
    - Attribution capture and output validation.
11. `src/evaluation.py`:
    - Benchmark summaries and release gates.
12. `src/pipeline.py`:
    - Stage orchestration.
13. `cli.py`:
    - User entry point (`generate`, `render`, per-stage commands).

## 7. Data Models

### Candidate Article Record

1. `page_id: int`
2. `title: str`
3. `depth: int`
4. `links_back_to_seed: bool`
5. `rel_score: float`
6. `red_score: float`
7. `selection_score: float`
8. `status: keep|borderline|cut`
9. `reason: str`

### Answer Candidate Record

1. `answer: str`
2. `normalized_answer: str`
3. `length: int`
4. `theme_score: float`
5. `entity_type_score: float`
6. `crosswordability_score: float`
7. `shape_penalty: float`
8. `article_frequency: int`
9. `lead_bold_signal: bool`
10. `source_method: str`
11. `source_mentions: list`

### Clue Record

1. `answer: str`
2. `clue_text: str`
3. `clue_score: float`
4. `source_method: str`
5. `source_page: str`
6. `revid: int`
7. `sentence_offset: int`
8. `oldid_url: str`

### Puzzle Output Record

1. `seed_title: str`
2. `lang: str`
3. `selected_articles: list`
4. `selected_k: int`
5. `size: int`
6. `grid_cells: list` (`size` x `size`, `#` = empty cell)
7. `across_entries: list`
8. `down_entries: list`
9. `fill_status: str` (`complete | failed`)
10. `puzzle_status: str` (`ok | insufficient_vocabulary | insufficient_clues | insufficient_quality`)
11. `layout: dict` (the layout metrics)
12. `diagnostics: dict`
13. `attribution: list`

Entry record: `number`, `answer`, `display_answer`, `enumeration`, `clue`, `clue_style` (`definition | fill_in_blank`), `clue_class`, `row`, `col`, `length`.

## 8. Implementation Backlog

### Phase 0: Scaffolding

1. Create module skeleton, config, deterministic RNG, and logging.
2. Implement disk cache layer keyed by `(endpoint, params_hash)`.
3. Add cache directories and artifact output structure.
4. Emit baseline `diagnostics.json` from first runnable command.
5. Pin `spaCy` package + model versions in dependency files.

### Phase 1: Wikipedia Ingestion

1. Implement outgoing-link fetch with pagination.
2. Implement backlink query and depth annotation.
3. Implement page content + revid retrieval with retry/backoff.
4. Implement lead-section boldface extraction with `mwparserfromhell`.
5. Ensure all ingestion endpoints are cache-backed.
6. Output candidate-level diagnostics even if later stages are disabled.

### Phase 2: Semantic and Structural Ranking

1. Implement TF-IDF + cosine.
2. Implement MMR with depth penalty and backlink bonus.
3. Produce explainable keep/borderline/cut diagnostics.

### Phase 3: K Optimization

1. Implement crossword-aware objective `J(K)`.
2. Implement marginal-gain stopping with min/max bounds.
3. Persist selection trace.
4. Implement thin-pool fallback sequence and diagnostics.

### Phase 4: Terms and Clues

1. Implement phrase/entity extraction with `spaCy` default backend.
2. Add `nltk` fallback mode behind feature flag.
3. Add lead-bold structural term signal and fusion scoring.
4. Add entity-type preference and article-frequency floor.
5. Add answer shape scoring.
6. Implement four-pass clue pipeline (extract, mask, validate, diversity) with leakage validation.

### Phase Gate: Vocabulary Readiness (Before Layout)

1. Do not start layout work until the word list quality gate passes.
2. Gate criteria:
   - At least `gate_min` (default 30) thematically coherent, crossword-friendly answers.
   - Acceptable clue quality/leakage on sampled entries.
   - Diagnostics support score-level auditing for kept/dropped terms.

### Phase 5: Free-Form Layout

1. Implement greedy crossing-based placement with restarts.
2. Implement layout validity checks and metrics.
3. Implement text rendering (`render`).

### Phase 6: Provenance and Packaging

1. Persist clue-level attribution metadata.
2. Validate attribution completeness before output.
3. Package puzzle + diagnostics + attribution bundle.

### Phase 7: Evaluation and Tuning

1. Benchmark across seed sets.
2. Tune weights and thresholds against coherence/layout/clue metrics.
3. Lock release defaults.

## 9. Testing Strategy

### Unit Tests

1. Link pagination and candidate dedup correctness.
2. Depth/backlink feature extraction.
3. Semantic scoring determinism and MMR behavior.
4. `K` stopping behavior on synthetic and real cases.
5. Entity-type preference and frequency-floor filtering.
6. Answer shape scoring calibration.
7. Lead-bold extraction correctness from representative pages using `mwparserfromhell`.
8. Backend parity smoke test (`spaCy` vs `nltk` fallback behavior).
9. Clue leakage rejection (exact + morphological cousin).
10. Clue diversity bucketing and cap enforcement.
11. Layout validity rules (crossing, connectivity, every letter run is one word).
12. Layout metrics and determinism for a fixed random seed.
13. Attribution completeness validation.
14. Cache key stability test for `(endpoint, params_hash)` addressing.
15. Dependency/version pin check for `spaCy` and model package.

### Integration Tests

1. End-to-end generation for benchmark seeds.
2. Cached vs uncached consistency.
3. API retry behavior under transient failures.
4. Layout stage stability for a fixed random seed.
5. Failed layouts emit `fill_status: failed`, `validity_errors` and `puzzle_status: insufficient_quality`.

### Quality Tests

1. Thermodynamics benchmark keep/cut sanity.
2. Secondary-domain benchmark keep/cut sanity (`Jazz` or `Ancient Rome`).
3. Clue readability and leakage thresholds.
4. Layout validity and crossing-quality regression suite.
5. Clue diversity bucket distribution check.

## 10. Operational Details

### Key Configuration Knobs

1. Language: `lang` (`en` | `el`), controls Wikipedia URL, spaCy model, alphabet filter, accent normalization.
2. Selection weights: `a`, `b`, `c`, `d` (see BASELINE_V1 defaults in Section 5.2).
3. `J(K)` weights: `w1-w3` (see BASELINE_V1 defaults in Section 5.3).
4. `K` optimization params: `epsilon`, `m`, `min_k`, `max_k`.
5. Term filter params: length bounds, `min_df`, entity-type weights, shape penalty weight.
6. Clue constraints: min/max words, leakage strictness, diversity bucket cap.
7. Layout limits: `size`, `layout_restarts`, `layout_candidates`, `min_words`.

### Weight Calibration Plan

1. Start with documented defaults for `a,b,c,d` and `w1..w3` that bias relevance over novelty.
2. Run a small grid search over benchmark seeds with logged outcomes.
3. Select weights that maximize coherence and layout quality while keeping clue leakage under threshold.
4. Record chosen defaults in diagnostics and configuration docs.

### CLI Example

```bash
# English
python cli.py generate \
  --seed "Thermodynamics" \
  --lang en \
  --grid-size 15 \
  --expansion one_hop_plus_bounded_two_hop \
  --model tfidf \
  --output outputs/thermo_15.json

# Greek
python cli.py generate \
  --seed "Θερμοδυναμική" \
  --lang el \
  --grid-size 13 \
  --expansion one_hop_only \
  --model tfidf \
  --output outputs/thermo_el_13.json
```

### Output Bundle

1. `puzzle.json`
2. `diagnostics.json`
3. `candidate_scores.csv`
4. `k_selection_trace.csv`
5. `attribution.json`

`diagnostics.json` is required from the first runnable pipeline stage, not only full puzzle runs.

## 11. Risks and Mitigations

1. Risk: semantic model misses technical nuance.
   - Mitigation: optional embedding backend and weighted ensemble.
2. Risk: link graph introduces topic drift.
   - Mitigation: depth penalty + backlink bonus + strict bounded 2-hop expansion.
3. Risk: answer set is thematic but hard to fill.
   - Mitigation: shape penalty, crossing-aware answer scoring, conflict-aware `J(K)`.
4. Risk: extractor quality variance across domains.
   - Mitigation: `spaCy` default, `nltk` fallback, lead-bold structural priors, and benchmark-based calibration.
5. Risk: clues are awkward or leaky.
   - Mitigation: four-pass clue pipeline with leakage validation and diversity deduplication.
6. Risk: attribution gaps discovered late.
   - Mitigation: capture revid/span metadata at extraction time and enforce output validation.
7. Risk: `J(K)` plateaus before sufficient vocabulary is available.
   - Mitigation: four-step rescue ladder (loosen min_df -> promote borderline -> 2-hop -> shorten min length) before failing gracefully.
8. Risk: Greek accent variants break layout intersection matching.
   - Mitigation: strip diacritics at normalization time for the grid; retain accented form only in display and clue text.
9. Risk: overfitting selection/weights to a single STEM seed.
   - Mitigation: tune and gate against at least two dissimilar seeds from the start.

## 12. Milestones (Suggested)

1. Week 1:
   - Ingestion, cache-first layer, backlink/depth features, lead-bold signal capture.
   - Establish benchmark seeds: `Thermodynamics` plus one dissimilar domain (`Jazz` or `Ancient Rome`).
2. Week 2:
   - Semantic scoring and drift diagnostics.
3. Week 3:
   - Crossword-aware `K` optimization.
4. Week 4:
   - Term extraction backends, structural signals, and clue pipeline.
5. Week 5:
   - Vocabulary readiness gate, then free-form layout.
6. Week 6:
   - Layout search, render, evaluation, tuning, attribution QA, release.

## 13. Acceptance Criteria

1. End-to-end command produces puzzle + diagnostics + attribution bundle.
2. Candidate selection uses relevance, redundancy, depth, and backlink signals.
3. Chosen `K` is justified via logged marginal utility.
4. Terms pass entity/frequency/shape quality controls with documented extraction method provenance.
5. Clues pass mask/trim/validate checks with low leakage.
6. Layout starts only after the vocabulary readiness gate passes.
7. Every layout is valid: each word crosses another, one connected piece, every run of 2+ letters is one placed word.
8. A failed layout emits `fill_status: failed` and `validity_errors`, never a partial grid labelled ok.
9. Every published clue includes stable source provenance.

## 14. Immediate Next Tasks (Execution Order)

1. Implement `wiki_client` (links, backlinks, content, `revid`) with caching.
2. Finalize cache key schema `(endpoint, params_hash)` and emit `diagnostics.json` on first runnable stage.
3. Add lead-section boldface extraction via `mwparserfromhell` and storage.
4. Select and fix benchmark seeds: `Thermodynamics` + one dissimilar domain (`Jazz` or `Ancient Rome`).
5. Implement semantic scorer with depth/backlink-aware MMR ranking.
6. Implement `K` optimizer with a conflict-aware objective, plus thin-pool fallbacks.
7. Implement term extraction (`spaCy` primary, `nltk` fallback) with entity/frequency/shape scoring.
8. Implement four-pass clue builder (extract, mask, validate, diversity) with leakage checks.
9. Gate layout start on vocabulary readiness criteria.
10. Implement the free-form layout (Phase 5) and `render`.
11. Add attribution persistence and validation in final output.
