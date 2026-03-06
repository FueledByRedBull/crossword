import shutil
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.crossword_csp import build_slots, render_grid
from src.csp_heuristics import (
    build_solver_vocabulary,
    evaluate_quality_gate,
    prune_conflicting_assignments,
    run_template_trial,
)
from src.pipeline import (
    _expand_selected_titles_from_scores,
    _effective_min_k_for_size,
    _inventory_min_k_target,
    _quality_rescue_target_count,
    _should_expand_selection_inventory,
    run_candidate_scoring_stage,
    run_clue_extraction_stage,
    run_k_selection_stage,
    run_packaging_stage,
    run_generate_pipeline,
    run_rescue_ladder,
    run_csp_solve_stage,
    run_seed_ingestion_stage,
    run_vocab_gate_stage,
)


class PipelineTests(unittest.TestCase):
    def test_seed_stage_writes_diagnostics_on_link_fetch_failure(self) -> None:
        output_path = Path("tests") / "tmp_outputs" / "diagnostics.json"
        cache_dir = Path("tests") / "tmp_cache_pipeline"

        if output_path.parent.exists():
            shutil.rmtree(output_path.parent, ignore_errors=True)
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)

        with patch("src.pipeline.WikiClient.fetch_links", side_effect=RuntimeError("network blocked")):
            result = run_seed_ingestion_stage(
                seed_title="Thermodynamics",
                lang="en",
                diagnostics_path=output_path,
                cache_dir=cache_dir,
                include_backlinks=True,
            )

        self.assertTrue(result.diagnostics_path.exists())
        self.assertEqual(result.diagnostics["counts"]["candidate_count"], 0)
        self.assertGreaterEqual(len(result.diagnostics["errors"]), 1)
        self.assertIn("link_fetch_failed", result.diagnostics["errors"][0])

        shutil.rmtree(output_path.parent, ignore_errors=True)
        shutil.rmtree(cache_dir, ignore_errors=True)

    def test_seed_stage_rejects_unsupported_lang(self) -> None:
        output_path = Path("tests") / "tmp_outputs" / "diagnostics_lang.json"
        cache_dir = Path("tests") / "tmp_cache_pipeline_lang"

        if output_path.parent.exists():
            shutil.rmtree(output_path.parent, ignore_errors=True)
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)

        result = run_seed_ingestion_stage(
            seed_title="Thermodynamics",
            lang="xx",
            diagnostics_path=output_path,
            cache_dir=cache_dir,
            include_backlinks=True,
        )

        self.assertTrue(result.diagnostics_path.exists())
        self.assertIn("unsupported_lang", result.diagnostics["errors"][0])
        self.assertEqual(result.diagnostics["lang"], "xx")

        shutil.rmtree(output_path.parent, ignore_errors=True)
        shutil.rmtree(cache_dir, ignore_errors=True)

    def test_scoring_stage_handles_empty_candidates(self) -> None:
        output_path = Path("tests") / "tmp_outputs" / "scores.csv"
        diagnostics_path = Path("tests") / "tmp_outputs" / "diagnostics_scores.json"
        cache_dir = Path("tests") / "tmp_cache_pipeline_scores"

        if output_path.parent.exists():
            shutil.rmtree(output_path.parent, ignore_errors=True)
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)

        with patch("src.pipeline.run_seed_ingestion_stage") as mocked_seed:
            mocked_seed.return_value.diagnostics = {
                "candidates": [],
                "errors": ["link_fetch_failed: network blocked"],
            }
            mocked_seed.return_value.diagnostics_path = diagnostics_path
            result = run_candidate_scoring_stage(
                seed_title="Thermodynamics",
                lang="en",
                cache_dir=cache_dir,
                diagnostics_path=diagnostics_path,
                scores_path=output_path,
            )

        self.assertEqual(result.diagnostics["candidate_count"], 0)
        self.assertFalse(result.diagnostics["scores_written"])

        shutil.rmtree(output_path.parent, ignore_errors=True)
        shutil.rmtree(cache_dir, ignore_errors=True)

    def test_k_selection_handles_empty_scores(self) -> None:
        output_path = Path("tests") / "tmp_outputs" / "selected.json"
        diagnostics_path = Path("tests") / "tmp_outputs" / "diagnostics_k.json"
        cache_dir = Path("tests") / "tmp_cache_pipeline_k"

        if output_path.parent.exists():
            shutil.rmtree(output_path.parent, ignore_errors=True)
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)

        with patch("src.pipeline.run_candidate_scoring_stage") as mocked_score:
            mocked_score.return_value.scores = []
            mocked_score.return_value.diagnostics = {"errors": ["link_fetch_failed: network blocked"]}
            result = run_k_selection_stage(
                seed_title="Thermodynamics",
                lang="en",
                cache_dir=cache_dir,
                diagnostics_path=diagnostics_path,
                selected_path=output_path,
            )

        self.assertEqual(result.diagnostics["selected_k"], 0)
        self.assertFalse(result.diagnostics["trace_written"])

        shutil.rmtree(output_path.parent, ignore_errors=True)
        shutil.rmtree(cache_dir, ignore_errors=True)

    def test_candidate_scoring_writes_lexicon_score_column(self) -> None:
        output_path = Path("tests") / "tmp_outputs" / "scores_lex.csv"
        diagnostics_path = Path("tests") / "tmp_outputs" / "diagnostics_scores_lex.json"
        cache_dir = Path("tests") / "tmp_cache_pipeline_scores_lex"
        lexicon_path = Path("tests") / "tmp_outputs" / "lexicon.txt"

        if output_path.parent.exists():
            shutil.rmtree(output_path.parent, ignore_errors=True)
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lexicon_path.write_text("ALPHA 10\nBETA 1\n", encoding="utf-8")

        mocked_seed = SimpleNamespace(
            diagnostics={
                "candidates": [
                    {"title": "Alpha topic", "depth": 1, "links_back_to_seed": True},
                    {"title": "Beta topic", "depth": 1, "links_back_to_seed": True},
                ],
                "errors": [],
            },
            diagnostics_path=diagnostics_path,
        )

        def fake_extract(_self, title, intro_only=True):  # noqa: ARG001
            return {"extract": f"{title} reference text"}

        with (
            patch("src.pipeline.run_seed_ingestion_stage", return_value=mocked_seed),
            patch("src.pipeline.WikiClient.fetch_page_extract", side_effect=fake_extract),
            patch("src.pipeline.WikiClient.fetch_lead_wikitext", return_value={"wikitext": ""}),
        ):
            run_candidate_scoring_stage(
                seed_title="Seed",
                lang="en",
                cache_dir=cache_dir,
                diagnostics_path=diagnostics_path,
                scores_path=output_path,
                lexicon_path=lexicon_path,
                lexicon_weight=0.5,
            )

        header = output_path.read_text(encoding="utf-8").splitlines()[0]
        self.assertIn("lexicon_score", header)

        shutil.rmtree(output_path.parent, ignore_errors=True)
        shutil.rmtree(cache_dir, ignore_errors=True)

    def test_k_selection_maps_selected_titles_by_candidate_index(self) -> None:
        output_path = Path("tests") / "tmp_outputs" / "selected_map.json"
        diagnostics_path = Path("tests") / "tmp_outputs" / "diagnostics_k_map.json"
        trace_path = Path("tests") / "tmp_outputs" / "trace_map.csv"
        cache_dir = Path("tests") / "tmp_cache_pipeline_k_map"

        if output_path.parent.exists():
            shutil.rmtree(output_path.parent, ignore_errors=True)
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)

        mocked_scoring = SimpleNamespace(
            scores=[
                {"rank": 1, "title": "TOP_A"},
                {"rank": 2, "title": "TOP_B"},
                {"rank": 3, "title": "TOP_C"},
            ],
            diagnostics={"errors": []},
            rel_scores=[0.3, 0.2, 0.9],
            pairwise=[
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            ranked_indices=[2, 0, 1],
            term_scores=[0.0, 0.0, 0.0],
            term_length_hists=[],
        )

        with patch("src.pipeline.run_candidate_scoring_stage", return_value=mocked_scoring):
            result = run_k_selection_stage(
                seed_title="Thermodynamics",
                lang="en",
                cache_dir=cache_dir,
                diagnostics_path=diagnostics_path,
                trace_path=trace_path,
                selected_path=output_path,
                min_k=2,
                max_k=2,
            )

        self.assertEqual(result.selected["selected_titles"], ["TOP_A", "TOP_B"])

        shutil.rmtree(output_path.parent, ignore_errors=True)
        shutil.rmtree(cache_dir, ignore_errors=True)

    def test_k_selection_uses_union_of_extracted_answer_inventory(self) -> None:
        output_path = Path("tests") / "tmp_outputs" / "selected_union.json"
        diagnostics_path = Path("tests") / "tmp_outputs" / "diagnostics_k_union.json"
        trace_path = Path("tests") / "tmp_outputs" / "trace_union.csv"
        cache_dir = Path("tests") / "tmp_cache_pipeline_k_union"

        if output_path.parent.exists():
            shutil.rmtree(output_path.parent, ignore_errors=True)
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)

        mocked_scoring = SimpleNamespace(
            scores=[
                {"rank": 1, "title": "TOP_A"},
                {"rank": 2, "title": "TOP_B"},
            ],
            diagnostics={"errors": []},
            rel_scores=[0.9, 0.8],
            pairwise=[
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            ranked_indices=[0, 1],
            term_scores=[1.0, 1.0],
            term_length_hists=[{4: 2}, {4: 2}],
            term_answer_inventories=[{"HEAT", "WORK"}, {"HEAT", "AXIS"}],
        )
        seen_hists: list[dict[int, int]] = []

        def fake_score_template(length_hist, template, min_len=3, **kwargs):  # noqa: ANN001,ARG001
            seen_hists.append(dict(length_hist))
            return {
                "template": template.name,
                "score": 1.0,
                "fill_conflict": 0.0,
                "weighted_shortage_penalty": 0.0,
                "long_slot_penalty": 0.0,
                "auto_block_density": 0.0,
                "disconnected_penalty": 0.0,
            }

        with (
            patch("src.pipeline.run_candidate_scoring_stage", return_value=mocked_scoring),
            patch("src.pipeline.get_templates", return_value=[SimpleNamespace(name="only")]),
            patch("src.pipeline.score_template_from_length_hist", side_effect=fake_score_template),
        ):
            run_k_selection_stage(
                seed_title="Thermodynamics",
                lang="en",
                cache_dir=cache_dir,
                diagnostics_path=diagnostics_path,
                trace_path=trace_path,
                selected_path=output_path,
                min_k=1,
                max_k=2,
                size=13,
                inventory_min_df=1,
            )

        self.assertGreaterEqual(len(seen_hists), 2)
        self.assertEqual(seen_hists[0].get(4), 2)
        self.assertEqual(seen_hists[1].get(4), 3)

        shutil.rmtree(output_path.parent, ignore_errors=True)
        shutil.rmtree(cache_dir, ignore_errors=True)

    def test_vocab_gate_missing_terms(self) -> None:
        diagnostics_path = Path("tests") / "tmp_outputs" / "diagnostics_vocab_gate.json"
        result = run_vocab_gate_stage(
            seed_title="Thermodynamics",
            lang="en",
            diagnostics_path=diagnostics_path,
            terms_path="tests/missing_terms.csv",
        )
        self.assertFalse(result.diagnostics["passed"])
        self.assertEqual(result.diagnostics["reason"], "terms_missing")

    def test_clue_stage_missing_terms(self) -> None:
        diagnostics_path = Path("tests") / "tmp_outputs" / "diagnostics_clues.json"
        result = run_clue_extraction_stage(
            seed_title="Thermodynamics",
            lang="en",
            diagnostics_path=diagnostics_path,
            terms_path="tests/missing_terms.csv",
        )
        self.assertFalse(result.diagnostics["clues_written"])
        if diagnostics_path.parent.exists():
            shutil.rmtree(diagnostics_path.parent, ignore_errors=True)

    def test_csp_partial_fill_output(self) -> None:
        diagnostics_path = Path("tests") / "tmp_outputs" / "diagnostics_csp.json"
        grid_path = Path("tests") / "tmp_outputs" / "grid.json"
        # Minimal terms file to force partial fill
        terms_path = Path("tests") / "tmp_outputs" / "terms.csv"
        terms_path.parent.mkdir(parents=True, exist_ok=True)
        terms_path.write_text(
            "answer,normalized_answer,length,source_method,lead_bold_signal,source_titles,doc_frequency\nCAT,CAT,3,spacy,False,Test,1\n",
            encoding="utf-8",
        )
        result = run_csp_solve_stage(
            seed_title="Test",
            lang="en",
            terms_path=terms_path,
            diagnostics_path=diagnostics_path,
            grid_path=grid_path,
            size=5,
            min_slot_len=3,
            template_name="open",
            max_steps=1000,
            min_domain=1,
            require_gate=False,
        )
        self.assertIn("unfilled_slots", result.diagnostics)
        grid_payload = grid_path.read_text(encoding="utf-8")
        self.assertIn('"fill_status"', grid_payload)
        self.assertIn('"unfilled_slots"', grid_payload)
        shutil.rmtree(terms_path.parent, ignore_errors=True)

    def test_quality_gate_accepts_strong_partial_fill(self) -> None:
        passed, reasons = evaluate_quality_gate(
            fill_percent=0.85,
            invalid_slots=[],
            filler_used_ratio=0.1,
            clued_entry_ratio=1.0,
            clue_answers_available=True,
            long_slot_non_theme_count=0,
        )

        self.assertTrue(passed)
        self.assertEqual(reasons, [])

    def test_solver_vocabulary_prefers_clued_answers_for_clued_lengths(self) -> None:
        tmp_dir = Path("tests") / "tmp_outputs" / "solver_vocab_clues"
        terms_path = tmp_dir / "terms.csv"
        clues_path = tmp_dir / "clues.csv"
        filler_path = tmp_dir / "filler_words.txt"

        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        terms_path.write_text(
            "\n".join(
                [
                    "answer,normalized_answer,length,source_method,lead_bold_signal,source_titles,doc_frequency,theme_score,entity_type_score,crosswordability_score,lexicon_score,shape_penalty,answer_score",
                    "Alpha,ALPHA,5,spacy,False,Test,1,0.2,0.0,0.4,0.2,0.0,0.8",
                    "Bravo,BRAVO,5,spacy,False,Test,1,0.2,0.0,0.4,0.2,0.0,0.7",
                    "Cat,CAT,3,spacy,False,Test,1,0.2,0.0,0.4,0.2,0.0,0.6",
                ]
            ),
            encoding="utf-8",
        )
        clues_path.write_text(
            "\n".join(
                [
                    "answer,normalized_answer,clue,clue_score,source_method,source_page,revid,sentence_offset,oldid_url",
                    "Alpha,ALPHA,First clue,0.9,spacy,Test,1,0,https://example.com",
                ]
            ),
            encoding="utf-8",
        )
        filler_path.write_text("", encoding="utf-8")

        vocabulary = build_solver_vocabulary(
            terms_path=terms_path,
            lang="en",
            filler_path=filler_path,
            filler_min_len=3,
            filler_max_len=12,
            filler_max_per_length=100,
            filler_weight=0.01,
        )

        self.assertIn("ALPHA", vocabulary.themed_set)
        self.assertIn("CAT", vocabulary.themed_set)
        self.assertNotIn("BRAVO", vocabulary.themed_set)

        shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_clue_aware_vocabulary_caps_filler_to_short_lengths(self) -> None:
        tmp_dir = Path("tests") / "tmp_outputs" / "vocab_short_filler_only"
        terms_path = tmp_dir / "terms.csv"
        clues_path = tmp_dir / "clues.csv"
        filler_path = tmp_dir / "filler.txt"

        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        terms_path.write_text(
            "\n".join(
                [
                    "answer,normalized_answer,length,source_method,lead_bold_signal,source_titles,doc_frequency,theme_score,entity_type_score,crosswordability_score,lexicon_score,shape_penalty,answer_score",
                    "Alpha,ALPHA,5,spacy,False,Test,1,1,0,1,0,0,1",
                    "Sound,SOUND,5,spacy,False,Test,1,1,0,1,0,0,0.8",
                ]
            ),
            encoding="utf-8",
        )
        clues_path.write_text(
            "\n".join(
                [
                    "answer,normalized_answer,clue,clue_score,source_method,source_page,revid,sentence_offset,oldid_url",
                    "Alpha,ALPHA,First clue,0.9,spacy,Test,1,0,https://example.com",
                ]
            ),
            encoding="utf-8",
        )
        filler_path.write_text("BRIDGE\nBARRIER\nABSENCE\n", encoding="utf-8")

        vocabulary = build_solver_vocabulary(
            terms_path=terms_path,
            lang="en",
            filler_path=filler_path,
            filler_min_len=3,
            filler_max_len=12,
            filler_max_per_length=100,
            filler_weight=0.01,
        )

        self.assertEqual(vocabulary.effective_filler_max_len, 5)
        self.assertNotIn("BARRIER", vocabulary.words)
        self.assertNotIn("ABSENCE", vocabulary.words)

        shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_prune_conflicting_assignments_clears_invalid_crossings(self) -> None:
        grid = [
            [".", ".", "."],
            [".", ".", "."],
            [".", ".", "."],
        ]
        slots = build_slots(grid, min_len=3)
        down_slots = [slot for slot in slots if slot.direction == "down"]
        assignments = {
            down_slots[0].id: "ABC",
            down_slots[1].id: "DEF",
            down_slots[2].id: "GHI",
        }

        final_assignments, invalid_slots, rendered, removed = prune_conflicting_assignments(
            grid=grid,
            slots=slots,
            assignments_in=assignments,
            word_scores={"ABC": 1.0, "DEF": 0.9, "GHI": 0.8},
            themed_set={"ABC", "DEF", "GHI"},
            clue_answers=set(),
            clue_answers_available=False,
            render_grid_fn=render_grid,
        )

        self.assertEqual(invalid_slots, [])
        self.assertTrue(removed)
        self.assertLess(len(final_assignments), len(assignments))
        self.assertEqual(len(rendered), 3)

    def test_rescue_ladder_runs(self) -> None:
        diagnostics_path = Path("tests") / "tmp_outputs" / "diagnostics_rescue.json"
        result = run_rescue_ladder(
            seed_title="Thermodynamics",
            lang="en",
            diagnostics_path=diagnostics_path,
            selected_path="tests/missing_selected.json",
            terms_path="tests/missing_terms.csv",
        )
        self.assertIn("passed", result)
        self.assertFalse(result["passed"])

    def test_clue_stage_adds_fallback_without_revid(self) -> None:
        tmp_dir = Path("tests") / "tmp_outputs" / "clue_fallback_missing_revid"
        terms_path = tmp_dir / "terms.csv"
        clues_path = tmp_dir / "clues.csv"
        diagnostics_path = tmp_dir / "diagnostics_clues.json"

        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        terms_path.write_text(
            "\n".join(
                [
                    "answer,normalized_answer,source_titles,source_method",
                    "Entropy,ENTROPY,Entropy,spacy",
                ]
            ),
            encoding="utf-8",
        )

        with patch(
            "src.pipeline.WikiClient.fetch_page_extract_with_revid",
            return_value={"title": "Entropy", "extract": "", "revid": None},
        ):
            result = run_clue_extraction_stage(
                seed_title="Thermodynamics",
                lang="en",
                diagnostics_path=diagnostics_path,
                terms_path=terms_path,
                clues_path=clues_path,
            )

        self.assertEqual(result.diagnostics["clue_count"], 1)
        self.assertEqual(result.diagnostics["fallback_template_added"], 1)
        self.assertEqual(result.diagnostics["template_fallback_clue_count"], 1)
        clue_payload = clues_path.read_text(encoding="utf-8")
        self.assertIn("template_fallback", clue_payload)

        shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_clue_stage_marks_source_backed_clue_class(self) -> None:
        tmp_dir = Path("tests") / "tmp_outputs" / "clue_source_backed"
        terms_path = tmp_dir / "terms.csv"
        clues_path = tmp_dir / "clues.csv"
        diagnostics_path = tmp_dir / "diagnostics_clues.json"

        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        terms_path.write_text(
            "\n".join(
                [
                    "answer,normalized_answer,source_titles,source_method",
                    "Entropy,ENTROPY,Entropy,spacy",
                ]
            ),
            encoding="utf-8",
        )

        with (
            patch(
                "src.pipeline.WikiClient.fetch_page_extract_with_revid",
                return_value={
                    "title": "Entropy",
                    "extract": "Entropy is a measure of disorder in thermodynamics.",
                    "revid": 123,
                },
            ),
            patch(
                "src.pipeline.clue_pass_extract_with_offset",
                return_value=("Entropy is a measure of disorder in thermodynamics.", 0),
            ),
            patch("src.pipeline.clue_pass_validate", return_value=True),
        ):
            result = run_clue_extraction_stage(
                seed_title="Thermodynamics",
                lang="en",
                diagnostics_path=diagnostics_path,
                terms_path=terms_path,
                clues_path=clues_path,
            )

        self.assertEqual(result.diagnostics["source_backed_clue_count"], 1)
        clue_payload = clues_path.read_text(encoding="utf-8")
        self.assertIn("source_backed", clue_payload)

        shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_packaging_adds_short_filler_fallback_clue(self) -> None:
        tmp_dir = Path("tests") / "tmp_outputs" / "package_short_filler"
        selected_path = tmp_dir / "selected_candidates.json"
        grid_path = tmp_dir / "grid.json"
        clues_path = tmp_dir / "clues.csv"
        puzzle_path = tmp_dir / "puzzle.json"
        attribution_path = tmp_dir / "attribution.json"
        diagnostics_path = tmp_dir / "diagnostics_package.json"

        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        selected_path.write_text(
            json.dumps(
                {
                    "seed_title": "Test",
                    "lang": "en",
                    "selected_k": 1,
                    "selected_titles": ["Seed"],
                }
            ),
            encoding="utf-8",
        )
        grid_path.write_text(
            json.dumps(
                {
                    "seed_title": "Test",
                    "lang": "en",
                    "template": "open",
                    "size": 5,
                    "min_slot_len": 3,
                    "effective_min_slot_len": 4,
                    "grid": [["A", "B", "C", "D"]],
                    "assignments": {"0": "ABCD"},
                    "slots": [
                        {
                            "id": 0,
                            "direction": "across",
                            "length": 4,
                            "cells": [[0, 0], [0, 1], [0, 2], [0, 3]],
                        }
                    ],
                    "fill_status": "partial",
                    "fill_percent": 1.0,
                    "unfilled_slots": [],
                }
            ),
            encoding="utf-8",
        )
        clues_path.write_text(
            "answer,normalized_answer,clue,clue_score,source_method,source_page,revid,sentence_offset,oldid_url\n",
            encoding="utf-8",
        )

        result = run_packaging_stage(
            seed_title="Test",
            lang="en",
            selected_path=selected_path,
            grid_path=grid_path,
            clues_path=clues_path,
            puzzle_path=puzzle_path,
            attribution_path=attribution_path,
            diagnostics_path=diagnostics_path,
        )

        self.assertEqual(result.diagnostics["synthetic_filler_clue_count"], 1)
        self.assertEqual(result.diagnostics["clued_entry_count"], 1)
        self.assertEqual(result.diagnostics["clued_entry_ratio"], 1.0)
        self.assertEqual(result.diagnostics["source_backed_entry_ratio"], 0.0)
        self.assertEqual(result.diagnostics["fallback_only_entry_ratio"], 0.0)

        shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_packaging_reports_source_backed_entry_ratio(self) -> None:
        tmp_dir = Path("tests") / "tmp_outputs" / "package_source_backed"
        selected_path = tmp_dir / "selected_candidates.json"
        grid_path = tmp_dir / "grid.json"
        clues_path = tmp_dir / "clues.csv"
        puzzle_path = tmp_dir / "puzzle.json"
        attribution_path = tmp_dir / "attribution.json"
        diagnostics_path = tmp_dir / "diagnostics_package.json"

        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        selected_path.write_text(
            json.dumps(
                {
                    "seed_title": "Test",
                    "lang": "en",
                    "selected_k": 1,
                    "selected_titles": ["Seed"],
                }
            ),
            encoding="utf-8",
        )
        grid_path.write_text(
            json.dumps(
                {
                    "seed_title": "Test",
                    "lang": "en",
                    "template": "open",
                    "size": 5,
                    "min_slot_len": 3,
                    "effective_min_slot_len": 4,
                    "grid": [["A", "B", "C", "D"]],
                    "assignments": {"0": "ABCD"},
                    "slots": [
                        {
                            "id": 0,
                            "direction": "across",
                            "length": 4,
                            "cells": [[0, 0], [0, 1], [0, 2], [0, 3]],
                        }
                    ],
                    "fill_status": "partial",
                    "fill_percent": 1.0,
                    "unfilled_slots": [],
                }
            ),
            encoding="utf-8",
        )
        clues_path.write_text(
            "\n".join(
                [
                    "answer,normalized_answer,clue,clue_score,clue_class,source_method,source_page,revid,sentence_offset,oldid_url",
                    "ABCD,ABCD,Source clue,0.9,source_backed,spacy,Test,1,0,https://example.com",
                ]
            ),
            encoding="utf-8",
        )

        result = run_packaging_stage(
            seed_title="Test",
            lang="en",
            selected_path=selected_path,
            grid_path=grid_path,
            clues_path=clues_path,
            puzzle_path=puzzle_path,
            attribution_path=attribution_path,
            diagnostics_path=diagnostics_path,
        )

        self.assertEqual(result.diagnostics["clued_entry_ratio"], 1.0)
        self.assertEqual(result.diagnostics["source_backed_entry_ratio"], 1.0)
        self.assertEqual(result.diagnostics["fallback_only_entry_ratio"], 0.0)

        shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_csp_stage_passes_composite_word_scores_to_solver(self) -> None:
        tmp_dir = Path("tests") / "tmp_outputs" / "csp_word_scores"
        diagnostics_path = tmp_dir / "diagnostics_csp.json"
        grid_path = tmp_dir / "grid.json"
        terms_path = tmp_dir / "terms.csv"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        terms_path.write_text(
            "\n".join(
                [
                    "answer,normalized_answer,length,source_method,lead_bold_signal,source_titles,doc_frequency,theme_score,entity_type_score,crosswordability_score,lexicon_score,shape_penalty,answer_score",
                    "Alpha,ALPHA,5,spacy,False,Test,1,0.2,0.0,0.4,0.2,0.0,0.6",
                    "Beta,BETA,4,spacy,False,Test,1,0.2,0.0,0.8,0.1,0.0,0.3",
                ]
            ),
            encoding="utf-8",
        )

        with patch("src.pipeline.solve_crossword") as mocked_solver:
            mocked_solver.return_value = {
                "solved": False,
                "assignments": {},
                "steps": 0,
                "restarts": 0,
                "local_repair_applied": False,
            }
            run_csp_solve_stage(
                seed_title="Test",
                lang="en",
                terms_path=terms_path,
                diagnostics_path=diagnostics_path,
                grid_path=grid_path,
                size=5,
                min_slot_len=3,
                template_name="open",
                max_steps=100,
                require_gate=False,
            )

        scores = mocked_solver.call_args.kwargs["word_scores"]
        self.assertIn("ALPHA", scores)
        self.assertIn("BETA", scores)
        self.assertAlmostEqual(scores["ALPHA"], (0.6 * 0.6) + (0.25 * 0.2) + (0.15 * 0.4), places=6)
        self.assertAlmostEqual(scores["BETA"], (0.6 * 0.3) + (0.25 * 0.1) + (0.15 * 0.8), places=6)

        shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_rescue_ladder_promotes_borderline_from_csv(self) -> None:
        tmp_dir = Path("tests") / "tmp_outputs" / "rescue_csv"
        selected_path = tmp_dir / "selected_candidates.json"
        candidate_scores_path = tmp_dir / "candidate_scores.csv"
        diagnostics_path = tmp_dir / "diagnostics_rescue.json"
        terms_path = tmp_dir / "answer_candidates.csv"
        selected_override_path = tmp_dir / "selected_candidates_rescue.json"
        terms_diag_path = tmp_dir / "diagnostics_terms_rescue.json"

        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        selected_path.write_text(
            json.dumps(
                {
                    "seed_title": "Thermodynamics",
                    "lang": "en",
                    "selected_k": 1,
                    "selected_titles": ["InitialTitle"],
                }
            ),
            encoding="utf-8",
        )
        candidate_scores_path.write_text(
            "\n".join(
                [
                    "rank,title,depth,links_back_to_seed,rel_score,red_score,selection_score,status,reason",
                    "1,\"Storyville, New Orleans\",1,True,0.2,0.1,0.1,BORDERLINE,relevance_above_borderline_threshold",
                    "2,\"Music of Washington, D.C.\",1,True,0.2,0.1,0.1,BORDERLINE,relevance_above_borderline_threshold",
                    "3,\"Thermodynamics\",1,True,0.9,0.0,0.9,KEEP,relevance_above_keep_threshold",
                ]
            ),
            encoding="utf-8",
        )

        failed_gate = SimpleNamespace(
            passed=False,
            reason="insufficient_terms",
            term_count=0,
            min_required=40,
            max_allowed=80,
        )

        with (
            patch("src.pipeline.run_term_extraction_stage", return_value=SimpleNamespace(terms=[])),
            patch("src.pipeline.evaluate_vocab_gate", return_value=failed_gate),
            patch(
                "src.pipeline.run_seed_ingestion_stage",
                return_value=SimpleNamespace(diagnostics={"candidates": []}),
            ),
        ):
            result = run_rescue_ladder(
                seed_title="Thermodynamics",
                lang="en",
                selected_path=selected_path,
                terms_path=terms_path,
                diagnostics_path=diagnostics_path,
                candidate_scores_path=candidate_scores_path,
                terms_diagnostics_path=terms_diag_path,
                selected_override_path=selected_override_path,
            )

        steps = result["diagnostics"]["steps"]
        promote_steps = [step for step in steps if step.get("action") == "promote_borderline"]
        self.assertTrue(promote_steps)
        self.assertEqual(promote_steps[0]["value"], 2)

        shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_csp_includes_filler_words(self) -> None:
        tmp_dir = Path("tests") / "tmp_outputs" / "csp_filler"
        diagnostics_path = tmp_dir / "diagnostics_csp.json"
        grid_path = tmp_dir / "grid.json"
        terms_path = tmp_dir / "terms.csv"
        filler_path = tmp_dir / "filler_words.txt"

        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        terms_path.write_text(
            "\n".join(
                [
                    "answer,normalized_answer,length,source_method,lead_bold_signal,source_titles,doc_frequency,theme_score,entity_type_score,crosswordability_score,lexicon_score,shape_penalty,answer_score",
                    "Theme,THEME,5,spacy,False,Test,1,0.2,0.0,0.4,0.2,0.0,0.6",
                ]
            ),
            encoding="utf-8",
        )
        filler_path.write_text("FILLER\n", encoding="utf-8")

        with patch("src.pipeline.solve_crossword") as mocked_solver:
            mocked_solver.return_value = {
                "solved": False,
                "assignments": {},
                "steps": 0,
                "restarts": 0,
                "local_repair_applied": False,
            }
            run_csp_solve_stage(
                seed_title="Test",
                lang="en",
                terms_path=terms_path,
                diagnostics_path=diagnostics_path,
                grid_path=grid_path,
                size=5,
                min_slot_len=3,
                template_name="open",
                max_steps=100,
                require_gate=False,
                filler_path=filler_path,
                filler_max_per_length=100,
            )

        words_arg = mocked_solver.call_args.args[2]
        self.assertIn("THEME", words_arg)
        self.assertIn("FILLER", words_arg)

        shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_csp_stage_prefers_fuller_passing_template(self) -> None:
        tmp_dir = Path("tests") / "tmp_outputs" / "csp_template_rank"
        diagnostics_path = tmp_dir / "diagnostics_csp.json"
        grid_path = tmp_dir / "grid.json"
        terms_path = tmp_dir / "terms.csv"

        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        terms_path.write_text(
            "\n".join(
                [
                    "answer,normalized_answer,length,source_method,lead_bold_signal,source_titles,doc_frequency,theme_score,entity_type_score,crosswordability_score,lexicon_score,shape_penalty,answer_score",
                    "Alpha,ALPHA,5,spacy,False,Test,1,0.2,0.0,0.4,0.2,0.0,0.6",
                ]
            ),
            encoding="utf-8",
        )

        slot = SimpleNamespace(
            id=0,
            direction="across",
            length=5,
            cells=[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
        )
        base_trial = {
            "rendered": [["A", "L", "P", "H", "A"]] + [["#"] * 5 for _ in range(4)],
            "result": {
                "solved": False,
                "assignments": {0: "ALPHA"},
                "steps": 10,
                "restarts": 0,
                "local_repair_applied": False,
            },
            "slots": [slot],
            "active_slots": [slot],
            "auto_block": {
                "added_blocks": [],
                "long_slots_before": [],
                "long_slots_after": [],
                "iterations": 0,
            },
            "unfilled_slots": [],
            "trial_errors": [],
            "invalid_slots": [],
            "removed_assignments": [],
            "implicit_added_count": 0,
            "solved_final": False,
            "themed_assigned_count": 1,
            "clued_assigned_count": 1,
            "assigned_count": 1,
            "filler_used_count": 0,
            "filler_used_ratio": 0.0,
            "clued_entry_ratio": 1.0,
            "long_slot_assigned_count": 0,
            "long_slot_non_theme_count": 0,
            "long_slot_theme_ratio": 1.0,
            "unclued_removed_count": 0,
            "phase_a": {
                "steps": 1,
                "restarts": 0,
                "assigned_count": 1,
                "preferred_long_words_count": 0,
            },
        }
        lower_fill_trial = {
            **base_trial,
            "template": SimpleNamespace(name="sparse"),
            "fill_percent": 0.72,
            "fill_status": "partial",
            "fill_count": 18,
            "total_cells": 25,
            "quality_objective": 9.0,
        }
        higher_fill_trial = {
            **base_trial,
            "template": SimpleNamespace(name="dense"),
            "fill_percent": 0.86,
            "fill_status": "partial",
            "fill_count": 21,
            "total_cells": 25,
            "quality_objective": 3.0,
        }

        vocabulary = SimpleNamespace(
            word_scores={"ALPHA": 1.0},
            words=["ALPHA"],
            themed_words=["ALPHA"],
            themed_set={"ALPHA"},
            clue_answers=set(),
            clue_answers_available=False,
            filler_words=[],
            filler_raw_count=0,
            filler_added=0,
            filler_limit_per_length=0,
            filler_weight=0.01,
            long_filler_weight=0.0025,
            min_word_len=5,
            max_word_len=5,
        )
        passing_gate = SimpleNamespace(
            passed=True,
            reason="ok",
            term_count=1,
            min_required=40,
            max_allowed=250,
        )

        with (
            patch("src.pipeline.build_solver_vocabulary", return_value=vocabulary),
            patch("src.pipeline.evaluate_vocab_gate", return_value=passing_gate),
            patch(
                "src.pipeline.get_templates",
                return_value=[SimpleNamespace(name="sparse"), SimpleNamespace(name="dense")],
            ),
            patch(
                "src.pipeline.select_best_template",
                return_value={
                    "selected": "sparse",
                    "scored": [{"template": "sparse"}, {"template": "dense"}],
                },
            ),
            patch(
                "src.pipeline.run_template_trial",
                side_effect=[lower_fill_trial, higher_fill_trial],
            ) as mocked_trial,
        ):
            result = run_csp_solve_stage(
                seed_title="Test",
                lang="en",
                terms_path=terms_path,
                diagnostics_path=diagnostics_path,
                grid_path=grid_path,
                size=5,
                min_slot_len=3,
                max_steps=100,
                require_gate=False,
                template_trials=2,
            )

        self.assertEqual(mocked_trial.call_count, 2)
        self.assertEqual(result.diagnostics["template"], "dense")
        self.assertAlmostEqual(result.diagnostics["fill_percent"], 0.86, places=6)

        shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_csp_prefers_source_backed_trial_over_fallback_heavy_peer(self) -> None:
        tmp_dir = Path("tests") / "tmp_outputs" / "csp_source_backed_rank"
        diagnostics_path = tmp_dir / "diagnostics_csp.json"
        grid_path = tmp_dir / "grid.json"
        terms_path = tmp_dir / "terms.csv"

        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        terms_path.write_text("answer,normalized_answer\n", encoding="utf-8")

        base_trial = {
            "rendered": [["A"]],
            "result": {
                "solved": False,
                "assignments": {0: "ALPHA"},
                "steps": 1,
                "restarts": 0,
                "local_repair_applied": False,
            },
            "slots": [],
            "active_slots": [],
            "auto_block": {
                "added_blocks": [],
                "long_slots_before": [],
                "long_slots_after": [],
                "iterations": 0,
            },
            "fill_percent": 0.8,
            "fill_status": "partial",
            "fill_count": 20,
            "total_cells": 25,
            "unfilled_slots": [],
            "trial_errors": [],
            "invalid_slots": [],
            "removed_assignments": [],
            "implicit_added_count": 0,
            "solved_final": False,
            "themed_assigned_count": 1,
            "clued_assigned_count": 1,
            "assigned_count": 1,
            "filler_used_count": 0,
            "filler_used_ratio": 0.0,
            "clued_entry_ratio": 1.0,
            "long_slot_assigned_count": 1,
            "long_slot_non_theme_count": 0,
            "long_slot_theme_ratio": 1.0,
            "unclued_removed_count": 0,
            "phase_a": {
                "steps": 1,
                "restarts": 0,
                "assigned_count": 1,
                "preferred_long_words_count": 0,
            },
        }
        source_backed_trial = {
            **base_trial,
            "template": SimpleNamespace(name="cleaner"),
            "quality_objective": 1.8,
            "source_backed_entry_count": 1,
            "source_backed_entry_ratio": 1.0,
            "fallback_only_assigned_count": 0,
            "fallback_only_entry_ratio": 0.0,
            "fallback_only_long_count": 0,
            "fallback_only_long_ratio": 0.0,
        }
        fallback_trial = {
            **base_trial,
            "template": SimpleNamespace(name="fallback_heavy"),
            "quality_objective": 2.5,
            "source_backed_entry_count": 0,
            "source_backed_entry_ratio": 0.0,
            "fallback_only_assigned_count": 1,
            "fallback_only_entry_ratio": 1.0,
            "fallback_only_long_count": 1,
            "fallback_only_long_ratio": 1.0,
        }

        vocabulary = SimpleNamespace(
            word_scores={"ALPHA": 1.0},
            words=["ALPHA"],
            themed_words=["ALPHA"],
            themed_set={"ALPHA"},
            clue_answers={"ALPHA"},
            source_backed_answers={"ALPHA"},
            fallback_only_answers=set(),
            clue_answers_available=True,
            filler_words=[],
            filler_raw_count=0,
            filler_added=0,
            filler_limit_per_length=0,
            filler_weight=0.01,
            long_filler_weight=0.0025,
            min_word_len=5,
            max_word_len=5,
        )
        passing_gate = SimpleNamespace(
            passed=True,
            reason="ok",
            term_count=1,
            min_required=40,
            max_allowed=250,
        )

        with (
            patch("src.pipeline.build_solver_vocabulary", return_value=vocabulary),
            patch("src.pipeline.evaluate_vocab_gate", return_value=passing_gate),
            patch(
                "src.pipeline.get_templates",
                return_value=[SimpleNamespace(name="cleaner"), SimpleNamespace(name="fallback_heavy")],
            ),
            patch(
                "src.pipeline.select_best_template",
                return_value={
                    "selected": "cleaner",
                    "scored": [{"template": "cleaner"}, {"template": "fallback_heavy"}],
                },
            ),
            patch(
                "src.pipeline.run_template_trial",
                side_effect=[fallback_trial, source_backed_trial],
            ),
        ):
            result = run_csp_solve_stage(
                seed_title="Test",
                lang="en",
                terms_path=terms_path,
                diagnostics_path=diagnostics_path,
                grid_path=grid_path,
                size=5,
                min_slot_len=3,
                max_steps=100,
                require_gate=False,
                template_trials=2,
            )

        self.assertEqual(result.diagnostics["template"], "cleaner")
        self.assertEqual(result.diagnostics["source_backed_entry_ratio"], 1.0)

        shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_run_template_trial_penalizes_long_fallback_more_than_short_fallback(self) -> None:
        long_slot = SimpleNamespace(id=0, direction="across", length=6, cells=[(0, i) for i in range(6)])
        short_slot = SimpleNamespace(id=1, direction="across", length=4, cells=[(1, i) for i in range(4)])
        slots = [long_slot, short_slot]

        def build_grid_fn(_template):
            return [
                [".", ".", ".", ".", ".", "."],
                [".", ".", ".", "."],
            ]

        def auto_block_long_slots_fn(grid, **_kwargs):
            return {
                "grid": [list(row) for row in grid],
                "added_blocks": [],
                "long_slots_before": [],
                "long_slots_after": [],
                "iterations": 0,
            }

        def build_slots_fn(_grid, min_len):
            return [slot for slot in slots if slot.length >= min_len]

        def render_grid_fn(grid, slot_records, assignments):
            rendered = [list(row) for row in grid]
            slot_map = {slot.id: slot for slot in slot_records}
            for slot_id, word in assignments.items():
                slot = slot_map[slot_id]
                for index, (row, col) in enumerate(slot.cells):
                    rendered[row][col] = word[index]
            return rendered

        def solver_factory(assignments):
            def _solver(_grid, _slots, _words, **_kwargs):
                return {
                    "solved": True,
                    "assignments": assignments,
                    "steps": 1,
                    "restarts": 0,
                    "local_repair_applied": False,
                }

            return _solver

        common_kwargs = {
            "template": SimpleNamespace(name="fixture"),
            "trial_seed": 13,
            "build_grid_fn": build_grid_fn,
            "auto_block_long_slots_fn": auto_block_long_slots_fn,
            "build_slots_fn": build_slots_fn,
            "render_grid_fn": render_grid_fn,
            "words": ["SOURCE", "TEMPLE", "CORE", "FILL"],
            "themed_words": ["SOURCE", "TEMPLE", "CORE", "FILL"],
            "themed_set": {"SOURCE", "TEMPLE", "CORE", "FILL"},
            "clue_answers": {"SOURCE", "TEMPLE", "CORE", "FILL"},
            "source_backed_answers": {"SOURCE", "CORE"},
            "fallback_only_answers": {"TEMPLE", "FILL"},
            "clue_answers_available": True,
            "word_scores": {"SOURCE": 1.0, "TEMPLE": 1.0, "CORE": 1.0, "FILL": 1.0},
            "long_filler_weight": 0.01,
            "max_word_len": 6,
            "effective_min_slot_len": 3,
            "min_domain": 1,
            "max_steps": 100,
            "max_restarts": 0,
            "use_ac3": False,
            "beam_width": 8,
            "enable_local_repair": False,
            "repair_steps": 0,
        }

        long_fallback_trial = run_template_trial(
            solver=solver_factory({0: "TEMPLE", 1: "CORE"}),
            **common_kwargs,
        )
        short_fallback_trial = run_template_trial(
            solver=solver_factory({0: "SOURCE", 1: "FILL"}),
            **common_kwargs,
        )

        self.assertEqual(long_fallback_trial["fallback_only_assigned_count"], 1)
        self.assertEqual(short_fallback_trial["fallback_only_assigned_count"], 1)
        self.assertEqual(long_fallback_trial["fallback_only_long_count"], 1)
        self.assertEqual(short_fallback_trial["fallback_only_long_count"], 0)
        self.assertLess(
            long_fallback_trial["quality_objective"],
            short_fallback_trial["quality_objective"],
        )

    def test_run_template_trial_only_penalizes_fallback_when_source_backed_depth_covers_length(self) -> None:
        slots = [
            SimpleNamespace(id=0, direction="across", length=6, cells=[(0, i) for i in range(6)]),
            SimpleNamespace(id=1, direction="across", length=6, cells=[(1, i) for i in range(6)]),
        ]

        def build_grid_fn(_template):
            return [
                [".", ".", ".", ".", ".", "."],
                [".", ".", ".", ".", ".", "."],
            ]

        def auto_block_long_slots_fn(grid, **_kwargs):
            return {
                "grid": [list(row) for row in grid],
                "added_blocks": [],
                "long_slots_before": [],
                "long_slots_after": [],
                "iterations": 0,
            }

        def build_slots_fn(_grid, min_len):
            return [slot for slot in slots if slot.length >= min_len]

        def render_grid_fn(grid, slot_records, assignments):
            rendered = [list(row) for row in grid]
            slot_map = {slot.id: slot for slot in slot_records}
            for slot_id, word in assignments.items():
                slot = slot_map[slot_id]
                for index, (row, col) in enumerate(slot.cells):
                    rendered[row][col] = word[index]
            return rendered

        phase_b_scores_by_case: dict[str, dict[str, float]] = {}

        def solver_factory(case_name: str):
            call_count = {"value": 0}

            def _solver(_grid, _slots, _words, **kwargs):
                call_count["value"] += 1
                if call_count["value"] == 2:
                    phase_b_scores_by_case[case_name] = dict(kwargs.get("word_scores") or {})
                return {
                    "solved": False,
                    "assignments": {},
                    "steps": 1,
                    "restarts": 0,
                    "local_repair_applied": False,
                }

            return _solver

        common_kwargs = {
            "template": SimpleNamespace(name="fixture"),
            "trial_seed": 13,
            "build_grid_fn": build_grid_fn,
            "auto_block_long_slots_fn": auto_block_long_slots_fn,
            "build_slots_fn": build_slots_fn,
            "render_grid_fn": render_grid_fn,
            "words": ["SOURCE", "ANCHOR", "TEMPLE"],
            "themed_words": ["SOURCE", "ANCHOR", "TEMPLE"],
            "themed_set": {"SOURCE", "ANCHOR", "TEMPLE"},
            "clue_answers": {"SOURCE", "ANCHOR", "TEMPLE"},
            "fallback_only_answers": {"TEMPLE"},
            "clue_answers_available": True,
            "word_scores": {"SOURCE": 1.0, "ANCHOR": 1.0, "TEMPLE": 1.0},
            "long_filler_weight": 0.01,
            "max_word_len": 6,
            "effective_min_slot_len": 3,
            "min_domain": 1,
            "max_steps": 100,
            "max_restarts": 0,
            "use_ac3": False,
            "beam_width": 8,
            "enable_local_repair": False,
            "repair_steps": 0,
        }

        run_template_trial(
            solver=solver_factory("shallow"),
            source_backed_answers={"SOURCE"},
            **common_kwargs,
        )
        run_template_trial(
            solver=solver_factory("deep"),
            source_backed_answers={"SOURCE", "ANCHOR"},
            **common_kwargs,
        )

        self.assertAlmostEqual(phase_b_scores_by_case["shallow"]["TEMPLE"], 1.02)
        self.assertLess(phase_b_scores_by_case["deep"]["TEMPLE"], phase_b_scores_by_case["shallow"]["TEMPLE"])

    def test_csp_raises_effective_min_slot_len_to_clue_floor(self) -> None:
        tmp_dir = Path("tests") / "tmp_outputs" / "csp_clue_floor"
        diagnostics_path = tmp_dir / "diagnostics_csp.json"
        grid_path = tmp_dir / "grid.json"
        terms_path = tmp_dir / "terms.csv"

        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        terms_path.write_text("answer,normalized_answer\n", encoding="utf-8")

        vocabulary = SimpleNamespace(
            word_scores={"ALPHA": 1.0, "ETA": 0.5},
            words=["ALPHA", "ETA"],
            themed_words=["ALPHA", "ETA"],
            themed_set={"ALPHA", "ETA"},
            clue_answers={"ALPHA"},
            clue_answers_available=True,
            filler_words=[],
            filler_raw_count=0,
            filler_added=0,
            filler_limit_per_length=0,
            filler_weight=0.01,
            long_filler_weight=0.0025,
            min_word_len=3,
            max_word_len=5,
        )
        passing_gate = SimpleNamespace(
            passed=True,
            reason="ok",
            term_count=1,
            min_required=40,
            max_allowed=250,
        )

        with (
            patch("src.pipeline.build_solver_vocabulary", return_value=vocabulary),
            patch("src.pipeline.evaluate_vocab_gate", return_value=passing_gate),
            patch("src.pipeline.get_templates", return_value=[SimpleNamespace(name="open")]),
            patch(
                "src.pipeline.select_best_template",
                return_value={"selected": "open", "scored": [{"template": "open"}]},
            ),
            patch(
                "src.pipeline.run_template_trial",
                return_value={
                    "template": SimpleNamespace(name="open"),
                    "rendered": [["#"]],
                    "result": {
                        "solved": False,
                        "assignments": {},
                        "steps": 0,
                        "restarts": 0,
                        "local_repair_applied": False,
                    },
                    "slots": [],
                    "active_slots": [],
                    "auto_block": {
                        "added_blocks": [],
                        "long_slots_before": [],
                        "long_slots_after": [],
                        "iterations": 0,
                    },
                    "fill_percent": 0.0,
                    "fill_status": "failed",
                    "fill_count": 0,
                    "total_cells": 0,
                    "unfilled_slots": [],
                    "trial_errors": [],
                    "invalid_slots": [],
                    "removed_assignments": [],
                    "implicit_added_count": 0,
                    "solved_final": False,
                    "quality_objective": 0.0,
                    "themed_assigned_count": 0,
                    "clued_assigned_count": 0,
                    "assigned_count": 0,
                    "filler_used_count": 0,
                    "filler_used_ratio": 0.0,
                    "clued_entry_ratio": 0.0,
                    "long_slot_assigned_count": 0,
                    "long_slot_non_theme_count": 0,
                    "long_slot_theme_ratio": 1.0,
                    "unclued_removed_count": 0,
                    "phase_a": {
                        "steps": 0,
                        "restarts": 0,
                        "assigned_count": 0,
                        "preferred_long_words_count": 0,
                    },
                },
            ) as mocked_trial,
        ):
            result = run_csp_solve_stage(
                seed_title="Test",
                lang="en",
                terms_path=terms_path,
                diagnostics_path=diagnostics_path,
                grid_path=grid_path,
                size=5,
                min_slot_len=3,
                max_steps=100,
                require_gate=False,
                template_trials=1,
            )

        self.assertEqual(result.diagnostics["effective_min_slot_len"], 5)
        self.assertEqual(mocked_trial.call_args.kwargs["effective_min_slot_len"], 5)

        shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_csp_uses_topology_order_as_hint_without_forcing_single_template(self) -> None:
        tmp_dir = Path("tests") / "tmp_outputs" / "csp_topology_hint"
        diagnostics_path = tmp_dir / "diagnostics_csp.json"
        grid_path = tmp_dir / "grid.json"
        terms_path = tmp_dir / "terms.csv"

        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        terms_path.write_text("answer,normalized_answer\n", encoding="utf-8")

        vocabulary = SimpleNamespace(
            word_scores={"ALPHA": 1.0, "BETA": 0.8},
            words=["ALPHA", "BETA"],
            themed_words=["ALPHA", "BETA"],
            themed_set={"ALPHA", "BETA"},
            clue_answers={"ALPHA", "BETA"},
            clue_answers_available=True,
            filler_words=[],
            filler_raw_count=0,
            filler_added=0,
            filler_limit_per_length=0,
            filler_weight=0.01,
            long_filler_weight=0.0025,
            min_word_len=4,
            max_word_len=5,
        )
        passing_gate = SimpleNamespace(
            passed=True,
            reason="ok",
            term_count=2,
            min_required=40,
            max_allowed=250,
        )

        trials = [
            {
                "template": SimpleNamespace(name="medium_open"),
                "rendered": [["#"]],
                "result": {"solved": False, "assignments": {}, "steps": 0, "restarts": 0, "local_repair_applied": False},
                "slots": [],
                "active_slots": [],
                "auto_block": {"added_blocks": [], "long_slots_before": [], "long_slots_after": [], "iterations": 0},
                "fill_percent": 0.62,
                "fill_status": "failed",
                "fill_count": 0,
                "total_cells": 0,
                "unfilled_slots": [],
                "trial_errors": [],
                "invalid_slots": [],
                "removed_assignments": [],
                "implicit_added_count": 0,
                "solved_final": False,
                "quality_objective": 0.62,
                "themed_assigned_count": 0,
                "clued_assigned_count": 0,
                "assigned_count": 0,
                "filler_used_count": 0,
                "filler_used_ratio": 0.0,
                "clued_entry_ratio": 0.0,
                "long_slot_assigned_count": 0,
                "long_slot_non_theme_count": 0,
                "long_slot_theme_ratio": 1.0,
                "unclued_removed_count": 0,
                "phase_a": {"steps": 0, "restarts": 0, "assigned_count": 0, "preferred_long_words_count": 0},
            },
            {
                "template": SimpleNamespace(name="nyt_classic"),
                "rendered": [["#"]],
                "result": {"solved": False, "assignments": {}, "steps": 0, "restarts": 0, "local_repair_applied": False},
                "slots": [],
                "active_slots": [],
                "auto_block": {"added_blocks": [], "long_slots_before": [], "long_slots_after": [], "iterations": 0},
                "fill_percent": 0.65,
                "fill_status": "failed",
                "fill_count": 0,
                "total_cells": 0,
                "unfilled_slots": [],
                "trial_errors": [],
                "invalid_slots": [],
                "removed_assignments": [],
                "implicit_added_count": 0,
                "solved_final": False,
                "quality_objective": 0.65,
                "themed_assigned_count": 0,
                "clued_assigned_count": 0,
                "assigned_count": 0,
                "filler_used_count": 0,
                "filler_used_ratio": 0.0,
                "clued_entry_ratio": 0.0,
                "long_slot_assigned_count": 0,
                "long_slot_non_theme_count": 0,
                "long_slot_theme_ratio": 1.0,
                "unclued_removed_count": 0,
                "phase_a": {"steps": 0, "restarts": 0, "assigned_count": 0, "preferred_long_words_count": 0},
            },
        ]

        with (
            patch("src.pipeline.build_solver_vocabulary", return_value=vocabulary),
            patch("src.pipeline.evaluate_vocab_gate", return_value=passing_gate),
            patch(
                "src.pipeline.get_templates",
                return_value=[SimpleNamespace(name="symmetric_sparse"), SimpleNamespace(name="medium_open"), SimpleNamespace(name="nyt_classic")],
            ),
            patch(
                "src.pipeline.select_best_template",
                return_value={
                    "selected": "symmetric_sparse",
                    "scored": [
                        {"template": "symmetric_sparse"},
                        {"template": "medium_open"},
                        {"template": "nyt_classic"},
                    ],
                },
            ),
            patch("src.pipeline.run_template_trial", side_effect=trials) as mocked_trial,
        ):
            result = run_csp_solve_stage(
                seed_title="Test",
                lang="en",
                terms_path=terms_path,
                diagnostics_path=diagnostics_path,
                grid_path=grid_path,
                size=5,
                min_slot_len=3,
                max_steps=100,
                require_gate=False,
                template_trials=2,
                template_priority_names=["medium_open", "nyt_classic"],
            )

        tried_templates = [
            call.kwargs["template"].name
            for call in mocked_trial.call_args_list
        ]
        self.assertEqual(tried_templates, ["medium_open", "nyt_classic"])
        self.assertEqual(result.diagnostics["template"], "nyt_classic")

        shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_expand_selection_inventory_on_failed_short_pool(self) -> None:
        should_expand = _should_expand_selection_inventory(
            size=15,
            terms=[
                {"normalized_answer": "HEAT"},
                {"normalized_answer": "WORK"},
                {"normalized_answer": "STATE"},
            ],
            solve_diagnostics={"fill_status": "failed", "fill_percent": 0.71},
        )
        self.assertTrue(should_expand)

    def test_expand_selected_titles_from_scores_appends_ranked_titles(self) -> None:
        tmp_dir = Path("tests") / "tmp_outputs" / "expand_selection"
        scores_path = tmp_dir / "candidate_scores.csv"

        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        scores_path.write_text(
            "\n".join(
                [
                    "rank,title,status",
                    "1,Alpha,KEEP",
                    "2,Beta,KEEP",
                    "3,Gamma,KEEP",
                ]
            ),
            encoding="utf-8",
        )

        expanded = _expand_selected_titles_from_scores(
            selected_titles=["Alpha"],
            candidate_scores_path=scores_path,
            target_count=3,
        )

        self.assertEqual(expanded, ["Alpha", "Beta", "Gamma"])
        shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_inventory_min_k_target_uses_short_answer_thresholds(self) -> None:
        self.assertEqual(
            _inventory_min_k_target(
                size=15,
                ranked_indices=[0, 1, 2],
                term_answer_inventories=[
                    {"HEAT", "WORK", "FORM", "TIME"},
                    {"ATOMS", "STATE", "PHASE", "FORCE"},
                    {"BULK", "LAWS", "BASIS", "FIELD", "FIRST", "STUDY"},
                ],
                inventory_min_df=1,
            ),
            3,
        )
        self.assertEqual(
            _inventory_min_k_target(
                size=15,
                ranked_indices=[0, 1],
                term_answer_inventories=[
                    {"HEAT", "WORK"},
                    {"ATOMS"},
                ],
                inventory_min_df=1,
            ),
            2,
        )

    def test_effective_min_k_for_15x15_uses_inventory_target(self) -> None:
        self.assertEqual(
            _effective_min_k_for_size(
                size=15,
                min_k=5,
                candidate_count=20,
                ranked_indices=[0, 1, 2],
                term_answer_inventories=[
                    {"HEAT", "WORK", "FORM", "TIME"},
                    {"ATOMS", "STATE", "PHASE", "FORCE"},
                    {
                        "BULK", "LAWS", "RATE", "HELP", "SOME", "STUDY",
                        "BASIS", "FIELD", "FIRST", "TERMS", "UNITS", "GIBBS",
                    },
                ],
                inventory_min_df=1,
            ),
            5,
        )
        with patch("src.pipeline._inventory_min_k_target", return_value=17):
            self.assertEqual(
                _effective_min_k_for_size(
                    size=15,
                    min_k=5,
                    candidate_count=20,
                    ranked_indices=list(range(20)),
                    term_answer_inventories=[set() for _ in range(20)],
                    inventory_min_df=2,
                ),
                12,
            )

    def test_quality_rescue_target_count_can_expand_past_twelve(self) -> None:
        self.assertEqual(
            _quality_rescue_target_count(
                size=15,
                current_selected_count=12,
                terms=[{"normalized_answer": f"A{i:03d}"} for i in range(10)]
                + [{"normalized_answer": f"B{i:03d}C"} for i in range(12)],
            ),
            14,
        )

    def test_generate_pipeline_expands_selection_after_failed_short_inventory(self) -> None:
        tmp_dir = Path("tests") / "tmp_outputs" / "generate_quality_rescue"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        scores_path = tmp_dir / "candidate_scores.csv"
        selected_path = tmp_dir / "selected_candidates.json"
        scores_path.write_text(
            "\n".join(
                [
                    "rank,title,status",
                    "1,Alpha,KEEP",
                    "2,Beta,KEEP",
                    "3,Gamma,KEEP",
                    "4,Delta,KEEP",
                    "5,Epsilon,KEEP",
                    "6,Zeta,KEEP",
                    "7,Eta,KEEP",
                    "8,Theta,KEEP",
                    "9,Iota,KEEP",
                    "10,Kappa,KEEP",
                    "11,Lambda,KEEP",
                    "12,Mu,KEEP",
                    "13,Nu,KEEP",
                    "14,Xi,KEEP",
                ]
            ),
            encoding="utf-8",
        )
        selected_path.write_text(
            json.dumps(
                {
                    "seed_title": "Thermodynamics",
                    "lang": "en",
                    "selected_k": 12,
                    "selected_titles": [
                        "Alpha",
                        "Beta",
                        "Gamma",
                        "Delta",
                        "Epsilon",
                        "Zeta",
                        "Eta",
                        "Theta",
                        "Iota",
                        "Kappa",
                        "Lambda",
                        "Mu",
                    ],
                }
            ),
            encoding="utf-8",
        )

        low_inventory_terms = [
            {"normalized_answer": "HEAT"},
            {"normalized_answer": "WORK"},
            {"normalized_answer": "STATE"},
        ]
        high_inventory_terms = (
            [{"normalized_answer": f"A{i:03d}B"} for i in range(12)]
            + [{"normalized_answer": f"C{i:03d}D"} for i in range(12)]
        )
        low_terms_result = SimpleNamespace(
            terms_path=tmp_dir / "answer_candidates.csv",
            diagnostics_path=tmp_dir / "diagnostics_terms.json",
            terms=low_inventory_terms,
            diagnostics={"term_count": len(low_inventory_terms)},
        )
        high_terms_result = SimpleNamespace(
            terms_path=tmp_dir / "answer_candidates.csv",
            diagnostics_path=tmp_dir / "diagnostics_terms_rescue.json",
            terms=high_inventory_terms,
            diagnostics={"term_count": len(high_inventory_terms)},
        )
        gate_result = SimpleNamespace(
            diagnostics_path=tmp_dir / "diagnostics_vocab_gate.json",
            diagnostics={"passed": True},
        )
        clue_result = SimpleNamespace(
            clues_path=tmp_dir / "clues.csv",
            diagnostics_path=tmp_dir / "diagnostics_clues.json",
            clues=[],
            diagnostics={},
        )
        topology_result = SimpleNamespace(
            diagnostics_path=tmp_dir / "diagnostics_topology.json",
            diagnostics={"selected_template": "nyt_classic", "scored": [{"template": "nyt_classic"}]},
        )
        failed_solve = SimpleNamespace(
            grid_path=tmp_dir / "grid.json",
            diagnostics_path=tmp_dir / "diagnostics_csp.json",
            diagnostics={"fill_status": "failed", "fill_percent": 0.71, "filler": {"used_ratio": 0.3}},
        )
        improved_solve = SimpleNamespace(
            grid_path=tmp_dir / "grid.json",
            diagnostics_path=tmp_dir / "diagnostics_csp.json",
            diagnostics={"fill_status": "partial", "fill_percent": 0.74, "filler": {"used_ratio": 0.15}},
        )
        package_result = SimpleNamespace(
            puzzle_path=tmp_dir / "puzzle.json",
            attribution_path=tmp_dir / "attribution.json",
            diagnostics_path=tmp_dir / "diagnostics_package.json",
            diagnostics={"puzzle_status": "ok"},
        )

        with (
            patch("src.pipeline.run_candidate_scoring_stage", return_value=SimpleNamespace()),
            patch("src.pipeline.run_k_selection_stage", return_value=SimpleNamespace()),
            patch("src.pipeline.run_term_extraction_stage", side_effect=[low_terms_result, high_terms_result]) as mocked_terms,
            patch("src.pipeline.run_vocab_gate_stage", return_value=gate_result),
            patch("src.pipeline.run_clue_extraction_stage", side_effect=[clue_result, clue_result]),
            patch("src.pipeline.run_topology_selection_stage", side_effect=[topology_result, topology_result]),
            patch("src.pipeline.run_csp_solve_stage", side_effect=[failed_solve, improved_solve]),
            patch("src.pipeline.run_packaging_stage", return_value=package_result) as mocked_package,
        ):
            run_generate_pipeline(
                seed_title="Thermodynamics",
                lang="en",
                output_dir=tmp_dir,
                cache_dir="data/cache/wiki",
                rescue=True,
                size=15,
                min_df=2,
                use_topology=False,
            )

        self.assertEqual(mocked_terms.call_count, 2)
        second_terms_selected = Path(mocked_terms.call_args_list[1].kwargs["selected_path"])
        self.assertEqual(second_terms_selected.name, "selected_candidates_quality_rescue.json")
        self.assertEqual(Path(mocked_package.call_args.kwargs["selected_path"]).name, "selected_candidates_quality_rescue.json")
        rescue_payload = json.loads((tmp_dir / "selected_candidates_quality_rescue.json").read_text(encoding="utf-8"))
        self.assertEqual(rescue_payload["selected_k"], 14)
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
