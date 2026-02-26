import shutil
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.pipeline import (
    run_candidate_scoring_stage,
    run_clue_extraction_stage,
    run_k_selection_stage,
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


if __name__ == "__main__":
    unittest.main()
