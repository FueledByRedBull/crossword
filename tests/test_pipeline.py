import csv
import json
import shutil
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.clue_builder import definition_clue, split_sentences
from src.layout import validate_puzzle
from src.pipeline import (
    _build_title_answer_rows,
    _effective_min_k_for_size,
    _inventory_min_k_target,
    _definition_clue_rows,
    _layout_candidates,
    _linked_titles,
    _uses_seed_word,
    _seed_mentions,
    run_candidate_scoring_stage,
    run_clue_extraction_stage,
    run_k_selection_stage,
    run_layout_stage,
    run_packaging_stage,
    run_rescue_ladder,
    run_seed_ingestion_stage,
    run_term_extraction_stage,
    run_vocab_gate_stage,
)


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        # Stage tests never reach the network; the batched prefetch is checked where it matters.
        for name in ("prefetch_intro_extracts", "prefetch_lead_wikitext"):
            patcher = patch(f"src.pipeline.WikiClient.{name}")
            setattr(self, name, patcher.start())
            self.addCleanup(patcher.stop)
        patcher = patch(
            "src.pipeline.WikiClient.fetch_page_extract_with_revid",
            return_value={"extract": "", "revid": None},
        )
        patcher.start()
        self.addCleanup(patcher.stop)

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

    def test_seed_stage_link_cap_samples_the_whole_alphabet(self) -> None:
        out_dir = Path("tests") / "tmp_outputs"
        cache_dir = Path("tests") / "tmp_cache_pipeline_cap"
        links = [{"title": f"{letter}{n}", "page_id": None, "depth": 1} for letter in "ABCDEFGHIJ" for n in range(10)]
        payload = {"seed_title": "Seed", "seed_page_id": 1, "links": links}

        with patch("src.pipeline.WikiClient.fetch_links", return_value=payload):
            result = run_seed_ingestion_stage(
                seed_title="Seed",
                diagnostics_path=out_dir / "diagnostics_cap.json",
                cache_dir=cache_dir,
                include_backlinks=False,
                max_links=10,
            )

        sampled = [c["title"] for c in result.diagnostics["candidates"]]
        self.assertEqual(sampled, [f"{letter}0" for letter in "ABCDEFGHIJ"])
        shutil.rmtree(out_dir, ignore_errors=True)
        shutil.rmtree(cache_dir, ignore_errors=True)

    def test_seed_stage_prefetches_the_intros_of_the_two_hop_parents(self) -> None:
        payload = {"seed_title": "Seed", "seed_page_id": 1, "links": [{"title": "A"}, {"title": "B"}]}

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("src.pipeline.WikiClient.fetch_links", return_value=payload),
            patch("src.pipeline.WikiClient.fetch_page_extract", return_value={"extract": "some text"}),
        ):
            result = run_seed_ingestion_stage(
                seed_title="Seed",
                diagnostics_path=Path(tmp) / "diagnostics.json",
                cache_dir=Path(tmp) / "cache",
                include_backlinks=False,
                expansion="one_hop_plus_bounded_two_hop",
                max_two_hop_parents=1,
            )

        self.prefetch_intro_extracts.assert_called_once_with(["A", "B"])
        self.assertEqual(result.diagnostics["expansion"]["two_hop_parents_used"], 1)

    def test_scoring_stage_prefetches_its_batch_and_carries_on_when_that_fails(self) -> None:
        candidates = [{"title": "Alpha topic", "depth": 1}, {"title": "Beta topic", "depth": 1}]
        seed_result = SimpleNamespace(diagnostics={"candidates": candidates, "errors": []})
        self.prefetch_lead_wikitext.side_effect = RuntimeError("boom")

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("src.pipeline.run_seed_ingestion_stage", return_value=seed_result),
            patch("src.pipeline.WikiClient.fetch_page_extract", return_value={"extract": "reference text"}),
            patch("src.pipeline.WikiClient.fetch_lead_wikitext", return_value={"wikitext": ""}),
        ):
            result = run_candidate_scoring_stage(
                seed_title="Seed",
                cache_dir=Path(tmp) / "cache",
                diagnostics_path=Path(tmp) / "diagnostics.json",
                scores_path=Path(tmp) / "scores.csv",
            )

        self.prefetch_intro_extracts.assert_called_once_with(["Seed", "Alpha topic", "Beta topic"])
        self.prefetch_lead_wikitext.assert_called_once_with(["Alpha topic", "Beta topic"])
        self.assertIn("prefetch_failed:boom", result.diagnostics["errors"])
        self.assertEqual(len(result.scores), 2)

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

    def test_k_selection_reuses_a_scoring_result_it_is_given(self) -> None:
        out_dir = Path("tests") / "tmp_outputs"
        scoring = SimpleNamespace(scores=[], diagnostics={"errors": []})

        with patch("src.pipeline.run_candidate_scoring_stage") as mocked_score:
            result = run_k_selection_stage(
                seed_title="Thermodynamics",
                diagnostics_path=out_dir / "diagnostics_k.json",
                selected_path=out_dir / "selected.json",
                scoring=scoring,
            )

        mocked_score.assert_not_called()
        self.assertEqual(result.diagnostics["selected_k"], 0)
        shutil.rmtree(out_dir, ignore_errors=True)

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

    ROME_INTROS = {
        "Ancient Rome": "Ancient Rome was a civilization in Italy.",
        "Romania": "Romania was a civilization in Italy.",
        "Latium": "Latium is the region around Rome.",
        "Diocletian": "Diocletian was an emperor.",
        "Hannibal": "Hannibal was a general.",
    }

    def score_rome(
        self,
        tmp: str,
        intros: dict[str, str] | None = None,
        seed_text: str | None = None,
        extra_titles: list[str] | None = None,
    ):
        intros = intros or self.ROME_INTROS
        candidates = [{"title": title, "depth": 1, "links_back_to_seed": True} for title in list(intros)[1:]]
        seed_result = SimpleNamespace(diagnostics={"candidates": candidates, "errors": []})
        seed_text = seed_text or "Hannibal crossed the Alps. Diocletian split the empire; later Diocletian retired."
        with (
            patch("src.pipeline.run_seed_ingestion_stage", return_value=seed_result),
            patch(
                "src.pipeline.WikiClient.fetch_page_extract",
                side_effect=lambda title, intro_only=True: {"extract": intros[title]},
            ),
            patch(
                "src.pipeline.WikiClient.fetch_page_extract_with_revid",
                return_value={"extract": seed_text, "revid": 3},
            ),
            patch("src.pipeline.WikiClient.fetch_lead_wikitext", return_value={"wikitext": ""}),
        ):
            return run_candidate_scoring_stage(
                seed_title="Ancient Rome",
                cache_dir=Path(tmp) / "cache",
                diagnostics_path=Path(tmp) / "diagnostics.json",
                scores_path=Path(tmp) / "scores.csv",
                lexicon_path=None,
                extra_titles=extra_titles,
            )

    def test_scoring_cuts_off_topic_candidates_and_ranks_on_topic_ones_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = self.score_rome(tmp)
            with (Path(tmp) / "scores.csv").open(encoding="utf-8", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))

        by_title = {row["title"]: row for row in result.scores}
        # Romania is the candidate most like the seed, but its intro lacks the seed stem (Rome) and the seed never names it.
        self.assertEqual(max(by_title, key=lambda title: by_title[title]["rel_score"]), "Romania")
        self.assertEqual((by_title["Romania"]["status"], by_title["Romania"]["reason"]), ("CUT", "cut_off_topic"))
        self.assertEqual(by_title["Hannibal"]["reason"], "cut_off_topic")  # named once is not enough
        self.assertNotEqual(by_title["Latium"]["reason"], "cut_off_topic")  # the intro says Rome
        self.assertNotEqual(by_title["Diocletian"]["reason"], "cut_off_topic")  # named twice
        self.assertEqual({row["title"] for row in result.scores[:2]}, {"Latium", "Diocletian"})
        # rank, ranked_indices and the order of scores describe the same ranking.
        titles = list(self.ROME_INTROS)[1:]
        self.assertEqual([row["rank"] for row in result.scores], [1, 2, 3, 4])
        self.assertEqual([titles[idx] for idx in result.ranked_indices], [row["title"] for row in result.scores])
        self.assertEqual([row["title"] for row in csv_rows], [row["title"] for row in result.scores])
        signals = {row["title"]: (row["intro_on_topic"], row["seed_mentions"]) for row in csv_rows}
        self.assertEqual(
            signals,
            {"Latium": ("True", "0"), "Diocletian": ("False", "2"), "Hannibal": ("False", "1"), "Romania": ("False", "0")},
        )
        self.assertEqual(by_title["Latium"]["intro_on_topic"], True)
        self.assertEqual(by_title["Diocletian"]["seed_mentions"], 2)

    def test_scoring_puts_pages_with_an_on_topic_intro_before_pages_the_seed_merely_names(self) -> None:
        intros = {
            "Ancient Rome": "Ancient Rome was a civilization in Italy.",
            "Named": "A civilization in Italy was a civilization in Italy.",
            "Intro": "Rome is a city.",
            "Other": "Hannibal was a general.",
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = self.score_rome(tmp, intros, "Named appears here. Named again.")

        by_title = {row["title"]: row for row in result.scores}
        self.assertGreater(by_title["Named"]["rel_score"], by_title["Intro"]["rel_score"])  # MMR alone would lead with it
        self.assertEqual([row["title"] for row in result.scores], ["Intro", "Named", "Other"])
        self.assertEqual(by_title["Other"]["reason"], "cut_off_topic")

    def test_scoring_scores_extra_titles_as_two_hop_pages_by_the_same_rule(self) -> None:
        intros = {**self.ROME_INTROS, "Roman law": "Roman law developed over centuries.", "Plankton": "Plankton drifts."}
        with tempfile.TemporaryDirectory() as tmp:
            self.score_rome(tmp, intros, extra_titles=["Roman law", "Plankton"])
            with (Path(tmp) / "scores.csv").open(encoding="utf-8", newline="") as handle:
                csv_rows = {row["title"]: row for row in csv.DictReader(handle)}

        # Not linked by the seed: depth 2, no back-link; only the intro arm (or the seed naming it) admits it.
        self.assertEqual((csv_rows["Roman law"]["depth"], csv_rows["Roman law"]["links_back_to_seed"]), ("2", "False"))
        self.assertEqual(csv_rows["Roman law"]["intro_on_topic"], "True")
        self.assertEqual(csv_rows["Plankton"]["reason"], "cut_off_topic")
        self.assertEqual(csv_rows["Hannibal"]["depth"], "1")

    def test_linked_titles_takes_an_even_share_of_each_parents_new_links(self) -> None:
        client = MagicMock()
        client.fetch_links_concurrent.return_value = (
            {
                "A": {"links": [{"title": t} for t in ("Known", "a1", "a2", "a3", "a4", "a5")]},
                "B": {"links": [{"title": t} for t in ("a1", "b1")]},
                "C": {"links": []},
            },
            {"C": "boom"},
        )
        errors: list[str] = []
        with patch("src.pipeline._RESCUE_NEW_PAGES", 6):  # a share of two per parent
            titles = _linked_titles(client, ["A", "B", "C"], {"Known"}, errors)

        self.assertEqual(titles, ["a1", "a3", "b1"])  # even steps through A; B's a1 is A's already
        self.assertEqual(errors, ["two_hop_links_failed:C:boom"])
        client.fetch_links_concurrent.assert_called_once_with(["A", "B", "C"], max_links=500)

    def test_rescue_ladder_widens_from_the_on_topic_selected_articles_and_selects_the_best_new_pages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            (tmp_dir / "selected.json").write_text(
                json.dumps({"seed_title": "Seed", "lang": "en", "selected_k": 2, "selected_titles": ["On", "Named"]}),
                encoding="utf-8",
            )
            (tmp_dir / "scores.csv").write_text(
                "\n".join(
                    [
                        "rank,title,depth,links_back_to_seed,rel_score,intro_on_topic,seed_mentions,status,reason",
                        "1,On,1,True,0.5,True,0,KEEP,x",
                        "2,Named,1,True,0.5,False,2,KEEP,x",
                    ]
                ),
                encoding="utf-8",
            )
            widened = SimpleNamespace(
                scores_path=tmp_dir / "candidate_scores_rescue.csv",
                scores=[
                    {"title": "Heat", "depth": 2, "status": "KEEP"},
                    {"title": "On", "depth": 1, "status": "KEEP"},
                    {"title": "Cold", "depth": 2, "status": "CUT"},
                ],
            )
            gates = [SimpleNamespace(passed=False)] * 3 + [SimpleNamespace(passed=True)]
            with (
                patch("src.pipeline.run_term_extraction_stage", return_value=SimpleNamespace(terms=[])) as terms,
                patch("src.pipeline.evaluate_vocab_gate", side_effect=gates),
                patch(
                    "src.pipeline.WikiClient.fetch_links_concurrent",
                    return_value=({"On": {"links": [{"title": "Heat"}, {"title": "Cold"}, {"title": "On"}]}}, {}),
                ) as links,
                patch("src.pipeline.run_candidate_scoring_stage", return_value=widened) as rescore,
            ):
                result = run_rescue_ladder(
                    seed_title="Seed",
                    selected_path=tmp_dir / "selected.json",
                    terms_path=tmp_dir / "terms.csv",
                    diagnostics_path=tmp_dir / "diagnostics_rescue.json",
                    candidate_scores_path=tmp_dir / "scores.csv",
                    terms_diagnostics_path=tmp_dir / "diagnostics_terms_rescue.json",
                    selected_override_path=tmp_dir / "selected_rescue.json",
                    scoring_args={"seed_title": "Seed"},
                )
            selected = json.loads((tmp_dir / "selected_rescue.json").read_text(encoding="utf-8"))["selected_titles"]

        # Only the selected article whose intro is on topic is widened from, and only new pages are scored.
        links.assert_called_once_with(["On"], max_links=500)
        self.assertEqual(rescore.call_args.kwargs["extra_titles"], ["Heat", "Cold"])
        self.assertEqual(result["diagnostics"]["reason"], "widen_from_selected")
        self.assertEqual(selected, ["On", "Named", "Heat"])  # new pages that scoring did not cut
        self.assertEqual(terms.call_args.kwargs["scores_path"], widened.scores_path)

    def test_scoring_counts_a_derived_form_of_the_seed_word_only_in_the_first_sentence(self) -> None:
        intros = {
            **self.ROME_INTROS,
            "Plautus": "Plautus was a Roman playwright. He wrote comedies.",
            "Savoy": "Savoy was a duchy. It was a state of the Holy Roman Empire.",
            "Lazio": "Lazio is a region of Italy. Its capital is Rome.",
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = self.score_rome(tmp, intros)

        on_topic = {row["title"]: row["intro_on_topic"] for row in result.scores}
        # "Roman" defines Plautus and is a passing mention in Savoy; "Rome" itself counts anywhere.
        self.assertEqual((on_topic["Plautus"], on_topic["Savoy"], on_topic["Lazio"]), (True, False, True))

    def test_seed_word_match_takes_derived_forms_on_request_and_the_rarest_word_of_the_seed_title(self) -> None:
        def uses(text, seed, lexicon=None, derived=False):
            return _uses_seed_word(text, seed, lang="en", lexicon=lexicon or {}, derived=derived)

        # Thermodynamic matches thermodynamics; Roman is a derived form of Rome, Romania and Romagna are not.
        self.assertTrue(uses("A thermodynamic cycle.", "Thermodynamics"))
        self.assertFalse(uses("A thermostat.", "Thermodynamics"))
        self.assertTrue(uses("All roads lead to Rome.", "Ancient Rome"))
        self.assertFalse(uses("The Roman army marched.", "Ancient Rome"))
        self.assertTrue(uses("The Roman army marched.", "Ancient Rome", derived=True))
        self.assertTrue(uses("Romans built roads.", "Ancient Rome", derived=True))
        self.assertFalse(uses("Romania is a country.", "Ancient Rome", derived=True))
        self.assertFalse(uses("Romagna is a region.", "Ancient Rome", derived=True))
        # Where the lexicon ranks the words, only the rarest one counts; where it cannot, any word does.
        lexicon = {"ANCIENT": 0.7}
        self.assertFalse(uses("Homs is an ancient city.", "Ancient Rome", lexicon))
        self.assertTrue(uses("Homs is an ancient city.", "Ancient Rome"))
        self.assertTrue(uses("Rome was ancient.", "Ancient Rome", lexicon))

    def test_k_selection_takes_on_topic_articles_first_and_falls_back_when_they_run_out(self) -> None:
        picks = {}
        with tempfile.TemporaryDirectory() as tmp:
            scoring = self.score_rome(tmp)
            for k in (2, 3):
                result = run_k_selection_stage(
                    seed_title="Ancient Rome",
                    diagnostics_path=Path(tmp) / "diagnostics_k.json",
                    trace_path=Path(tmp) / "trace.csv",
                    selected_path=Path(tmp) / "selected.json",
                    min_k=k,
                    max_k=k,
                    scoring=scoring,
                )
                picks[k] = result.selected["selected_titles"]

        self.assertEqual(set(picks[2]), {"Latium", "Diocletian"})
        # Out of on-topic articles, the next in MMR order is the off-topic one most like the seed.
        self.assertEqual(picks[3][2], "Romania")
        self.assertEqual(set(picks[3][:2]), {"Latium", "Diocletian"})

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

    def test_clue_stage_leaves_missing_clue_unfilled_without_placeholder(self) -> None:
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

        with (
            patch(
                "src.pipeline.WikiClient.fetch_page_extract_with_revid",
                return_value={"title": "Entropy", "extract": "", "revid": None},
            ),
            patch("src.pipeline.WikiClient.fetch_page_extract", return_value={"extract": ""}),
            patch("src.pipeline.WikiClient.fetch_lead_wikitext", return_value={"wikitext": "", "revid": None}),
        ):
            result = run_clue_extraction_stage(
                seed_title="Thermodynamics",
                lang="en",
                diagnostics_path=diagnostics_path,
                terms_path=terms_path,
                clues_path=clues_path,
            )

        self.assertEqual(result.diagnostics["clue_count"], 0)
        self.assertEqual(result.diagnostics["unclued_term_count"], 1)
        clue_payload = clues_path.read_text(encoding="utf-8")
        self.assertNotIn("Theme-related entry from the selected source article.", clue_payload)

        shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_candidate_scoring_escapes_titles_in_csv(self) -> None:
        output_path = Path("tests") / "tmp_outputs" / "scores_quoted.csv"
        diagnostics_path = Path("tests") / "tmp_outputs" / "diagnostics_scores_quoted.json"
        cache_dir = Path("tests") / "tmp_cache_pipeline_scores_quoted"

        if output_path.parent.exists():
            shutil.rmtree(output_path.parent, ignore_errors=True)
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)

        mocked_seed = SimpleNamespace(
            diagnostics={
                "candidates": [
                    {"title": 'Alpha, "quoted" topic', "depth": 1, "links_back_to_seed": True},
                ],
                "errors": [],
            },
            diagnostics_path=diagnostics_path,
        )

        with (
            patch("src.pipeline.run_seed_ingestion_stage", return_value=mocked_seed),
            patch(
                "src.pipeline.WikiClient.fetch_page_extract",
                return_value={"extract": "Alpha quoted topic reference text"},
            ),
            patch("src.pipeline.WikiClient.fetch_lead_wikitext", return_value={"wikitext": ""}),
        ):
            run_candidate_scoring_stage(
                seed_title="Seed",
                lang="en",
                cache_dir=cache_dir,
                diagnostics_path=diagnostics_path,
                scores_path=output_path,
            )

        import csv

        with output_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        self.assertEqual(rows[0]["title"], 'Alpha, "quoted" topic')

        shutil.rmtree(output_path.parent, ignore_errors=True)
        shutil.rmtree(cache_dir, ignore_errors=True)

    def test_rescue_ladder_preserves_term_extraction_configuration(self) -> None:
        tmp_dir = Path("tests") / "tmp_outputs" / "rescue_config_passthrough"
        selected_path = tmp_dir / "selected_candidates.json"
        diagnostics_path = tmp_dir / "diagnostics_rescue.json"

        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        selected_path.write_text(
            json.dumps(
                {
                    "seed_title": "Thermodynamics",
                    "lang": "en",
                    "selected_k": 1,
                    "selected_titles": ["Entropy"],
                }
            ),
            encoding="utf-8",
        )

        term_result = SimpleNamespace(
            terms_path=tmp_dir / "answer_candidates.csv",
            diagnostics_path=tmp_dir / "diagnostics_terms.json",
            terms=[],
            diagnostics={},
        )

        with (
            patch("src.pipeline.run_term_extraction_stage", return_value=term_result) as mocked_terms,
            patch("src.pipeline.run_clue_extraction_stage", return_value=SimpleNamespace(clues=[])) as mocked_clues,
        ):
            result = run_rescue_ladder(
                seed_title="Thermodynamics",
                lang="en",
                cache_dir="data/cache/wiki",
                selected_path=selected_path,
                terms_path=tmp_dir / "answer_candidates.csv",
                diagnostics_path=diagnostics_path,
                min_df=3,
                nlp_backend="nltk",
                entity_type_scoring=True,
                wikidata_cache_dir="data/cache/wikidata",
                lexicon_path="data/lexicon/combined_wordfreq.txt",
                lexicon_weight=0.22,
                candidate_scores_path=tmp_dir / "candidate_scores.csv",
            )

        self.assertFalse(result["passed"])
        first_call = mocked_terms.call_args_list[0].kwargs
        second_call = mocked_terms.call_args_list[1].kwargs
        self.assertEqual(first_call["scores_path"], tmp_dir / "candidate_scores.csv")
        self.assertEqual(first_call["min_df"], 3)
        self.assertEqual(first_call["nlp_backend"], "nltk")
        self.assertTrue(first_call["entity_type_scoring"])
        self.assertEqual(first_call["lexicon_weight"], 0.22)
        self.assertEqual(second_call["min_df"], 1)
        self.assertEqual(second_call["nlp_backend"], "nltk")
        self.assertEqual(mocked_clues.call_args_list[0].kwargs["selected_path"], selected_path)

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

    def _run_clues_for(
        self,
        tmp_name: str,
        term_rows: list[str],
        extracts: dict[str, str],
        selected: list[str] | None = None,
        revid=lambda intro_only: 77 if intro_only else 88,
    ):
        tmp_dir = Path("tests") / "tmp_outputs" / tmp_name
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        (tmp_dir / "selected_candidates.json").write_text(
            json.dumps({"seed_title": "Jazz", "lang": "en", "selected_k": 0, "selected_titles": selected or []}),
            encoding="utf-8",
        )
        (tmp_dir / "terms.csv").write_text(
            "\n".join(["answer,normalized_answer,source_titles,source_method,intro_on_topic", *term_rows]),
            encoding="utf-8",
        )

        def with_revid(title, intro_only=True):
            return {"title": title, "extract": extracts.get(title, ""), "revid": revid(intro_only)}

        with (
            patch("spacy.load", side_effect=OSError("no model in unit test")),
            patch("src.pipeline.WikiClient.fetch_page_extract_with_revid", side_effect=with_revid),
            patch("src.pipeline.WikiClient.fetch_page_extract", side_effect=AssertionError("plain text has no revision")),
            patch("src.pipeline.WikiClient.fetch_lead_wikitext", side_effect=AssertionError("lead revision is another response")),
        ):
            result = run_clue_extraction_stage(
                seed_title="Jazz",
                lang="en",
                diagnostics_path=tmp_dir / "diagnostics_clues.json",
                terms_path=tmp_dir / "terms.csv",
                clues_path=tmp_dir / "clues.csv",
                selected_path=tmp_dir / "selected_candidates.json",
            )
        clues_header = (tmp_dir / "clues.csv").read_text(encoding="utf-8").splitlines()[0]
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return result, clues_header

    def test_clue_stage_gives_article_title_term_a_definition_from_its_own_lead(self) -> None:
        result, header = self._run_clues_for(
            "clue_definition",
            ["Bebop,BEBOP,Bebop,article_title"],
            {"Bebop": "Bebop is a style of jazz developed in the early 1940s. It features fast tempos."},
        )

        self.assertEqual(len(result.clues), 1)
        row = result.clues[0]
        self.assertEqual(row["clue"], "A style of jazz developed in the early 1940s")
        self.assertEqual(row["clue_style"], "definition")
        self.assertEqual(row["clue_class"], "source_backed")
        self.assertEqual((row["source_page"], row["sentence_offset"], row["revid"]), ("Bebop", 0, 77))
        self.assertIn("clue_style", header)
        self.assertEqual(result.diagnostics["definition_clue_count"], 1)

    def test_clue_stage_cites_the_seed_and_selected_pages_from_full_text_and_the_rest_from_the_intro(self) -> None:
        result, _ = self._run_clues_for(
            "clue_revision_source",
            [
                "Jazz,JAZZ,Jazz,article_title",
                "Bebop,BEBOP,Bebop,article_title",
                "Swing,SWING,Swing,article_title",
            ],
            {
                "Jazz": "Jazz is a music genre that originated in New Orleans.",
                "Bebop": "Bebop is a style of jazz developed in the early 1940s.",
                "Swing": "Swing is a style of dance music played by big bands.",
            },
            selected=["Bebop"],
        )

        self.prefetch_intro_extracts.assert_called_once_with(["Swing"], with_revid=True)
        # 88 arrives with a full extract, 77 with an intro; no revision is ever paired with other text.
        self.assertEqual({row["source_page"]: row["revid"] for row in result.clues}, {"Jazz": 88, "Bebop": 88, "Swing": 77})
        self.assertEqual({row["source_page"]: row["oldid_url"][-2:] for row in result.clues}, {"Jazz": "88", "Bebop": "88", "Swing": "77"})

    def test_clue_stage_makes_no_clue_from_text_that_arrived_without_a_revision(self) -> None:
        result, _ = self._run_clues_for(
            "clue_no_revision",
            ["Bebop,BEBOP,Bebop,article_title"],
            {"Bebop": "Bebop is a style of jazz developed in the early 1940s."},
            revid=lambda intro_only: None,
        )

        self.assertEqual(result.clues, [])
        self.assertEqual(result.diagnostics["unclued_term_count"], 1)

    def test_clue_stage_ignores_reference_sections_of_full_text(self) -> None:
        result, _ = self._run_clues_for(
            "clue_reference_sections",
            ["Gillespie,GILLESPIE,Bebop,spacy", "Hollywood,HOLLYWOOD,Bebop,spacy"],
            {
                "Bebop": "Gillespie played bebop with Parker in the early 1940s in New York clubs.\n\n\n"
                "References\nParker recorded the album Dizzy Atmosphere with Hollywood studio players in 1945."
            },
            selected=["Bebop"],
        )

        self.assertEqual([row["answer"] for row in result.clues], ["Gillespie"])

    def test_clue_stage_falls_back_to_blank_clue_when_definition_leaks_answer(self) -> None:
        result, _ = self._run_clues_for(
            "clue_definition_leak",
            ["Swing,SWING,Swing,article_title"],
            {"Swing": "Swing is a swing style of jazz music played in big bands."},
        )

        row = result.clues[0]
        self.assertEqual(row["clue_style"], "fill_in_blank")
        self.assertIn("____", row["clue"])

    def test_clue_stage_uses_blank_clue_for_extracted_terms(self) -> None:
        result, _ = self._run_clues_for(
            "clue_extracted_term",
            ["Bebop,BEBOP,Bebop,spacy"],
            {"Bebop": "Fans say the bebop style of jazz grew fast in the early 1940s in Harlem clubs."},
        )

        self.assertEqual(result.clues[0]["clue_style"], "fill_in_blank")

    def test_clue_stage_does_not_reuse_a_sentence_for_two_answers(self) -> None:
        two_sentences = {
            "Bebop": (
                "Bebop musicians favored a fast tempo in nearly every performance they gave. "
                "Players loved how fast the band could swing through the changes."
            )
        }
        result, _ = self._run_clues_for(
            "clue_shared_sentence",
            ["Tempo,TEMPO,Bebop,spacy", "fast,FAST,Bebop,spacy"],
            two_sentences,
        )

        offsets = {row["answer"]: row["sentence_offset"] for row in result.clues}
        self.assertEqual(offsets, {"Tempo": 0, "fast": 1})

        one_sentence = {"Bebop": "Bebop musicians favored a fast tempo in nearly every performance they gave."}
        result, _ = self._run_clues_for(
            "clue_shared_sentence_single",
            ["Tempo,TEMPO,Bebop,spacy", "fast,FAST,Bebop,spacy"],
            one_sentence,
        )

        self.assertEqual([row["answer"] for row in result.clues], ["Tempo"])
        self.assertEqual(result.diagnostics["unclued_term_count"], 1)

    def test_clue_stage_never_repeats_clue_text(self) -> None:
        result, _ = self._run_clues_for(
            "clue_unique_text",
            ["Bebop,BEBOP,Bebop,article_title", "Bop,BOP,Bop,article_title"],
            {
                "Bebop": "Bebop is a style of jazz developed in the early 1940s.",
                "Bop": "Bop is a style of jazz developed in the early 1940s.",
            },
        )

        styles = {row["answer"]: row["clue_style"] for row in result.clues}
        self.assertEqual(styles, {"Bebop": "definition", "Bop": "fill_in_blank"})
        texts = [row["clue"].lower() for row in result.clues]
        self.assertEqual(len(texts), len(set(texts)))

    def test_clue_stage_lengthens_a_definition_another_answer_already_uses(self) -> None:
        result, _ = self._run_clues_for(
            "clue_definition_collision",
            ["Ann Able,ANNABLE,Ann Able,article_title", "Bo Baker,BOBAKER,Bo Baker,article_title"],
            {
                "Ann Able": "Ann Able was an American theoretical physicist.",
                "Bo Baker": (
                    "Bo Baker was an American theoretical physicist who played a preeminent role in the "
                    "development of the theory of elementary particles."
                ),
            },
        )

        clues = {row["answer"]: row["clue"] for row in result.clues}
        self.assertEqual(clues["Ann Able"], "An American theoretical physicist")
        self.assertEqual(
            clues["Bo Baker"],
            "An American theoretical physicist who played a preeminent role in the development of the "
            "theory of elementary particles",
        )

    def test_clue_stage_drops_generic_definitions_and_prefers_a_longer_cut(self) -> None:
        result, _ = self._run_clues_for(
            "clue_generic_definition",
            [
                "Ana Vale,ANAVALE,Ana Vale,article_title",
                "Ben Cole,BENCOLE,Ben Cole,article_title",
                "Cy Dunn,CYDUNN,Cy Dunn,article_title",
                "Fay Ito,FAYITO,Fay Ito,article_title",
                "Mars,MARS,Mars,article_title",
            ],
            {
                "Ana Vale": "Ana Vale was a noted physicist and teacher.",
                "Ben Cole": "Ben Cole was a noted teacher and physicist.",
                "Cy Dunn": "Cy Dunn was a physicist and noted teacher.",
                "Fay Ito": (
                    "Fay Ito was a noted physicist and teacher who founded the first laboratory for quantum "
                    "optics in Kyoto."
                ),
                "Mars": "Mars is the god of war.",
            },
        )

        definitions = {row["answer"]: row["clue"] for row in result.clues if row["clue_style"] == "definition"}
        self.assertEqual(
            definitions,
            {
                "Fay Ito": "A noted physicist and teacher who founded the first laboratory for quantum optics in Kyoto",
                "Mars": "The god of war",
            },
        )
        # No tagger here, so a generic answer falls back to its lead sentence as a fill-in.
        self.assertEqual(
            next(row["clue_style"] for row in result.clues if row["answer"] == "Ana Vale"), "fill_in_blank"
        )

    def test_clue_stage_prefers_a_complete_sentence_and_skips_two_ended_fragments(self) -> None:
        fragment_then_sentence = (
            "Researchers who studied the matter for many years in several countries finally agreed, "
            "after long debate and careful measurement, that the field of entropy was settled. "
            "Isolated systems never lose entropy overall."
        )
        result, _ = self._run_clues_for(
            "clue_complete_sentence",
            ["Entropy,ENTROPY,Entropy,spacy", "Debate,DEBATE,Entropy,spacy"],
            {"Entropy": fragment_then_sentence},
        )

        clues = {row["answer"]: row["clue"] for row in result.clues}
        self.assertEqual(clues["Entropy"], "Isolated systems never lose ____ overall.")
        # The only window for "debate" would need an ellipsis at both ends.
        self.assertNotIn("Debate", clues)

    def test_clue_stage_clues_a_title_the_seed_merely_names_from_the_seed_and_the_rest_from_their_own_page(self) -> None:
        extracts = {
            "Olive": "An olive is a subspecies of subtropical evergreen tree in the family Oleaceae.",
            "Jazz": "Jazz drummers often sit beside an olive tree outside the clubs.",
        }
        named, _ = self._run_clues_for("clue_named", ["Olive,OLIVE,Olive|Jazz,article_title,False"], extracts)
        on_topic, _ = self._run_clues_for("clue_on_topic", ["Olive,OLIVE,Olive|Jazz,article_title,True"], extracts)

        self.assertEqual([(row["clue_style"], row["source_page"]) for row in named.clues], [("fill_in_blank", "Jazz")])
        self.assertEqual([(row["clue_style"], row["source_page"]) for row in on_topic.clues], [("definition", "Olive")])

    def test_clue_stage_prefers_a_seed_sentence_to_an_equally_clean_one_from_another_page(self) -> None:
        result, _ = self._run_clues_for(
            "clue_seed_sentence",
            ["Tempo,TEMPO,Bebop|Jazz,spacy"],
            {
                "Bebop": "Bebop musicians favored a fast tempo in every show.",
                "Jazz": "Jazz drummers keep a steady tempo while soloists improvise over chords.",
            },
            selected=["Bebop"],
        )

        # Bebop's sentence is the closer fit to nine words; the seed's wins because both are complete sentences.
        self.assertEqual([(row["source_page"], row["revid"]) for row in result.clues], [("Jazz", 88)])

    def test_vocab_gate_counts_distinct_clued_answers(self) -> None:
        tmp_dir = Path("tests") / "tmp_outputs" / "gate_clued"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        url = "https://en.wikipedia.org/w/index.php?title={0}&oldid=5"
        (tmp_dir / "clues.csv").write_text(
            "\n".join(
                [
                    "answer,normalized_answer,clue,clue_class,source_method,source_page,revid,sentence_offset,oldid_url",
                    f"Bebop,BEBOP,A style of jazz,source_backed,article_title,Bebop,5,0,{url.format('Bebop')}",
                    f"Bebop,BEBOP,Another clue,source_backed,article_title,Bebop,5,1,{url.format('Bebop')}",
                    f"Swing,SWING,A dance style,source_backed,article_title,Swing,5,0,{url.format('Swing')}",
                    "Ragtime,RAGTIME,No provenance,source_backed,article_title,Ragtime,,0,",
                ]
            ),
            encoding="utf-8",
        )

        passed = run_vocab_gate_stage(
            seed_title="Jazz",
            diagnostics_path=tmp_dir / "gate.json",
            clues_path=tmp_dir / "clues.csv",
            min_required=2,
        )
        failed = run_vocab_gate_stage(
            seed_title="Jazz",
            diagnostics_path=tmp_dir / "gate.json",
            clues_path=tmp_dir / "clues.csv",
            min_required=3,
        )
        missing = run_vocab_gate_stage(
            seed_title="Jazz",
            diagnostics_path=tmp_dir / "gate.json",
            clues_path=tmp_dir / "absent.csv",
        )

        self.assertTrue(passed.diagnostics["passed"])
        self.assertEqual(passed.diagnostics["term_count"], 2)
        self.assertIsNone(passed.diagnostics["max_allowed"])
        self.assertFalse(failed.diagnostics["passed"])
        self.assertEqual(missing.diagnostics["reason"], "clues_missing")
        shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_term_stage_builds_title_answer_pool_from_candidate_scores(self) -> None:
        tmp_dir = Path("tests") / "tmp_outputs" / "term_title_pool"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        (tmp_dir / "selected_candidates.json").write_text(
            json.dumps({"seed_title": "Jazz", "lang": "en", "selected_k": 1, "selected_titles": ["Bebop"]}),
            encoding="utf-8",
        )
        (tmp_dir / "candidate_scores.csv").write_text(
            "\n".join(
                [
                    "rank,title,depth,links_back_to_seed,rel_score,red_score,selection_score,intro_on_topic,seed_mentions,status,reason",
                    "1,1940s in jazz,1,True,0.9,0.0,0.9,True,0,KEEP,x",
                    "2,Bebop,1,True,0.5,0.0,0.5,True,0,KEEP,x",
                    "3,Swing (jazz performance style),1,True,0.4,0.0,0.4,True,0,KEEP,x",
                    "4,Swings,1,True,0.1,0.0,0.1,False,0,CUT,cut_off_topic",
                    "5,List of jazz musicians,1,True,0.8,0.0,0.8,True,0,KEEP,x",
                    "6,Obscure,1,True,0.3,0.0,0.3,False,2,KEEP,x",
                    "7,Ragtime,1,True,0.7,0.0,0.7,True,0,DROP,x",
                    "8,Barcelona,1,True,0.45,0.0,0.45,False,2,KEEP,x",
                    "9,Jazz fusion,1,True,0.35,0.0,0.35,True,0,KEEP,x",
                    "10,Paris,1,False,0.6,0.0,0.6,False,0,CUT,cut_off_topic",
                    "11,Swing music,1,True,0.2,0.0,0.2,True,0,KEEP,x",
                    "12,Time,1,True,0.3,0.0,0.3,True,0,KEEP,x",
                ]
            ),
            encoding="utf-8",
        )
        extracts = {
            "Jazz": "Jazz is a music genre. It reached Barcelona early. Barcelona still hosts a jazz festival.",
            "Bebop": "Bebop is a style of jazz developed in the early 1940s by Gillespie and other players.",
            "Swing (jazz performance style)": "Swing is a style of jazz.",
            "Swing music": "Swing is a style of jazz.",
            "Paris": "Paris is a city in France.",
            "Time": "Time is a jazz concept.",
            "Swings": "Swings are playground seats.",
            "Barcelona": "Barcelona is a city in Spain.",
            "Jazz fusion": "Jazz fusion mixes jazz with rock.",
            "Obscure": "",
        }

        def intro(title, intro_only=True):
            return {"title": title, "extract": extracts.get(title, "")}

        def with_revid(title, intro_only=True):
            return {"title": title, "extract": extracts.get(title, ""), "revid": 5}

        with (
            patch("src.pipeline._resolve_nlp_backend", return_value=(None, None, [], None, None)),
            patch("src.pipeline.WikiClient.fetch_page_extract", side_effect=intro),
            patch("src.pipeline.WikiClient.fetch_page_extract_with_revid", side_effect=with_revid),
            patch(
                "src.pipeline.WikiClient.fetch_lead_wikitext",
                return_value={"wikitext": "'''Bebop''' by '''Gillespie''' and '''other'''", "revid": 5},
            ),
        ):
            result = run_term_extraction_stage(
                seed_title="Jazz",
                lang="en",
                diagnostics_path=tmp_dir / "diagnostics_terms.json",
                selected_path=tmp_dir / "selected_candidates.json",
                terms_path=tmp_dir / "answer_candidates.csv",
                min_df=1,
                scores_path=tmp_dir / "candidate_scores.csv",
            )

        self.prefetch_lead_wikitext.assert_called_once_with(["Bebop"])
        pool_intros = self.prefetch_intro_extracts.call_args.args[0]
        self.assertEqual(
            (pool_intros[0], "Barcelona" in pool_intros, "Paris" in pool_intros, "Ragtime" in pool_intros),
            ("Jazz", True, False, False),
        )
        by_key = {row["normalized_answer"]: row for row in result.terms}
        # Scoring cut Paris, "Swing music" repeats Swing's page, "Time" is generic.
        self.assertEqual(set(by_key), {"JAZZ", "BEBOP", "SWING", "BARCELONA", "JAZZFUSION", "GILLESPIE"})
        for key, row in by_key.items():
            self.assertEqual(row["source_method"], "lead_bold" if key == "GILLESPIE" else "article_title")
        self.assertEqual(by_key["SWING"]["answer"], "Swing")
        self.assertEqual(by_key["SWING"]["source_titles"], "Swing (jazz performance style)")
        self.assertEqual(by_key["JAZZ"]["source_titles"], "Jazz|Bebop")
        self.assertEqual(by_key["BEBOP"]["source_titles"], "Bebop")
        self.assertEqual(by_key["BARCELONA"]["source_titles"], "Barcelona|Jazz")  # the seed names it
        scores = [row["answer_score"] for row in result.terms]
        self.assertEqual(scores, sorted(scores, reverse=True))
        # On-topic pages beat off-topic ones despite lower relevance; answers repeating the seed word come last.
        self.assertEqual(
            [row["normalized_answer"] for row in result.terms],
            ["JAZZ", "BEBOP", "SWING", "BARCELONA", "JAZZFUSION", "GILLESPIE"],
        )
        header = (tmp_dir / "answer_candidates.csv").read_text(encoding="utf-8").splitlines()[0]
        self.assertTrue(header.startswith("answer,normalized_answer,length,source_method,"))
        shutil.rmtree(tmp_dir, ignore_errors=True)

    def title_pool(
        self,
        seed: str,
        rows: list[str],
        intros: dict[str, str],
        cap: int = 10,
        redirects: dict[str, str] | None = None,
        extracts: dict[str, str] | None = None,
    ) -> list[dict]:
        header = "rank,title,depth,links_back_to_seed,rel_score,red_score,selection_score,intro_on_topic,seed_mentions,status,reason"
        with tempfile.TemporaryDirectory() as tmp:
            scores = Path(tmp) / "candidate_scores.csv"
            scores.write_text("\n".join([header, *rows]), encoding="utf-8")
            client = MagicMock()

            def fetch(title, intro_only=True):
                page = (redirects or {}).get(title, title)
                return {"title": page, "extract": intros.get(page, "")}

            client.fetch_page_extract.side_effect = fetch
            return _build_title_answer_rows(
                seed_title=seed,
                scores_path=scores,
                client=client,
                extracts=extracts or {},
                lexicon={},
                lang="en",
                max_len=15,
                max_title_answers=cap,
                errors=[],
            )

    def test_greek_seed_mentions_tolerate_endings_case_and_accents_but_not_longer_words(self) -> None:
        text = "Η ΠΙΕΣΗ μετράται. Η τιμή της πίεσης αυξάνεται και οι πιέσεις εξισορροπούνται. Η στατιστική μηχανική."
        self.assertEqual(_seed_mentions("Πίεση (φυσική)", text, lang="el"), 3)
        self.assertEqual(_seed_mentions("Στατιστικής μηχανικής", text, lang="el"), 1)
        self.assertEqual(_seed_mentions("Μηχανική στατιστική", text, lang="el"), 0)  # words must be consecutive and in order
        self.assertEqual(_seed_mentions("Ύλη", "Η ύλη και το υλικό. Η μάζα της ύλης.", lang="el"), 2)
        self.assertEqual(_seed_mentions("Ενέργεια", "Η ενέργεια και η ενεργειακή κρίση. Οι ενέργειες.", lang="el"), 2)
        # English stays an exact phrase.
        self.assertEqual(_seed_mentions("Pressure (physics)", "Pressure, pressure. Pressures.", lang="en"), 2)

    def test_definition_rows_follow_the_language_rules(self) -> None:
        lead = "Η Κινηματική είναι κλάδος της μηχανικής που περιγράφει την κίνηση των σωμάτων."
        expected = definition_clue(split_sentences(lead)[0], answer="Κινηματική", max_words=16, lang="el")
        self.assertTrue(expected)

        def rows(lang: str) -> list[dict]:
            return _definition_clue_rows(
                answer="Κινηματική",
                normalized_answer="ΚΙΝΗΜΑΤΙΚΗ",
                answer_key_value="ΚΙΝΗΜΑΤΙΚΗ",
                title="Κινηματική",
                payload={"title": "Κινηματική", "extract": lead, "revid": 1},
                term_source_method="article_title",
                lang=lang,
                nlp=None,
                lead_counts=Counter(),
                pool_size=1,
                seed_title="Φυσική",
            )

        self.assertIn(expected, [row["clue"] for row in rows("el")])
        self.assertEqual(rows("en"), [])
        with patch("src.pipeline.is_generic_definition", return_value=False) as generic:
            rows("el")
        self.assertEqual(generic.call_args.kwargs["seed_title"], "Φυσική")

    def test_title_pool_admits_only_the_kept_titles_scoring_found_on_topic(self) -> None:
        rows = [
            "1,Thermostat,1,True,0.5,0.0,0.5,False,0,CUT,cut_off_topic",
            "2,Fan,1,True,0.4,0.0,0.4,False,1,CUT,cut_off_topic",
            "3,Entropy,1,False,0.4,0.0,0.4,False,2,BORDERLINE,x",
            "4,Roman,1,True,0.4,0.0,0.4,False,0,CUT,cut_off_topic",
        ]
        intros = {
            "Thermodynamics": "Thermodynamics is a branch of physics.",
            "Thermostat": "A thermostat regulates temperature, and a thermometer reads it.",
            "Fan": "A fan moves air.",
            "Entropy": "Entropy measures disorder.",
            "Roman": "Roman law developed over centuries.",
        }
        pool = self.title_pool("Thermodynamics", rows, intros)
        self.assertEqual([row["answer"] for row in pool], ["Thermodynamics", "Entropy"])

    def test_title_pool_scores_by_the_signals_in_the_csv(self) -> None:
        rows = [
            "1,Romania,1,True,0.5,0.0,0.5,False,0,CUT,cut_off_topic",
            "2,Romagna,1,True,0.5,0.0,0.5,False,0,CUT,cut_off_topic",
            "3,Romanticism,1,True,0.5,0.0,0.5,False,0,CUT,cut_off_topic",
            "4,Five Eyes,1,True,0.5,0.0,0.5,False,0,CUT,cut_off_topic",
            "5,Hannibal,1,True,0.4,0.0,0.4,False,1,CUT,cut_off_topic",
            "6,Diocletian,1,False,0.4,0.0,0.4,False,2,KEEP,x",
            "7,Pompey,1,False,0.4,0.0,0.4,False,12,KEEP,x",
            "8,Latium,1,False,0.4,0.0,0.4,True,0,KEEP,x",
            "9,Rome (city),1,False,0.4,0.0,0.4,True,0,KEEP,x",
            "10,Roman emperor,1,False,0.4,0.0,0.4,True,0,KEEP,x",
        ]
        intros = {
            "Ancient Rome": "Ancient Rome was a civilization.",
            "Roman emperor": "The Roman emperor ruled the empire.",
            "Romania": "Romania is a country in Europe.",
            "Romagna": "Romagna is a region of Italy.",
            "Romanticism": "Romanticism was an artistic movement.",
            "Five Eyes": "The Five Eyes is an intelligence alliance.",
            "Hannibal": "Hannibal was a Carthaginian general.",
            "Diocletian": "Diocletian was an emperor.",
            "Pompey": "Pompey was a general.",
            "Latium": "Latium is the region around Rome.",
            "Rome (city)": "Rome is the capital of Italy.",
        }
        pool = self.title_pool("Ancient Rome", rows, intros)
        # Pompey is named often but the bonus stops at ten mentions.
        scores = {row["answer"]: row["answer_score"] for row in pool}
        self.assertEqual(set(scores), {"Ancient Rome", "Pompey", "Latium", "Diocletian", "Rome", "Roman emperor"})
        self.assertEqual(scores["Pompey"], 2.4)  # 1 + 0.4 + 0.1 * 10
        self.assertEqual(scores["Diocletian"], 1.6)  # two mentions, no intro mention
        self.assertEqual(scores["Latium"], 1.9)  # the intro mentions Rome: 0.5
        self.assertEqual(scores["Rome"], 0.9)  # on topic, but it repeats a seed word: -1
        self.assertEqual(scores["Roman emperor"], 1.9)  # a derived form of the seed word is no repeat

    def test_title_pool_cap_keeps_the_titles_with_the_most_evidence_not_the_highest_tfidf(self) -> None:
        rows = [
            "1,Alpha,1,True,0.9,0.0,0.9,False,0,BORDERLINE,x",
            "2,Bravo,1,True,0.3,0.0,0.3,False,37,KEEP,x",
            "3,Gamma,1,True,0.5,0.0,0.5,False,5,KEEP,x",
        ]
        intros = {
            "Jazz": "Jazz is music.",
            "Alpha": "Alpha is a letter.",
            "Bravo": "Bravo is a word.",
            "Gamma": "Gamma is another.",
        }
        pool = self.title_pool("Jazz", rows, intros, cap=2)
        # The seed row counts toward the cap; the mention bonus runs to ten, so Bravo beats the higher tf-idf Alpha.
        self.assertEqual([row["answer"] for row in pool], ["Jazz", "Bravo"])

    def test_title_pool_takes_fill_in_sources_only_from_selected_articles_with_an_on_topic_intro(self) -> None:
        rows = [
            "1,Bebop,1,True,0.6,0.0,0.6,True,0,KEEP,x",
            "2,Cool,1,True,0.5,0.0,0.5,True,0,KEEP,x",
            "3,Harlem,1,True,0.4,0.0,0.4,False,2,KEEP,x",
        ]
        intros = {
            "Jazz": "Jazz is music.",
            "Bebop": "Bebop is a style of jazz.",
            "Cool": "Cool is a style of jazz.",
            "Harlem": "Harlem is a district.",
        }
        extracts = {"Cool": "Bebop is fast.", "Harlem": "Bebop was played in Harlem."}
        pool = self.title_pool("Jazz", rows, intros, extracts=extracts)
        # Harlem is in the pool because the seed names it; its text says nothing about the theme.
        self.assertEqual({row["answer"]: row["source_titles"] for row in pool}["Bebop"], "Bebop|Cool")

    def test_title_pool_names_a_redirect_after_the_page_it_leads_to(self) -> None:
        rows = [
            "1,Big bands,1,True,0.6,0.0,0.6,True,3,KEEP,x",
            "2,Big band,1,True,0.4,0.0,0.4,True,0,KEEP,x",
        ]
        intros = {"Jazz": "Jazz is music.", "Big band": "A big band is a type of jazz ensemble."}
        pool = self.title_pool("Jazz", rows, intros, redirects={"Big bands": "Big band"})
        big_band = next(row for row in pool if row["answer"] != "Jazz")
        self.assertEqual([row["answer"] for row in pool], ["Jazz", "Big band"])
        # The signals stay those of the link title reached first; the page leads the sources.
        self.assertEqual((big_band["source_titles"], big_band["theme_score"]), ("Big band|Jazz", 0.6))

    def test_title_pool_keeps_a_redirects_name_when_its_page_title_is_no_answer(self) -> None:
        rows = ["1,Hugh Everett,1,True,0.6,0.0,0.6,True,0,KEEP,x"]
        page = "Hugh Everett the Third Person"
        intros = {"Jazz": "Jazz is music.", page: "Hugh Everett was a jazz physicist."}
        pool = self.title_pool("Jazz", rows, intros, redirects={"Hugh Everett": page})
        self.assertEqual(
            [(row["answer"], row["source_titles"]) for row in pool], [("Jazz", "Jazz"), ("Hugh Everett", page)]
        )

    def test_title_pool_cites_the_seed_article_after_the_titles_own_page_when_it_names_the_title(self) -> None:
        rows = [
            "1,Pompey,1,True,0.4,0.0,0.4,False,3,KEEP,x",
            "2,Latium,1,True,0.4,0.0,0.4,True,0,KEEP,x",
        ]
        intros = {
            "Ancient Rome": "Ancient Rome was a civilization.",
            "Pompey": "Pompey was a general.",
            "Latium": "Latium is near Rome.",
        }
        pool = self.title_pool("Ancient Rome", rows, intros)
        self.assertEqual(
            {row["answer"]: row["source_titles"] for row in pool},
            {"Ancient Rome": "Ancient Rome", "Pompey": "Pompey|Ancient Rome", "Latium": "Latium"},
        )

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


CLUE_HEADER = [
    "answer", "normalized_answer", "clue", "clue_score", "clue_class", "clue_style",
    "source_method", "source_page", "revid", "sentence_offset", "oldid_url",
]


def _clue_row(answer: str, normalized: str, *, revid: str = "7", style: str = "definition") -> list[str]:
    oldid = f"https://en.wikipedia.org/w/index.php?title=Test&oldid={revid}" if revid else ""
    return [
        answer, normalized, f"Clue for {normalized.lower()}", "0.9", "source_backed", style,
        "spacy", "Test", revid, "0", oldid,
    ]


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


class LayoutAndPackagingTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def _fixture(self, words: list[str]) -> None:
        _write_csv(
            self.tmp / "terms.csv",
            ["answer", "normalized_answer", "answer_score"],
            [[word.title(), word, str(1.0 - 0.01 * i)] for i, word in enumerate(words)],
        )
        _write_csv(
            self.tmp / "clues.csv",
            CLUE_HEADER,
            [_clue_row(word.title(), word, style="fill_in_blank" if i % 4 == 0 else "definition")
             for i, word in enumerate(words)],
        )
        (self.tmp / "selected.json").write_text(
            json.dumps({"seed_title": "Test", "lang": "en", "selected_k": 1, "selected_titles": ["Seed"]}),
            encoding="utf-8",
        )

    def _layout_and_package(self, *, min_words: int, gate_min: int) -> tuple[dict, dict]:
        layout = run_layout_stage(
            seed_title="Test",
            terms_path=self.tmp / "terms.csv",
            clues_path=self.tmp / "clues.csv",
            grid_path=self.tmp / "grid.json",
            diagnostics_path=self.tmp / "diagnostics_layout.json",
            restarts=4,
            min_words=min_words,
            gate_min=gate_min,
        )
        package = run_packaging_stage(
            seed_title="Test",
            selected_path=self.tmp / "selected.json",
            grid_path=layout.grid_path,
            clues_path=self.tmp / "clues.csv",
            puzzle_path=self.tmp / "puzzle.json",
            attribution_path=self.tmp / "attribution.json",
            diagnostics_path=self.tmp / "diagnostics_package.json",
            gate_min=gate_min,
        )
        return layout.diagnostics, json.loads(package.puzzle_path.read_text(encoding="utf-8"))

    def test_layout_candidates_are_packageable_ranked_and_share_no_long_word(self) -> None:
        # (answer, normalized, score, has a clue with a revision id)
        rows = [
            ("Heat engine", "HEATENGINE", 0.5, True),
            ("Heat", "HEAT", 0.9, True),
            ("HEAT", "HEAT", 0.1, True),
            ("Jazz", "JAZZ", 0.8, True),
            ("Acid jazz", "ACIDJAZZ", 0.7, True),
            ("Entropy", "ENTROPY", 0.6, False),
            ("Carnot cycle", "CARNOTCYCLE", 0.4, True),
            ("Gas", "GAS", 0.3, True),
            ("Gas giant", "GASGIANT", 0.25, True),
            ("Café", "CAFE", 0.2, True),
            ("Cafe society", "CAFESOCIETY", 0.15, True),
        ]
        terms = [
            {"answer": a, "normalized_answer": n, "answer_score": str(score)} for a, n, score, _ in rows
        ]
        clues = [dict(zip(CLUE_HEADER, _clue_row(a, n, revid="7" if clued else ""))) for a, n, _, clued in rows]

        picked = _layout_candidates(terms, clues, lang="en", limit=80)
        self.assertEqual(
            [key for key, _ in picked],
            ["HEAT", "JAZZ", "CARNOTCYCLE", "GAS", "GASGIANT", "CAFE"],
        )
        self.assertEqual(picked[0], ("HEAT", 0.9))
        capped = _layout_candidates(terms, clues, lang="en", limit=3)
        self.assertEqual([key for key, _ in capped], ["HEAT", "JAZZ", "CARNOTCYCLE"])

    def test_layout_fill_status_and_generated_puzzle_validity(self) -> None:
        words = [
            "HEAT", "ENERGY", "ENTROPY", "WORK", "TEMPERATURE", "PRESSURE", "VOLUME", "ENGINE",
            "CARNOT", "KELVIN", "JOULE", "FLUID", "PHASE", "STATE", "CYCLE", "BOILING", "SYSTEM",
        ]
        self._fixture(words)

        layout, puzzle = self._layout_and_package(min_words=6, gate_min=5)
        self.assertEqual(layout["fill_status"], "complete")
        self.assertEqual(layout["validity_errors"], [])
        self.assertGreaterEqual(layout["word_count"], 6)
        self.assertEqual(puzzle["fill_status"], "complete")
        self.assertEqual(puzzle["puzzle_status"], "ok")
        self.assertEqual(validate_puzzle(puzzle), [])
        self.assertEqual(
            len(puzzle["across_entries"]) + len(puzzle["down_entries"]), layout["word_count"]
        )

        layout, puzzle = self._layout_and_package(min_words=99, gate_min=5)
        self.assertEqual(layout["fill_status"], "failed")
        self.assertEqual(layout["validity_errors"], [])
        self.assertEqual(puzzle["puzzle_status"], "insufficient_quality")

        layout, puzzle = self._layout_and_package(min_words=6, gate_min=99)
        self.assertEqual(layout["fill_status"], "failed")
        self.assertEqual(layout["word_count"], 0)
        self.assertEqual(layout["errors"], ["vocab_gate_failed"])
        self.assertTrue((self.tmp / "grid.json").exists())
        self.assertTrue(all(cell == "#" for row in puzzle["grid_cells"] for cell in row))
        self.assertEqual(puzzle["puzzle_status"], "insufficient_vocabulary")

    def test_packaging_numbers_enumerates_and_reports_each_status(self) -> None:
        grid = [["#"] * 11 for _ in range(11)]
        for col, letter in enumerate("MILESDAVIS"):
            grid[0][col] = letter
        for row, letter in enumerate("MILK"):
            grid[row][0] = letter
        placements = [
            {"answer": "MILESDAVIS", "row": 0, "col": 0, "direction": "across"},
            {"answer": "MILK", "row": 0, "col": 0, "direction": "down"},
        ]
        (self.tmp / "selected.json").write_text(
            json.dumps({"seed_title": "Test", "lang": "en", "selected_k": 1, "selected_titles": ["Seed"]}),
            encoding="utf-8",
        )

        def package(*, fill_status: str, milk_revid: str, gate_min: int):
            (self.tmp / "grid.json").write_text(
                json.dumps(
                    {
                        "seed_title": "Test", "lang": "en", "size": 11, "grid": grid,
                        "placements": placements, "fill_status": fill_status,
                    }
                ),
                encoding="utf-8",
            )
            _write_csv(
                self.tmp / "clues.csv",
                CLUE_HEADER,
                [
                    _clue_row("Miles Davis", "MILESDAVIS"),
                    _clue_row("Milk", "MILK", revid=milk_revid, style="fill_in_blank"),
                ],
            )
            result = run_packaging_stage(
                seed_title="Test",
                selected_path=self.tmp / "selected.json",
                grid_path=self.tmp / "grid.json",
                clues_path=self.tmp / "clues.csv",
                puzzle_path=self.tmp / "puzzle.json",
                attribution_path=self.tmp / "attribution.json",
                diagnostics_path=self.tmp / "diagnostics_package.json",
                gate_min=gate_min,
            )
            return result, json.loads(result.puzzle_path.read_text(encoding="utf-8"))

        result, puzzle = package(fill_status="complete", milk_revid="7", gate_min=2)
        self.assertEqual(puzzle["puzzle_status"], "ok")
        self.assertEqual(validate_puzzle(puzzle), [])
        across, down = puzzle["across_entries"][0], puzzle["down_entries"][0]
        self.assertEqual((across["number"], down["number"]), (1, 1))
        self.assertEqual((across["answer"], across["display_answer"]), ("MILESDAVIS", "Miles Davis"))
        self.assertEqual((across["enumeration"], down["enumeration"]), ("5,5", "4"))
        self.assertEqual((across["clue_style"], down["clue_style"]), ("definition", "fill_in_blank"))
        self.assertEqual((across["row"], across["col"], across["length"]), (0, 0, 10))
        self.assertEqual(puzzle["layout"]["word_count"], 2)
        self.assertEqual(len(puzzle["attribution"]), 2)
        self.assertEqual(
            {key: result.diagnostics[key] for key in (
                "entry_count", "definition_clue_count", "fill_in_blank_clue_count",
                "duplicate_clue_text_count", "used_clue_provenance_missing_count",
            )},
            {
                "entry_count": 2, "definition_clue_count": 1, "fill_in_blank_clue_count": 1,
                "duplicate_clue_text_count": 0, "used_clue_provenance_missing_count": 0,
            },
        )

        self.assertEqual(
            package(fill_status="complete", milk_revid="7", gate_min=3)[1]["puzzle_status"],
            "insufficient_vocabulary",
        )
        self.assertEqual(
            package(fill_status="complete", milk_revid="", gate_min=1)[1]["puzzle_status"],
            "insufficient_clues",
        )
        self.assertEqual(
            package(fill_status="failed", milk_revid="7", gate_min=2)[1]["puzzle_status"],
            "insufficient_quality",
        )


if __name__ == "__main__":
    unittest.main()
