import csv
import json
import shutil
import tempfile
import unittest
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.clue_builder import definition_clue, split_sentences
from src.layout import validate_puzzle
from src.pipeline import (
    run_candidate_scoring_stage,
    run_clue_extraction_stage,
    run_generate_pipeline,
    run_k_selection_stage,
    run_layout_stage,
    run_packaging_stage,
    run_rescue_ladder,
    run_seed_ingestion_stage,
    run_term_extraction_stage,
    run_vocab_gate_stage,
)
from src.pipeline.clues import _collect_source_backed_clue_candidates, _definition_clue_rows
from src.pipeline.common import _fetch_extract_with_revid, _uses_seed_word
from src.pipeline.generate import _linked_titles
from src.pipeline.packaging import _layout_candidates
from src.pipeline.scoring import _candidate_status, _effective_min_k_for_size, _inventory_min_k_target, _seed_mentions
from src.pipeline.titles import _build_title_answer_rows, _is_person, _seed_short_names


class PipelineTests(unittest.TestCase):
    def test_an_on_topic_candidate_is_never_cut_for_low_similarity(self) -> None:
        # group 0: the intro is on topic; 1: the seed names the page twice; 2: neither.
        self.assertEqual(_candidate_status(0), ("KEEP", "on_topic"))
        self.assertEqual(_candidate_status(1), ("KEEP", "on_topic"))
        self.assertEqual(_candidate_status(2), ("CUT", "cut_off_topic"))

    def setUp(self) -> None:
        # Stage tests never reach the network; the batched prefetch is checked where it matters.
        for name in ("prefetch_intro_extracts", "prefetch_lead_wikitext"):
            patcher = patch(f"src.wiki_client.WikiClient.{name}")
            setattr(self, name, patcher.start())
            self.addCleanup(patcher.stop)
        patcher = patch(
            "src.wiki_client.WikiClient.fetch_page_extract_with_revid",
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

        with patch("src.wiki_client.WikiClient.fetch_links", side_effect=RuntimeError("network blocked")):
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

        with patch("src.wiki_client.WikiClient.fetch_links", return_value=payload):
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
            patch("src.wiki_client.WikiClient.fetch_links", return_value=payload),
            patch("src.wiki_client.WikiClient.fetch_page_extract", return_value={"extract": "some text"}),
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
            patch("src.pipeline.scoring.run_seed_ingestion_stage", return_value=seed_result),
            patch("src.wiki_client.WikiClient.fetch_page_extract", return_value={"extract": "reference text"}),
            patch("src.wiki_client.WikiClient.fetch_lead_wikitext", return_value={"wikitext": ""}),
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

    def test_failed_full_text_is_recorded_and_the_intro_stands_in(self) -> None:
        def fetch(title, intro_only=True):
            if not intro_only:
                raise RuntimeError("network blocked")
            return {"title": title, "extract": "Jazz is a genre.", "revid": 7}

        client = MagicMock()
        client.fetch_page_extract_with_revid.side_effect = fetch
        errors: list[str] = []

        payload = _fetch_extract_with_revid(errors, client, "Jazz", full=True)

        self.assertEqual((payload["extract"], payload["revid"]), ("Jazz is a genre.", 7))
        self.assertEqual(errors, ["extract_with_revid_failed:Jazz:network blocked"])

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

        with patch("src.pipeline.scoring.run_seed_ingestion_stage") as mocked_seed:
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

        with patch("src.pipeline.scoring.run_candidate_scoring_stage") as mocked_score:
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

        with patch("src.pipeline.scoring.run_candidate_scoring_stage") as mocked_score:
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

        def fake_extract(title, intro_only=True):  # noqa: ARG001
            return {"extract": f"{title} reference text"}

        with (
            patch("src.pipeline.scoring.run_seed_ingestion_stage", return_value=mocked_seed),
            patch("src.wiki_client.WikiClient.fetch_page_extract", side_effect=fake_extract),
            patch("src.wiki_client.WikiClient.fetch_lead_wikitext", return_value={"wikitext": ""}),
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
        pages: dict[str, str] | None = None,
        max_len: int = 15,
        backlink_titles: list[str] | None = None,
        seed_title: str = "Ancient Rome",
    ):
        intros = intros or self.ROME_INTROS
        pages = pages or {}  # a redirect: link title -> the title of the page it leads to
        extras = [*(backlink_titles or []), *(extra_titles or [])]  # not linked by the seed
        candidates = [
            {"title": title, "depth": 1, "links_back_to_seed": True} for title in list(intros)[1:] if title not in extras
        ]
        seed_result = SimpleNamespace(diagnostics={"candidates": candidates, "errors": []})
        seed_text = seed_text or "Hannibal crossed the Alps. Diocletian split the empire; later Diocletian retired."
        with (
            patch("src.pipeline.scoring.run_seed_ingestion_stage", return_value=seed_result),
            patch(
                "src.wiki_client.WikiClient.fetch_page_extract",
                side_effect=lambda title, intro_only=True: {"title": pages.get(title, title), "extract": intros[title]},
            ),
            patch(
                "src.wiki_client.WikiClient.fetch_page_extract_with_revid",
                return_value={"extract": seed_text, "revid": 3},
            ),
            patch("src.wiki_client.WikiClient.fetch_lead_wikitext", return_value={"wikitext": ""}),
        ):
            return run_candidate_scoring_stage(
                seed_title=seed_title,
                cache_dir=Path(tmp) / "cache",
                diagnostics_path=Path(tmp) / "diagnostics.json",
                scores_path=Path(tmp) / "scores.csv",
                lexicon_path=None,
                backlink_titles=backlink_titles,
                extra_titles=extra_titles,
                max_len=max_len,
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

    def test_scoring_scores_backlink_titles_as_one_hop_pages_that_link_back(self) -> None:
        intros = {**self.ROME_INTROS, "Roman law": "Roman law developed over centuries.", "Senate": "Rome had a senate."}
        with tempfile.TemporaryDirectory() as tmp:
            self.score_rome(tmp, intros, backlink_titles=["Senate"], extra_titles=["Roman law"])
            with (Path(tmp) / "scores.csv").open(encoding="utf-8", newline="") as handle:
                csv_rows = {row["title"]: row for row in csv.DictReader(handle)}

        self.assertEqual((csv_rows["Senate"]["depth"], csv_rows["Senate"]["links_back_to_seed"]), ("1", "True"))
        self.assertEqual((csv_rows["Roman law"]["depth"], csv_rows["Roman law"]["links_back_to_seed"]), ("2", "False"))

    def test_scoring_does_not_count_an_adverb_of_the_seed_word_as_a_mention_by_a_backlink_page(self) -> None:
        intros = {
            "Thermodynamics": "Thermodynamics is a branch of physics.",
            "Heat": "Heat is energy in transit.",
            "Diamond": "Diamond is a solid. It is thermodynamically unstable.",
            "Cycle": "A cycle repeats. It is a thermodynamic process.",
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = self.score_rome(
                tmp, intros, "Heat flows.", backlink_titles=["Diamond", "Cycle"], seed_title="Thermodynamics"
            )

        status = {row["title"]: row["status"] for row in result.scores}
        self.assertEqual((status["Diamond"], status["Cycle"]), ("CUT", "KEEP"))  # an adjective still names the field

    def test_scoring_counts_the_seed_word_in_its_own_form_as_a_mention_whatever_the_tagger_says(self) -> None:
        intros = {
            "Thermodynamics": "Thermodynamics is a branch of physics.",
            "Heat": "Heat is energy in transit.",
            "Engine": "An engine moves. It obeys Thermodynamics.",
            "Diamond": "Diamond is a solid. It is thermodynamically unstable.",
        }

        def tagger(text):  # tags every word as an adverb
            return [SimpleNamespace(text=word, pos_="ADV") for word in text.split()]

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("src.pipeline.scoring._resolve_nlp_backend", return_value=(tagger, None, [], None, None)),
        ):
            result = self.score_rome(
                tmp, intros, "Heat flows.", backlink_titles=["Engine", "Diamond"], seed_title="Thermodynamics"
            )

        status = {row["title"]: row["status"] for row in result.scores}
        self.assertEqual((status["Engine"], status["Diamond"]), ("KEEP", "CUT"))  # only another form is no mention

    def test_scoring_ranks_a_backlink_page_by_whether_its_first_sentence_uses_the_seed_word(self) -> None:
        late = "Plankton drifts. It was once studied in Rome."
        intros = {**self.ROME_INTROS, "Senate": "Rome had a senate.", "Algae": late, "Plankton": late, "Reef": "Reefs form."}
        with tempfile.TemporaryDirectory() as tmp:
            result = self.score_rome(tmp, intros, backlink_titles=["Senate", "Plankton", "Reef"])
            with (Path(tmp) / "scores.csv").open(encoding="utf-8", newline="") as handle:
                csv_rows = {row["title"]: row for row in csv.DictReader(handle)}

        order = [row["title"] for row in result.scores]
        self.assertEqual([(csv_rows[t]["intro_on_topic"], csv_rows[t]["status"]) for t in ("Senate", "Plankton", "Reef")],
                         [("True", "KEEP"), ("False", "KEEP"), ("False", "CUT")])
        self.assertLess(order.index("Senate"), order.index("Plankton"))  # the second tier ranks after every on-topic page
        self.assertLess(order.index("Plankton"), order.index("Reef"))
        # A page the seed links itself stays on topic by a later sentence.
        self.assertEqual(csv_rows["Algae"]["intro_on_topic"], "True")

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
        with patch("src.pipeline.generate._RESCUE_NEW_PAGES", 6):  # a share of two per parent
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
                patch("src.pipeline.generate.run_term_extraction_stage", return_value=SimpleNamespace(terms=[])) as terms,
                patch("src.pipeline.generate.evaluate_vocab_gate", side_effect=gates),
                patch(
                    "src.wiki_client.WikiClient.fetch_links_concurrent",
                    return_value=({"On": {"links": [{"title": "Heat"}, {"title": "Cold"}, {"title": "On"}]}}, {}),
                ) as links,
                patch("src.pipeline.generate.run_candidate_scoring_stage", return_value=widened) as rescore,
                patch("src.wiki_client.WikiClient.fetch_links", return_value={"seed_title": "Seed"}),
                patch("src.wiki_client.WikiClient.fetch_backlinks", return_value=set()),
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

    def run_backlink_ladder(
        self,
        tmp_dir: Path,
        gates: list,
        backlinks: set[str] | Exception,
        scoring_args: dict | None,
        scored: list,
        selected: tuple[str, ...] = ("On", "Named"),
    ):
        """The ladder up to its backlink step: a scores CSV of On and Named, backlinks fetched or failing, and the
        rescue scoring stage answering with the given results in turn. Returns the result, the stage mock and the
        backlink fetch mock."""
        (tmp_dir / "selected.json").write_text(json.dumps({"selected_titles": list(selected)}), encoding="utf-8")
        (tmp_dir / "scores.csv").write_text(
            "rank,title,depth,links_back_to_seed,rel_score,intro_on_topic,seed_mentions,status,reason\n"
            "1,On,1,True,0.5,True,0,KEEP,x\n"
            "2,Named,1,True,0.5,False,2,KEEP,x\n",
            encoding="utf-8",
        )
        fetch = (
            {"side_effect": backlinks} if isinstance(backlinks, Exception) else {"return_value": backlinks}
        )
        with (
            patch("src.pipeline.generate.run_term_extraction_stage", return_value=SimpleNamespace(terms=[])),
            patch("src.pipeline.generate.evaluate_vocab_gate", side_effect=gates),
            patch("src.wiki_client.WikiClient.fetch_links", return_value={"seed_title": "Seed"}),
            patch("src.wiki_client.WikiClient.fetch_backlinks", **fetch) as fetched,
            patch(
                "src.wiki_client.WikiClient.fetch_links_concurrent",
                side_effect=lambda parents, max_links: (
                    {parent: {"links": [{"title": "Heat"}, {"title": "Ion"}]} for parent in parents},
                    {},
                ),
            ),
            patch("src.pipeline.generate.run_candidate_scoring_stage", side_effect=scored) as rescore,
        ):
            result = run_rescue_ladder(
                seed_title="Seed",
                selected_path=tmp_dir / "selected.json",
                terms_path=tmp_dir / "terms.csv",
                diagnostics_path=tmp_dir / "diagnostics_rescue.json",
                candidate_scores_path=tmp_dir / "scores.csv",
                terms_diagnostics_path=tmp_dir / "diagnostics_terms_rescue.json",
                selected_override_path=tmp_dir / "selected_rescue.json",
                scoring_args=scoring_args,
            )
        return result, rescore, fetched

    def test_rescue_ladder_widens_from_the_pages_that_link_to_the_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            widened = SimpleNamespace(
                scores_path=tmp_dir / "candidate_scores_rescue.csv",
                scores=[
                    {"title": "Ion", "status": "KEEP"},
                    {"title": "On", "status": "KEEP"},
                    {"title": "Cold", "status": "CUT"},
                    {"title": "Heat", "status": "KEEP"},
                ],
            )
            gates = [SimpleNamespace(passed=False)] * 3 + [SimpleNamespace(passed=True)]
            result, rescore, fetched = self.run_backlink_ladder(
                tmp_dir, gates, {"Seed", "On", "Heat", "Cold", "Ion"}, {"seed_title": "Seed", "max_backlinks": 50}, [widened]
            )
            selected = json.loads((tmp_dir / "selected_rescue.json").read_text(encoding="utf-8"))["selected_titles"]

        # The seed and the page the scores list are left out; the rest are scored in title order as backlink pages.
        fetched.assert_called_once_with("Seed", max_backlinks=50)
        self.assertEqual(rescore.call_args.kwargs["backlink_titles"], ["Cold", "Heat", "Ion"])
        self.assertNotIn("extra_titles", rescore.call_args.kwargs)
        self.assertEqual(result["diagnostics"]["reason"], "widen_from_backlinks")
        self.assertEqual(result["diagnostics"]["steps"][-1], {"action": "widen_from_backlinks", "value": 3})
        self.assertEqual(selected, ["On", "Named", "Ion", "Heat"])  # new pages in rank order, the cut one skipped

    def test_rescue_ladder_replaces_cut_selected_articles_with_further_new_pages(self) -> None:
        # The step before promotes On and Named into the selection; of 9 new pages two replace cut articles, of 7 one.
        for new_count, kept, joined in ((9, ["K1", "K2"], 8), (7, ["K1", "C1", "K2"], 7)):
            with self.subTest(new_count=new_count), tempfile.TemporaryDirectory() as tmp:
                new = [f"n{i}" for i in range(1, new_count + 1)]
                rows = [{"title": t, "status": "CUT" if t[0] == "C" else "KEEP"} for t in ("K1", "C1", "K2", "C2")]
                widened = SimpleNamespace(
                    scores_path=Path(tmp) / "scores.csv", scores=rows + [{"title": t, "status": "KEEP"} for t in new]
                )
                gates = [SimpleNamespace(passed=False)] * 3 + [SimpleNamespace(passed=True)]
                self.run_backlink_ladder(
                    Path(tmp), gates, set(new), {"seed_title": "Seed"}, [widened], selected=("K1", "C1", "K2", "C2")
                )
                selected = json.loads((Path(tmp) / "selected_rescue.json").read_text(encoding="utf-8"))["selected_titles"]

            # The best 6 join; each further page takes the place of the last cut article, none leaves unreplaced.
            self.assertEqual(selected, [*kept, "On", "Named", *new[:joined]])

    def test_rescue_ladder_scores_an_even_sample_of_the_backlinks_above_the_cap(self) -> None:
        backlinks = {f"b{i}" for i in range(10)} | {"On", "Seed"}
        with tempfile.TemporaryDirectory() as tmp:
            widened = SimpleNamespace(scores_path=Path(tmp) / "scores.csv", scores=[])
            with patch("src.pipeline.generate._RESCUE_BACKLINKS", 4):
                result, rescore, _ = self.run_backlink_ladder(
                    Path(tmp), [SimpleNamespace(passed=False, term_count=0)] * 9, backlinks, {"seed_title": "Seed"}, [widened] * 2
                )

        self.assertEqual(rescore.call_args_list[0].kwargs["backlink_titles"], ["b0", "b2", "b5", "b7"])
        self.assertEqual(result["diagnostics"]["steps"][2], {"action": "widen_from_backlinks", "value": 4})

    def test_rescue_ladder_skips_the_backlink_step_without_scoring_arguments_or_backlinks(self) -> None:
        cases = {
            "no scoring arguments": (None, {"On"}),
            "backlinks off": ({"seed_title": "Seed", "include_backlinks": False}, {"On"}),
            "fetch fails": ({"seed_title": "Seed"}, RuntimeError("offline_cache_miss:w_api_query_backlinks")),
        }
        for name, (scoring_args, backlinks) in cases.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as tmp:
                forward = SimpleNamespace(scores_path=Path(tmp) / "scores.csv", scores=[])
                result, rescore, fetched = self.run_backlink_ladder(
                    Path(tmp), [SimpleNamespace(passed=False, term_count=0)] * 9, backlinks, scoring_args, [forward]
                )

                diagnostics = result["diagnostics"]
                self.assertEqual(diagnostics["steps"][2], {"action": "widen_from_backlinks", "value": 0})
                self.assertFalse(any(call.kwargs["backlink_titles"] for call in rescore.call_args_list))  # no page scored
                self.assertEqual(fetched.called, name == "fetch fails")
                self.assertEqual(
                    diagnostics["errors"],
                    ["backlink_fetch_failed:offline_cache_miss:w_api_query_backlinks"] if name == "fetch fails" else [],
                )

    def test_rescue_ladder_widens_from_selected_on_top_of_a_backlink_step_that_did_not_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            rescue_scores = tmp_dir / "candidate_scores_rescue.csv"
            rescue_scores.write_text(
                "rank,title,depth,links_back_to_seed,rel_score,intro_on_topic,seed_mentions,status,reason\n"
                "1,On,1,True,0.5,True,0,KEEP,x\n"
                "2,Ion,1,True,0.5,True,0,KEEP,x\n",
                encoding="utf-8",
            )
            backlinked = SimpleNamespace(scores_path=rescue_scores, scores=[{"title": "Ion", "status": "KEEP"}])
            forward = SimpleNamespace(
                scores_path=rescue_scores, scores=[{"title": "Heat", "depth": 2, "status": "KEEP"}]
            )
            gates = [SimpleNamespace(passed=False)] * 4 + [SimpleNamespace(passed=True)]
            result, rescore, _ = self.run_backlink_ladder(
                tmp_dir, gates, {"Ion"}, {"seed_title": "Seed"}, [backlinked, forward]
            )
            selected = json.loads((tmp_dir / "selected_rescue.json").read_text(encoding="utf-8"))["selected_titles"]

        # The backlink page stays in the second scoring as a one-hop page and is not a two-hop extra again.
        self.assertEqual(rescore.call_args.kwargs["backlink_titles"], ["Ion"])
        self.assertEqual(rescore.call_args.kwargs["extra_titles"], ["Heat"])
        self.assertEqual(result["diagnostics"]["reason"], "widen_from_selected")
        self.assertEqual([step["action"] for step in result["diagnostics"]["steps"][-2:]],
                         ["widen_from_backlinks", "widen_from_selected"])
        self.assertEqual(selected, ["On", "Named", "Ion", "Heat"])

    def test_scoring_counts_only_seed_mentions_of_the_answer_itself_for_a_page_with_an_off_topic_intro(self) -> None:
        intros = {
            "Ancient Rome": "Ancient Rome was a civilization in Italy.",
            "Axiomatic": "An axiom is a premise.",
            "Trade (profession)": "A craft or trade is an occupation.",
            "Nature (journal)": "Nature is a journal. Papers appear in Nature every week.",
            "Camille (Monet)": "Camille is a painting. It shows Camille Doncieux.",
            "Pressure": "Pressure is a force. High pressure builds up.",
            "New Orleans": "New Orleans is a city. The port of New Orleans is large.",
        }
        seed_text = (
            "An axiomatic method is axiomatic. Trade grew: trade and more trade, by craft and craft guilds. "
            "The nature of things is the nature of things. Camille Pissarro met Camille Corot. "
            "Pressure and pressure. In New Orleans, bands of New Orleans played New Orleans Blues."
        )
        pages = {"Axiomatic": "Axiom", "Trade (profession)": "Craft"}
        with tempfile.TemporaryDirectory() as tmp:
            result = self.score_rome(tmp, intros, seed_text, pages=pages)
            short = self.score_rome(tmp, intros, seed_text, pages=pages, max_len=4)

        # With answers of at most four letters "Axiom" is none, so the answer and the count are the redirect's.
        self.assertEqual({row["title"]: row["seed_mentions"] for row in short.scores}["Axiomatic"], 2)
        self.assertEqual(
            {row["title"]: row["seed_mentions"] for row in result.scores},
            {
                "Axiomatic": 0,  # a redirect to Axiom, which the seed never names
                "Trade (profession)": 2,  # a redirect to Craft: "craft" is counted, not "trade"
                "Nature (journal)": 0,  # a proper name on its own page; the seed means the plain word
                "Camille (Monet)": 0,  # one part of longer names
                "Pressure": 2,  # a plain word on its own page: any case counts
                "New Orleans": 2,  # not "New Orleans Blues"; "In" only opens the sentence
            },
        )

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

    def test_seed_word_match_takes_the_agent_noun_and_adjective_of_a_greek_seed_in_ia(self) -> None:
        def uses(text, seed, lang="el", derived=False):
            return _uses_seed_word(text, seed, lang=lang, lexicon={}, derived=derived)

        self.assertTrue(uses("Ήταν Γερμανός φιλόσοφος.", "Φιλοσοφία"))
        self.assertTrue(uses("Ο σεισμολόγος μελετά.", "Σεισμολογία"))
        self.assertTrue(uses("Ένας σεισμολογικός σταθμός.", "Σεισμολογία", derived=True))
        # Another ending keeps the old stem: Φυσική needs "φυσικ"; "φυσιο" shares only len - 2 letters.
        self.assertTrue(uses("Η φυσικός ιδιότητα.", "Φυσική"))
        self.assertFalse(uses("Η φυσιολογία.", "Φυσική"))
        self.assertFalse(uses("A thermostat.", "Thermodynamics", lang="en"))

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

        with patch("src.pipeline.scoring.run_candidate_scoring_stage", return_value=mocked_scoring):
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
                "src.wiki_client.WikiClient.fetch_page_extract_with_revid",
                return_value={"title": "Entropy", "extract": "", "revid": None},
            ),
            patch("src.wiki_client.WikiClient.fetch_page_extract", return_value={"extract": ""}),
            patch("src.wiki_client.WikiClient.fetch_lead_wikitext", return_value={"wikitext": "", "revid": None}),
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
            patch("src.pipeline.scoring.run_seed_ingestion_stage", return_value=mocked_seed),
            patch(
                "src.wiki_client.WikiClient.fetch_page_extract",
                return_value={"extract": "Alpha quoted topic reference text"},
            ),
            patch("src.wiki_client.WikiClient.fetch_lead_wikitext", return_value={"wikitext": ""}),
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
            patch("src.pipeline.generate.run_term_extraction_stage", return_value=term_result) as mocked_terms,
            patch("src.pipeline.generate.run_clue_extraction_stage", return_value=SimpleNamespace(clues=[])) as mocked_clues,
        ):
            result = run_rescue_ladder(
                seed_title="Thermodynamics",
                lang="en",
                cache_dir="data/cache/wiki",
                selected_path=selected_path,
                terms_path=tmp_dir / "answer_candidates.csv",
                diagnostics_path=diagnostics_path,
                min_df=3,
                nlp_backend="spacy",
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
        self.assertEqual(first_call["nlp_backend"], "spacy")
        self.assertTrue(first_call["entity_type_scoring"])
        self.assertEqual(first_call["lexicon_weight"], 0.22)
        self.assertEqual(second_call["min_df"], 1)
        self.assertEqual(second_call["nlp_backend"], "spacy")
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
                "src.wiki_client.WikiClient.fetch_page_extract_with_revid",
                return_value={
                    "title": "Entropy",
                    "extract": "Entropy is a measure of disorder in thermodynamics.",
                    "revid": 123,
                },
            ),
            patch("src.pipeline.clues.clue_pass_validate", return_value=True),
        ):
            result = run_clue_extraction_stage(
                seed_title="Thermodynamics",
                lang="en",
                diagnostics_path=diagnostics_path,
                terms_path=terms_path,
                clues_path=clues_path,
            )

        self.assertEqual(result.diagnostics["clue_count"], 1)
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
        seed_title: str = "Jazz",
        lang: str = "en",
    ):
        tmp_dir = Path("tests") / "tmp_outputs" / tmp_name
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        (tmp_dir / "selected_candidates.json").write_text(
            json.dumps({"seed_title": seed_title, "lang": lang, "selected_k": 0, "selected_titles": selected or []}),
            encoding="utf-8",
        )
        (tmp_dir / "terms.csv").write_text(
            "\n".join(["answer,normalized_answer,source_titles,source_method,intro_on_topic,page_is_person", *term_rows]),
            encoding="utf-8",
        )

        def with_revid(title, intro_only=True):
            return {"title": title, "extract": extracts.get(title, ""), "revid": revid(intro_only)}

        with (
            patch("spacy.load", side_effect=OSError("no model in unit test")),
            patch("src.wiki_client.WikiClient.fetch_page_extract_with_revid", side_effect=with_revid),
            patch("src.wiki_client.WikiClient.fetch_page_extract", side_effect=AssertionError("plain text has no revision")),
            patch("src.wiki_client.WikiClient.fetch_lead_wikitext", side_effect=AssertionError("lead revision is another response")),
        ):
            result = run_clue_extraction_stage(
                seed_title=seed_title,
                lang=lang,
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

    def test_clue_stage_defines_from_the_second_sentence_only_when_nothing_else_clues_the_answer(self) -> None:
        follow_up = "She was one of the most influential figures in the development of quantum optics."

        def run(name: str, first: str):
            with patch("src.pipeline.clues.is_generic_definition", side_effect=lambda clue, *args, **kwargs: clue.startswith("A noted")):
                result, _ = self._run_clues_for(name, ["Ana Vale,ANAVALE,Ana Vale,article_title"], {"Ana Vale": f"{first} {follow_up}"})
            return [(row["clue_style"], row["sentence_offset"], row["clue"]) for row in result.clues]

        # The first sentence defines nothing (generic) and is too short for a fill-in: the second one defines.
        self.assertEqual(
            run("clue_follow_up", "Ana Vale was a noted physicist."),
            [("definition", 1, "One of the most influential figures in the development of quantum optics")],
        )
        # A valid fill-in comes before the follow-up.
        self.assertEqual(
            run("clue_follow_up_fill_in", "Ana Vale was a noted physicist and teacher in Kyoto.")[0][:2], ("fill_in_blank", 0)
        )
        # A first sentence that defines wins as before.
        self.assertEqual(
            run("clue_first_sentence", "Ana Vale was a pioneer of quantum optics in Kyoto."),
            [("definition", 0, "A pioneer of quantum optics in Kyoto")],
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

    def test_clue_stage_defines_a_title_the_seed_merely_names_by_its_own_page_before_the_seeds_sentence(self) -> None:
        extracts = {
            "Olive": "An olive is a subspecies of subtropical evergreen tree in the family Oleaceae.",
            "Jazz": "Jazz drummers often sit beside an olive tree outside the clubs.",
        }
        named, _ = self._run_clues_for("clue_named", ["Olive,OLIVE,Olive|Jazz,article_title,False"], extracts)
        on_topic, _ = self._run_clues_for("clue_on_topic", ["Olive,OLIVE,Olive|Jazz,article_title,True"], extracts)

        # The seed has a clean sentence for "olive", and the page's definition still comes first.
        self.assertEqual([(row["clue_style"], row["source_page"]) for row in named.clues], [("definition", "Olive")])
        self.assertEqual([(row["clue_style"], row["source_page"]) for row in on_topic.clues], [("definition", "Olive")])

    def test_clue_stage_takes_fill_ins_only_from_an_off_topic_page_that_sends_to_another_article(self) -> None:
        diary = (
            "Για το σύστημα χρονολόγησης δείτε: Ημερολόγιο (σύστημα χρονολόγησης)\n"
            "Το ημερολόγιο είναι μια καταγραφή με διακριτές εγγραφές ταξινομημένες ανά ημερομηνία."
        )
        seed = "Οι αστρονόμοι χρησιμοποιούν το Ημερολόγιο για να προβλέπουν τις εκλείψεις κάθε χρόνο."

        def run(name: str, on_topic: str, seed_text: str):
            rows = [f"Ημερολόγιο,ΗΜΕΡΟΛΟΓΙΟ,Ημερολόγιο|Αστρονομία,article_title,{on_topic},False"]
            extracts = {"Ημερολόγιο": diary, "Αστρονομία": seed_text}
            result, _ = self._run_clues_for(name, rows, extracts, seed_title="Αστρονομία", lang="el")
            return [(row["clue_style"], row["source_page"]) for row in result.clues]

        # The note marks the diary as another sense of the seed's word, as a title in parentheses would: no
        # definition, the seed's sentence, and no sentence of the diary's own page when the seed has none.
        self.assertEqual(run("clue_note_off_topic", "False", seed), [("fill_in_blank", "Αστρονομία")])
        self.assertEqual(run("clue_note_no_seed_sentence", "False", "Οι αστρονόμοι μετρούν τον ουρανό κάθε νύχτα."), [])
        self.assertEqual(run("clue_note_on_topic", "True", seed), [("definition", "Ημερολόγιο")])

    def test_clue_stage_lets_the_blank_stand_for_the_full_name_of_a_person_answered_by_the_surname(self) -> None:
        extracts = {
            "Wayne Shorter": "Wayne Shorter (1933 \u2013 2023) played the saxophone.",
            "Jazz": (
                "In 1964 the quintet hired Wayne Shorter as its new saxophonist. "
                "Critics praised Alice Shorter for another record that year."
            ),
        }
        rows = ["Shorter,SHORTER,Wayne Shorter|Jazz,article_title,True,True"]
        result, _ = self._run_clues_for("clue_surname", rows, extracts)

        # "Wayne ____" would be one part of a name; "Alice ____" is another person and still is.
        self.assertEqual(
            [(row["clue"], row["source_page"]) for row in result.clues],
            [("In 1964 the quintet hired ____ as its new saxophonist.", "Jazz")],
        )

    def test_a_fill_in_for_a_person_never_blanks_a_common_word(self) -> None:
        text = "A shorter set suited Shorter well. The band hired Shorter as its saxophonist in 1964."

        def fills(person: bool) -> list[str]:
            return [
                row["clue"]
                for row in _collect_source_backed_clue_candidates(
                    answer="Shorter",
                    normalized_answer="SHORTER",
                    source_titles=["Jazz"],
                    source_payloads={"Jazz": {"title": "Jazz", "extract": text, "revid": 1}},
                    seed_title="Jazz",
                    term_source_method="article_title",
                    lang="en",
                    min_words=4,
                    max_words=12,
                    nlp=None,
                    person=person,
                )
            ]

        self.assertEqual(len(fills(person=False)), 2)
        self.assertEqual(fills(person=True), ["The band hired ____ as its saxophonist in 1964."])

    def test_clue_stage_joins_a_generic_definition_of_a_person_with_what_a_later_sentence_says(self) -> None:
        text = "Ana Vale was a noted physicist and teacher in Kyoto. She was born in Osaka. Vale founded the first laboratory."

        def run(name: str, person: str):
            def who(head, sentence, answer, **kwargs):
                return "A noted physicist who founded the first laboratory" if "founded" in sentence else None

            with (
                patch("src.pipeline.clues.is_generic_definition", side_effect=lambda clue, *args, **kwargs: "who" not in clue),
                patch("src.pipeline.clues.who_definition", side_effect=who),
            ):
                result, _ = self._run_clues_for(name, [f"Ana Vale,ANAVALE,Ana Vale,article_title,True,{person}"], {"Ana Vale": text})
            return [(row["clue_style"], row["sentence_offset"], row["clue"]) for row in result.clues]

        # The third sentence is the source, and the definition comes before the valid fill-in of the first one.
        self.assertEqual(
            run("clue_who", "True"), [("definition", 2, "A noted physicist who founded the first laboratory")]
        )
        # A page that is no person has no "who".
        self.assertEqual(run("clue_who_no_person", "False")[0][:2], ("fill_in_blank", 0))

    def test_clue_stage_names_a_person_whose_definition_is_thin_when_nothing_else_clues_them(self) -> None:
        names = ["Ana Vale", "Ben Cole", "Cy Dunn", "Fay Ito", "Gus Moe", "Dan Roe"]
        extracts = {name: f"{name} was an American drummer." for name in names}
        extracts["Gus Moe"] += " Moe played the drums with the quintet in Kyoto every year."  # a fill-in comes first
        extracts["Dan Roe"] = "Dan Roe was a common term."  # generic, but for its kind of word: no name makes it specific
        rows = [f"{name.split()[1]},{name.split()[1].upper()},{name},article_title,True,True" for name in names]
        result, _ = self._run_clues_for("clue_named_person", rows, extracts)

        self.assertEqual(
            {row["answer"]: (row["clue_style"], row["clue"]) for row in result.clues},
            {
                "Vale": ("definition", "An American drummer (Ana ____)"),
                "Cole": ("definition", "An American drummer (Ben ____)"),
                "Dunn": ("definition", "An American drummer (Cy ____)"),
                "Ito": ("definition", "An American drummer (Fay ____)"),
                "Moe": ("fill_in_blank", "____ played the drums with the quintet in Kyoto every year."),
            },
        )

    def test_clue_stage_defines_an_answer_only_by_a_page_that_bears_its_name(self) -> None:
        extracts = {
            "Subatomic particle": "A subatomic particle is a particle smaller than an atom in every model.",
            "Sicily (theme)": (
                "Sicily was a Byzantine province existing from the late 7th to the 10th century. "
                "The fleet of Sicily guarded the grain routes for two long centuries."
            ),
            "Claude Monet": "Oscar-Claude Monet (14 November 1840 \u2013 5 December 1926) was a French painter of water lilies.",
            "Peter Guthrie Tait": "Peter Guthrie Tait (28 April 1831 \u2013 4 July 1901) was a Scottish physicist and pioneer of knots.",
            "Music of Chicago": (
                "Chicago is a major center for music in the Midwest. "
                "Visitors love Chicago for its many late-night clubs."
            ),
            "Jazz": "Physicists probe subatomic structure with large colliders every single year.",
        }
        rows = [
            "Subatomic,SUBATOMIC,Subatomic particle|Jazz,article_title,True,False",  # a redirect's name, not the page's
            "Sicily,SICILY,Sicily (theme),article_title,False,False",  # disambiguated and off topic: maybe another sense
            "Monet,MONET,Claude Monet,article_title,True,True",  # the surname of the person the page is about
            "Peter Tait,PETERTAIT,Peter Guthrie Tait,article_title,True,True",  # another name of that person
            "Chicago,CHICAGO,Music of Chicago|Jazz,article_title,True,False",  # the seed's name for a page that is no person
        ]
        with patch("src.pipeline.clues.is_generic_definition", return_value=False):
            result, _ = self._run_clues_for("clue_own_name", rows, extracts)

        # Chicago is the seed's word, so only a seed sentence may clue it, and this seed has none. Nor has it one
        # for Sicily, whose own page may be about another sense: no clue.
        self.assertEqual(
            {row["answer"]: (row["clue_style"], row["source_page"]) for row in result.clues},
            {
                "Subatomic": ("fill_in_blank", "Jazz"),
                "Monet": ("definition", "Claude Monet"),
                "Peter Tait": ("definition", "Peter Guthrie Tait"),
            },
        )

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

    def test_clue_stage_prefers_what_the_answers_intro_repeats_to_nine_words_and_keeps_that_order_without_a_page(self) -> None:
        extracts = {
            "Tempo": "Tempo is the speed of a musical piece, measured in beats per minute.",
            "Bebop": "Bebop is a style of jazz that arose in the 1940s.",
            "Swing": "Swing is a dance and a style of big band music.",
            "Jazz": "Many fans enjoyed the pace at the old club. A faster pace means more beats per minute in the music.",
        }
        pool = ["Bebop,BEBOP,Bebop,article_title,True", "Swing,SWING,Swing,article_title,True"]
        with_page, _ = self._run_clues_for("clue_about", ["Pace,PACE,Tempo|Jazz,article_title,True", *pool], extracts)
        without_page, _ = self._run_clues_for("clue_no_page", ["Pace,PACE,Jazz,spacy", *pool], extracts)

        def pace_clue(result) -> str:
            return next(row["clue"] for row in result.clues if row["answer"] == "Pace")

        self.assertEqual(pace_clue(with_page), "A faster ____ means more beats per minute in the music.")
        self.assertEqual(pace_clue(without_page), "Many fans enjoyed the ____ at the old club.")

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
            patch("src.pipeline.answers._resolve_nlp_backend", return_value=(None, None, [], None, None)),
            patch("src.wiki_client.WikiClient.fetch_page_extract", side_effect=intro),
            patch("src.wiki_client.WikiClient.fetch_page_extract_with_revid", side_effect=with_revid),
            patch(
                "src.wiki_client.WikiClient.fetch_lead_wikitext",
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

    def test_term_stage_does_not_answer_twice_with_a_selected_redirect_and_the_page_it_leads_to(self) -> None:
        tmp_dir = Path("tests") / "tmp_outputs" / "term_redirect"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        (tmp_dir / "selected_candidates.json").write_text(
            json.dumps({"seed_title": "Vehicle", "lang": "en", "selected_k": 1, "selected_titles": ["Automobile"]}),
            encoding="utf-8",
        )
        (tmp_dir / "candidate_scores.csv").write_text(
            "rank,title,depth,links_back_to_seed,rel_score,red_score,selection_score,intro_on_topic,seed_mentions,status,reason\n"
            "1,Automobile,1,True,0.5,0.0,0.5,True,0,KEEP,x\n",
            encoding="utf-8",
        )
        extract = "A motorcar or automobile is a wheeled vehicle that carries its own engine."

        def intro(title, intro_only=True):  # "Automobile" is a redirect to "Motorcar", the page both names belong to
            if title == "Vehicle":
                return {"title": title, "extract": "A vehicle moves people or goods."}
            return {"title": "Motorcar", "extract": extract}

        def with_revid(title, intro_only=True):
            return {**intro(title), "revid": 5}

        with (
            patch("src.pipeline.answers._resolve_nlp_backend", return_value=(None, None, [], None, None)),
            patch("src.wiki_client.WikiClient.fetch_page_extract", side_effect=intro),
            patch("src.wiki_client.WikiClient.fetch_page_extract_with_revid", side_effect=with_revid),
            patch(
                "src.wiki_client.WikiClient.fetch_lead_wikitext",
                return_value={"wikitext": "A '''motorcar''' or '''automobile''' is a vehicle.", "revid": 5},
            ),
        ):
            result = run_term_extraction_stage(
                seed_title="Vehicle",
                lang="en",
                diagnostics_path=tmp_dir / "diagnostics_terms.json",
                selected_path=tmp_dir / "selected_candidates.json",
                terms_path=tmp_dir / "answer_candidates.csv",
                min_df=1,
                scores_path=tmp_dir / "candidate_scores.csv",
            )

        # The redirect's name is a second spelling of the page's title answer, not a term of its own.
        self.assertEqual({row["normalized_answer"] for row in result.terms}, {"VEHICLE", "MOTORCAR"})
        shutil.rmtree(tmp_dir, ignore_errors=True)

    def title_pool(
        self,
        seed: str,
        rows: list[str],
        intros: dict[str, str],
        cap: int = 10,
        redirects: dict[str, str] | None = None,
        extracts: dict[str, str] | None = None,
        seed_text: str = "",
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
            client.fetch_page_extract_with_revid.return_value = {"title": seed, "extract": seed_text, "revid": 1}
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
        with patch("src.pipeline.clues.is_generic_definition", return_value=False) as generic:
            rows("el")
        self.assertEqual(generic.call_args.kwargs["seed_title"], "Φυσική")

    def test_title_pool_answers_with_the_name_the_seed_uses_for_a_page(self) -> None:
        people = {  # title -> the seed's mentions of the title, as scoring counted them
            "Claude Monet": 1,
            "Jean Monet (son of Claude Monet)": 0,
            "Pierre-Auguste Renoir": 0,  # too long to be an answer under its title
            "Edgar Degas": 1,
            "Mary Lou Williams": 1,
            "Tony Williams (drummer)": 1,
        }
        others = {
            "Monet (crater)": 4,
            "Mus\u00e9e Marmottan Monet": 0,
            "Music of Chicago": 0,
            "Sasanian Empire": 0,
            "Parthian Empire": 0,
            "Ancient Carthage": 0,
            "I Got Rhythm": 0,
            "Kind of Blue": 2,
            "Paris": 2,
            "School of Paris": 0,
        }
        rows = [f"{i},{title},1,True,0.5,0.0,0.5,True,{n},KEEP,x" for i, (title, n) in enumerate((people | others).items(), 1)]
        row = list(people | others).index("Monet (crater)")  # the crater ranks first
        rows[row] = rows[row].replace("0.5,0.0,0.5", "0.9,0.0,0.9")
        intros = {title: f"{title} (born 1900) is a painter." for title in people}
        intros |= {title: f"{title} is on topic." for title in others} | {"Impressionism": "Impressionism was a movement."}
        seed_text = (
            "Claude Monet painted. Monet and Renoir met; Monet admired Renoir. Degas came. "
            "Mary Lou Williams played with Tony Williams; Williams and Williams. "
            "Bands left for Chicago in 1917, and Chicago welcomed them. "
            "Trade crossed the Empire, and the Empire taxed every load. "
            "They razed Carthage. Rome hated Carthage. "
            "Rhythm matters more than melody to every dancer here. Rhythm drives the whole band forward each night. "
            "Kind of Blue sold, and Kind of Blue endures. They met in Paris, and Paris changed them."
        )
        extracts = {"Edgar Degas": "Degas visited Chicago and the Empire often."}  # an on-topic selected article
        pool = self.title_pool("Impressionism", rows, intros, cap=30, seed_text=seed_text, extracts=extracts)

        self.assertEqual(
            {row["answer"]: row["source_titles"] for row in pool},
            {
                "Impressionism": "Impressionism",
                # The seed uses his full name more than his son's, and names him alone twice. The museum is no
                # person and does not compete; the crater does not get to answer with a person's surname.
                "Monet": "Claude Monet|Impressionism",
                "Jean Monet": "Jean Monet (son of Claude Monet)",
                "Renoir": "Pierre-Auguste Renoir|Impressionism",  # an answer only through the surname
                "Degas": "Edgar Degas|Impressionism",  # alone once
                "Mary Lou Williams": "Mary Lou Williams|Impressionism",  # neither Williams is named more than the other
                "Tony Williams": "Tony Williams (drummer)|Impressionism",
                # No person, but a proper name in the seed, twice in mid-sentence: the seed alone is its source.
                "Chicago": "Music of Chicago|Impressionism",
                # "the Empire" is a class noun: two pages end in it, so neither takes it.
                "Parthian Empire": "Parthian Empire",
                "Sasanian Empire": "Sasanian Empire",
                "Ancient Carthage": "Ancient Carthage",  # the seed has no sentence long enough to clue "Carthage"
                "I Got Rhythm": "I Got Rhythm",  # "Rhythm" has a capital only where a sentence opens
                "Kind of Blue": "Kind of Blue|Impressionism",  # "Blue" only inside the title
                "Paris": "Paris|Impressionism",
                "School of Paris": "School of Paris",  # Paris is another page's own title
            },
        )
        # The mentions that count are the answer's: two for "Monet", not the one of "Claude Monet".
        self.assertEqual({row["answer"]: row["answer_score"] for row in pool}["Monet"], 2.2)
        self.assertEqual(
            {row["answer"] for row in pool if row["page_is_person"]},
            {"Monet", "Jean Monet", "Renoir", "Degas", "Mary Lou Williams", "Tony Williams"},
        )
        # A word of the seed title is not free, however often the seed uses it.
        rome = _seed_short_names(
            {"Fall of Rome": "The fall of Rome is on topic."},
            "Ancient Rome",
            "They left Rome early, and Rome burned.",
            lang="en",
            has_seed_fill_in=lambda display, normalized: True,
        )
        self.assertEqual(rome, {})
        unnamed = {row["answer"] for row in self.title_pool("Impressionism", rows, intros, cap=30)}  # a seed that names none
        # Two Monets and no seed to tell them apart keep their titles; Renoir has no rival, Chicago is no person.
        self.assertEqual((("Claude Monet" in unnamed), ("Renoir" in unnamed), ("Chicago" in unnamed)), (True, True, False))

    def test_a_person_page_that_spans_a_life_is_a_person_and_a_war_or_period_is_not(self) -> None:
        for lead in (
            "Claude Monet (14 November 1840 – 5 December 1926) was a painter.",
            "James Watt (; 30 January 1736 (19 January 1736 OS) – 25 August 1819) was an inventor.",
            "Pompey (29 September 106 – 28 September 48 BC) was a general.",
            "Gaius Marius (157–86 BC) was a general.",
            "Augustus (23 September 63 BC – 19 August AD 14) was the first emperor.",
            "Archie Shepp (born May 24, 1937) is a saxophonist.",
            "Postumus (fl. 259–269) was an emperor.",
        ):
            self.assertTrue(_is_person(lead, lang="en"), lead)
        for lead in (
            "The Third Punic War (149–146 BC) was a war.",
            "The Samnite Wars (343–290 BC) were three wars.",
            "The Napoleonic Wars (1803–1815) were a series of wars.",
            "Aegean art (2800–1100 BC) is the art of the Aegean.",
            "Renaissance art (1350 – 1620) is the art of the Renaissance.",
            "Graduate Texts in Mathematics (GTM) (ISSN 0072-5285) is a series.",
        ):
            self.assertFalse(_is_person(lead, lang="en"), lead)
        self.assertTrue(_is_person("Ο Πυθαγόρας (~580 π.Χ. - 496 π.Χ.) ήταν φιλόσοφος.", lang="el"))
        self.assertFalse(_is_person("Ο Μεσαίωνας (476-1492 μ.Χ) ήταν περίοδος.", lang="el"))

    def test_a_greek_lead_that_says_born_in_words_is_a_person(self) -> None:
        for lead in (
            "Ο Γκέοργκ Κάντορ (γερμανικά: Georg Cantor) ήταν μαθηματικός.\nΟ Κάντορ γεννήθηκε το 1845 στην Αγία Πετρούπολη.",
            "Η Ρένα Κρουαζιέ είναι Ελληνίδα ζωγράφος που γεννήθηκε στην Κέρκυρα.",
            "Η Σαντάλ Μουφ (γαλλ. Chantal Mouffe, γενν. 17 Ιουνίου 1943) είναι Βελγίδα φιλόσοφος.",
            "Ο Ζακ Ρανσιέρ (γαλλ. Jacques Rancière), γεννημένος το 1940 στην Αλγερία, είναι Γάλλος φιλόσοφος.",
            "* Το «Μαρξ» ανακατευθύνει εδώ.\nΟ Καρλ Μαρξ (γερμ. Karl Marx, γεν. 5 Μαΐου 1818) ήταν φιλόσοφος.",
            # A copula inside the name does not end it before the life dates.
            "Ο Σαίρεν Κίρκεγκωρ ή, όπως είναι ευρέως γνωστός, Σέρεν Κίρκεγκορ (5 Μαΐου 1813 - 11 Νοεμβρίου 1855) ήταν φιλόσοφος.",
        ):
            self.assertTrue(_is_person(lead, lang="el"), lead)
        for lead in (
            "Η κοσμολογία είναι κλάδος της αστροφυσικής που εξετάζει γιατί γεννήθηκε το σύμπαν.",  # no name before the verb
            "Το Βραβείο Άμπελ είναι βραβείο μαθηματικών. Ο Άμπελ γεννήθηκε το 1802.",  # a prize, not "Ο" or "Η"
            "Ο Απόλλωνας (Ομηρικά: Ἀπόλλων (γεν. Ἀπόλλωνος)) ήταν θεός.",  # "γεν." is the genitive
            # A page named after a person, whose lead gives the person's life dates, opens on no "Ο" or "Η".
            "Οι Μετασχηματισμοί Λόρεντς ονομάστηκαν προς τιμήν του Χέντρικ Λόρεντς (1853 - 1928).",
        ):
            self.assertFalse(_is_person(lead, lang="el"), lead)

    def short_names(self, intros: dict[str, str], seed_text: str = "Heat is energy.", seed_title: str = "Thermodynamics") -> dict:
        return _seed_short_names(
            intros, seed_title, seed_text, lang="en", has_seed_fill_in=lambda display, normalized: True
        )

    def test_a_person_is_answered_by_the_surname_unless_the_title_must_stay_whole(self) -> None:
        short_names = self.short_names
        kelvin = "Lord Kelvin (1824 – 1907) was a physicist. Kelvin proposed the scale. He died in 1907."
        # Nothing has to name the surname alone: a page with no sentence to show it is answered by it all the same.
        self.assertEqual(short_names({"Lord Kelvin": kelvin}), {"Lord Kelvin": ("Kelvin", "KELVIN", 0, True)})
        once = "Lord Kelvin (1824 – 1907) was a physicist. He used the Absolute Kelvin Scale."
        self.assertEqual(short_names({"Lord Kelvin": once}), {"Lord Kelvin": ("Kelvin", "KELVIN", 0, True)})
        # Another page's own title keeps the word, "Kelvin" the unit, unless the seed uses it alone twice.
        self.assertEqual(short_names({"Lord Kelvin": kelvin, "Kelvin (unit)": "The kelvin is a unit."}), {})
        self.assertEqual(
            short_names({"Lord Kelvin": kelvin, "Kelvin (unit)": "The kelvin is a unit."}, "Kelvin said so. Kelvin left."),
            {"Lord Kelvin": ("Kelvin", "KELVIN", 2, True)},
        )
        # A surname after a lower-case word is none, whoever uses it.
        for title in ("Louis de Broglie", "Cato the Younger"):
            last = title.split()[-1]
            intro = f"{title} (1892 – 1987) was a physicist. {last} proposed waves. {last} won."
            self.assertEqual(short_names({title: intro}, f"{last} and {last} agreed."), {}, title)
        # A war that spans a life is no person: it keeps its title.
        wars = "The Samnite Wars (343–290 BC) were three wars. Wars followed. Wars ended."
        self.assertEqual(short_names({"Samnite Wars": wars}), {})
        # A person's surname may have 3 to 15 letters, unless it is a word of the seed title.
        ory = "Kid Ory (1886 – 1973) was a trombonist. Ory led a band."
        heisenberg = "Werner Heisenberg (1901 – 1976) was a physicist. Heisenberg wrote."
        self.assertEqual(
            short_names({"Kid Ory": ory, "Werner Heisenberg": heisenberg}),
            {"Kid Ory": ("Ory", "ORY", 0, True), "Werner Heisenberg": ("Heisenberg", "HEISENBERG", 0, True)},
        )
        self.assertEqual(
            _seed_short_names({"Kid Ory": ory}, "Ory music", "Ory played.", lang="en", has_seed_fill_in=lambda d, n: True), {}
        )
        # Two people share a surname: the one the seed names in full more often than the other takes it.
        williams = {
            "Tony Williams (drummer)": "Tony Williams (1945 – 1997) was a drummer. Williams played.",
            "Mary Lou Williams": "Mary Lou Williams (1910 – 1981) was a pianist. Williams played.",
        }
        self.assertEqual(short_names(williams), {})
        self.assertEqual(
            short_names(williams, "Tony Williams and Tony Williams played; Mary Lou Williams listened."),
            {"Tony Williams (drummer)": ("Williams", "WILLIAMS", 0, True)},
        )

    def test_a_person_the_page_calls_by_another_word_or_writes_in_east_asian_script_keeps_the_title(self) -> None:
        short_names = self.short_names
        # The family name comes first, or the person goes by the given name: the page uses another word.
        ding = "Ding Liren (born 1992) is a chess player. Ding won the title. Ding lost a game."
        da_ponte = "Lorenzo Da Ponte (1749 - 1838) was a librettist. Da Ponte wrote the words. Da Ponte left Vienna."
        self.assertEqual(short_names({"Ding Liren": ding, "Lorenzo Da Ponte": da_ponte}), {})
        # The surname does not have to win every count, only not to lose: the seed uses both words alone once.
        both = "Ding Liren (born 1992) is a chess player. Ding won the title."
        self.assertEqual(short_names({"Ding Liren": both}, "Liren smiled."), {"Ding Liren": ("Liren", "LIREN", 1, True)})
        # A lead in East Asian script may write the family name first; nothing in the text says otherwise.
        ju = "Ju Wenjun (Chinese: \u5c45\u6587\u541b; born 1991) is a chess player."
        self.assertEqual(short_names({"Ju Wenjun": ju}), {})
        # A text that uses the surname alone shows it is what the person is called, in any script.
        nakamura = "Hikaru Nakamura (\u4e2d\u6751 \u5149, born 1987) is a player. Nakamura won."
        self.assertEqual(short_names({"Hikaru Nakamura": nakamura}), {"Hikaru Nakamura": ("Nakamura", "NAKAMURA", 0, True)})

    def test_a_person_whose_birth_name_shares_no_word_with_the_title_keeps_it_unless_the_surname_is_used(self) -> None:
        short_names = self.short_names
        # A stage name: the birth name in the opening parenthesis has nothing of the title.
        man = "Man Ray (born Emmanuel Radnitzky; August 27, 1890 - November 18, 1976) was an artist. He lived in Paris."
        self.assertEqual(short_names({"Man Ray": man}), {})
        # The birth name has the surname ("Kid Ory (born Edward Ory)"), or none is given: "(born May 24, 1937)".
        ory = "Kid Ory (born Edward Ory; December 25, 1886 - January 23, 1973) was a trombonist."
        ellington = "Duke Ellington (born Edward Kennedy Ellington; April 29, 1899 - 1974) was a pianist."
        shepp = "Archie Shepp (born May 24, 1937) is a saxophonist."
        self.assertEqual(
            set(short_names({"Kid Ory": ory, "Duke Ellington": ellington, "Archie Shepp": shepp})),
            {"Kid Ory", "Duke Ellington", "Archie Shepp"},
        )
        # A text that uses the surname alone shows what the person is called.
        self.assertEqual(short_names({"Man Ray": man + " Ray left."}), {"Man Ray": ("Ray", "RAY", 0, True)})

    def test_a_surname_after_a_name_particle_is_none(self) -> None:
        for title in ("Giuseppe De Nittis", "Mario Del Monaco", "Leigh Van Valen", "Louis-Charles Mahé de La Bourdonnais"):
            last = title.split()[-1]
            intro = f"{title} (1846 - 1884) was a painter. {last} worked. {last} left."
            self.assertEqual(self.short_names({title: intro}, f"{last} and {last} agreed."), {}, title)

    def test_a_surname_only_the_page_uses_keeps_the_pages_seed_mentions(self) -> None:
        rows = ["1,Lord Kelvin,1,True,0.5,0.0,0.5,False,5,KEEP,x"]
        intros = {
            "Thermodynamics": "Thermodynamics is a branch of physics.",
            "Lord Kelvin": "Lord Kelvin (1824 – 1907) was a physicist. Kelvin proposed the scale.",
        }
        pool = self.title_pool("Thermodynamics", rows, intros, seed_text="Lord Kelvin wrote. Lord Kelvin proved.")
        # 1.0 + 0.5 relevance + 0.1 for each of the five mentions of the page, not for the none of "Kelvin".
        self.assertEqual({row["answer"]: row["answer_score"] for row in pool}["Kelvin"], 2.0)

    def test_greek_person_pages_answer_by_the_surname_the_seed_writes_but_other_pages_keep_their_title(self) -> None:
        intros = {
            "Γιοχάνες Κέπλερ": "Ο Γιοχάνες Κέπλερ (1571 \u2013 1630) ήταν Γερμανός αστρονόμος.",
            "Ισαάκ Νεύτων": "Ο Ισαάκ Νεύτων (1643 \u2013 1727) ήταν Άγγλος φυσικός.",
            "Τύχο Μπράχε": "Ο Τύχο Μπράχε (1546 \u2013 1601) ήταν Δανός αστρονόμος.",
            "Ίππαρχος ο Ρόδιος": "Ο Ίππαρχος ο Ρόδιος (190 π.Χ. \u2013 120 π.Χ.) ήταν Έλληνας αστρονόμος.",
            "Κλίμακα Κέλβιν": "Η κλίμακα Κέλβιν είναι κλίμακα θερμοκρασίας.",
        }
        seed_text = (
            "Οι νόμοι του Κέπλερ ισχύουν. Ο Γιοχάνες Κέπλερ τους διατύπωσε. Η μηχανική του Κέπλερ επιβίωσε. "
            "Οι νόμοι του Νεύτωνα ισχύουν. Ο Ισαάκ Νεύτων τους διατύπωσε. Η μηχανική του Νεύτωνα επιβίωσε. "
            "Ο Τύχο Μπράχε και ο Μπράχε μέτρησαν. Ο Ίππαρχος ο Ρόδιος και ο Ίππαρχος ο Ρόδιος μέτρησαν τον ουρανό. "
            "Η θερμοκρασία του Κέλβιν και η τιμή του Κέλβιν ανεβαίνουν."
        )
        names = _seed_short_names(
            intros, "Αστρονομία", seed_text, lang="el", has_seed_fill_in=lambda display, normalized: True
        )

        # The seed's two stand-alone mentions of "Κέπλερ" count; the full name does not. The seed writes "του Νεύτωνα"
        # and "Μπράχε" once, but a person is answered by the surname in the title's own form all the same. An epithet
        # is never named alone, and a page that is no person keeps its title.
        self.assertEqual(
            names,
            {
                "Γιοχάνες Κέπλερ": ("Κέπλερ", "ΚΕΠΛΕΡ", 2, True),
                "Ισαάκ Νεύτων": ("Νεύτων", "ΝΕΥΤΩΝ", 0, True),
                "Τύχο Μπράχε": ("Μπράχε", "ΜΠΡΑΧΕ", 1, True),
            },
        )

        # A short Greek page that never names its person alone is answered by the surname too.
        nernst = {"Βάλτερ Νερνστ": "Ο Βάλτερ Νερνστ (1864 - 1941) ήταν Γερμανός χημικός."}
        names = _seed_short_names(nernst, "Θερμοδυναμική", "Η θερμότητα.", lang="el", has_seed_fill_in=lambda d, n: True)
        self.assertEqual(names, {"Βάλτερ Νερνστ": ("Νερνστ", "ΝΕΡΝΣΤ", 0, True)})

    def test_title_pool_admits_only_the_kept_titles_scoring_found_on_topic(self) -> None:
        rows = [
            "1,Thermostat,1,True,0.5,0.0,0.5,False,0,CUT,cut_off_topic",
            "2,Fan,1,True,0.4,0.0,0.4,False,1,CUT,cut_off_topic",
            "3,Entropy,1,False,0.4,0.0,0.4,False,2,KEEP,x",
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
            "1,Alpha,1,True,0.9,0.0,0.9,False,0,KEEP,x",
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

    def test_a_pool_too_thin_for_the_gate_but_enough_for_a_grid_is_accepted_and_laid_out(self) -> None:
        def ladder(tmp_dir: Path, clued: int) -> dict:
            (tmp_dir / "selected.json").write_text(json.dumps({"selected_titles": ["A"]}), encoding="utf-8")
            gate = SimpleNamespace(passed=False, term_count=clued)
            with (
                patch("src.pipeline.generate.run_term_extraction_stage"),
                patch("src.pipeline.generate.run_clue_extraction_stage", return_value=SimpleNamespace(clues=[])),
                patch("src.pipeline.generate.evaluate_vocab_gate", return_value=gate),
            ):
                return run_rescue_ladder(
                    seed_title="Seed",
                    selected_path=tmp_dir / "selected.json",
                    terms_path=tmp_dir / "terms.csv",
                    diagnostics_path=tmp_dir / "diagnostics_rescue.json",
                    candidate_scores_path=tmp_dir / "missing.csv",
                    min_words=12,
                )["diagnostics"]

        stage = SimpleNamespace(diagnostics={"passed": False}, grid_path="grid.json")
        with tempfile.TemporaryDirectory() as tmp:
            thin, too_thin = ladder(Path(tmp), 12), ladder(Path(tmp), 11)
            names = ("candidate_scoring", "k_selection", "term_extraction", "clue_extraction", "vocab_gate", "layout")
            with ExitStack() as stack:
                rescue = stack.enter_context(patch("src.pipeline.generate.run_rescue_ladder", return_value={"diagnostics": thin}))
                started = {
                    name: stack.enter_context(patch(f"src.pipeline.generate.run_{name}_stage", return_value=stage))
                    for name in (*names, "packaging")
                }
                run_generate_pipeline(seed_title="Seed", output_dir=tmp, gate_min=30, min_words=12, max_len=9)
                self.assertEqual(started["candidate_scoring"].call_args.kwargs["max_len"], 9)
                accepted = [started[name].call_args.kwargs for name in ("vocab_gate", "layout", "packaging")]
                run_generate_pipeline(seed_title="Seed", output_dir=tmp, gate_min=30, min_words=12, rescue=False)
                without_rescue = started["layout"].call_args.kwargs["gate_min"]

        self.assertEqual((thin["reason"], thin["steps"][-1]), ("accept_thin_pool", {"action": "accept_thin_pool", "value": 12}))
        self.assertEqual((too_thin["reason"], too_thin["steps"][-1]["action"]), ("insufficient_vocabulary", "shorten_min_len"))
        # After the step the gate, the layout and the package ask for min_words answers; --no-rescue stops at the gate.
        self.assertEqual(rescue.call_args.kwargs["min_words"], 12)
        self.assertEqual([accepted[0]["min_required"], accepted[1]["gate_min"], accepted[2]["gate_min"]], [12, 12, 12])
        self.assertEqual((rescue.call_count, without_rescue), (1, 30))

    def test_rescue_ladder_promotes_the_next_on_topic_candidates_by_rank(self) -> None:
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
                    "1,\"Storyville, New Orleans\",1,True,0.2,0.1,0.1,KEEP,on_topic",
                    "2,\"Associated Press\",1,True,0.3,0.1,0.2,CUT,cut_off_topic",
                    "3,\"Heat\",1,True,0.9,0.0,0.9,KEEP,on_topic",
                    "4,\"Music of Washington, D.C.\",1,True,0.2,0.1,0.1,KEEP,on_topic",
                    "5,\"Entropy\",1,True,0.8,0.0,0.8,KEEP,on_topic",
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
            patch("src.pipeline.generate.run_term_extraction_stage", return_value=SimpleNamespace(terms=[])),
            patch("src.pipeline.generate.evaluate_vocab_gate", return_value=failed_gate),
            patch(
                "src.pipeline.scoring.run_seed_ingestion_stage",
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
        promote_steps = [step for step in steps if step.get("action") == "promote_next_ranked"]
        self.assertTrue(promote_steps)
        self.assertEqual(promote_steps[0]["value"], 3)
        # The off-topic row is skipped, the three KEEP rows are taken in rank order, and the fourth candidate waits.
        self.assertEqual(
            json.loads(selected_override_path.read_text(encoding="utf-8"))["selected_titles"],
            ["InitialTitle", "Storyville, New Orleans", "Heat", "Music of Washington, D.C."],
        )

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
        with patch("src.pipeline.scoring._inventory_min_k_target", return_value=17):
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

    def test_layout_candidates_give_people_at_most_their_share_unless_nothing_else_is_clued(self) -> None:
        # four people outrank three works; the 4-candidate list is a third people at most
        rows = [("Ann Aa", "ANNAA", 0.9, "True"), ("Bob Bb", "BOBBB", 0.8, "True"), ("Cy Cc", "CYCC", 0.7, "True"),
                ("Dee Dd", "DEEDD", 0.6, "True"), ("Fugue", "FUGUE", 0.5, ""), ("Opera", "OPERA", 0.4, ""),
                ("Salon", "SALON", 0.3, "")]
        terms = [
            {"answer": a, "normalized_answer": n, "answer_score": str(score), "page_is_person": person}
            for a, n, score, person in rows
        ]
        clues = [dict(zip(CLUE_HEADER, _clue_row(a, n, revid="7"))) for a, n, _, _ in rows]

        picked = [key for key, _ in _layout_candidates(terms, clues, lang="en", limit=4)]
        self.assertEqual(picked, ["ANNAA", "FUGUE", "OPERA", "SALON"])
        # Without enough other answers the people fill the list as before.
        picked = [key for key, _ in _layout_candidates(terms[:5], clues[:5], lang="en", limit=4)]
        self.assertEqual(picked, ["ANNAA", "BOBBB", "CYCC", "FUGUE"])

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
