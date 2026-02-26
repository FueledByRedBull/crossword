import unittest

from src.diagnostics import build_seed_stage_diagnostics


class DiagnosticsTests(unittest.TestCase):
    def test_seed_stage_diagnostics_shape(self) -> None:
        diagnostics = build_seed_stage_diagnostics(
            lang="en",
            seed_requested="Thermodynamics",
            seed_resolved="Thermodynamics",
            seed_page_id=12345,
            candidates=[{"title": "Entropy", "depth": 1, "links_back_to_seed": True, "status": "unscored"}],
            cache_stats={"gets": 2, "hits": 1, "misses": 1, "writes": 1},
            include_backlinks=True,
            errors=[],
        )

        self.assertEqual(diagnostics["stage"], "seed_ingestion")
        self.assertEqual(diagnostics["lang"], "en")
        self.assertEqual(diagnostics["seed"]["requested_title"], "Thermodynamics")
        self.assertEqual(diagnostics["counts"]["candidate_count"], 1)
        self.assertEqual(len(diagnostics["candidates"]), 1)
        self.assertEqual(diagnostics["errors"], [])


if __name__ == "__main__":
    unittest.main()
