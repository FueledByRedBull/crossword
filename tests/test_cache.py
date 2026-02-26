import shutil
import unittest
from pathlib import Path

from src.cache import DiskCache, canonical_params_json, make_cache_key


class CacheTests(unittest.TestCase):
    def test_canonical_params_json_is_order_independent(self) -> None:
        first = {"b": 2, "a": 1, "nested": {"z": 9, "y": 8}}
        second = {"nested": {"y": 8, "z": 9}, "a": 1, "b": 2}
        self.assertEqual(canonical_params_json(first), canonical_params_json(second))

    def test_cache_key_changes_with_endpoint(self) -> None:
        params = {"title": "Thermodynamics"}
        key_one = make_cache_key("endpoint_one", params)
        key_two = make_cache_key("endpoint_two", params)
        self.assertNotEqual(key_one.params_hash, key_two.params_hash)

    def test_disk_cache_round_trip(self) -> None:
        tmp_dir = Path("tests") / "tmp_cache_test"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            cache = DiskCache(tmp_dir)
            endpoint = "w_api_query_links"
            params = {"title": "Thermodynamics", "limit": "max"}
            payload = {"query": {"pages": []}}

            self.assertIsNone(cache.get(endpoint, params))
            cache.set(endpoint, params, payload)
            self.assertEqual(cache.get(endpoint, params), payload)

            stats = cache.stats()
            self.assertEqual(stats["writes"], 1)
            self.assertEqual(stats["hits"], 1)
            self.assertEqual(stats["misses"], 1)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


class VersionPinTests(unittest.TestCase):
    def test_spacy_is_pinned_in_requirements(self) -> None:
        """requirements.txt must contain an exact spaCy pin (==), not a range."""
        req_path = Path("requirements.txt")
        if not req_path.exists():
            self.skipTest("requirements.txt not found")
        content = req_path.read_text(encoding="utf-8")
        spacy_lines = [line for line in content.splitlines() if line.lower().startswith("spacy")]
        self.assertTrue(spacy_lines, "No spacy entry found in requirements.txt")
        for line in spacy_lines:
            self.assertIn("==", line, f"spaCy not pinned with '==': {line}")
            self.assertNotIn(">=", line, f"spaCy has floating lower bound: {line}")
            self.assertNotIn("~=", line, f"spaCy has floating compat bound: {line}")

    def test_spacy_model_url_is_pinned_in_requirements(self) -> None:
        """requirements.txt must contain a versioned spaCy model wheel URL."""
        req_path = Path("requirements.txt")
        if not req_path.exists():
            self.skipTest("requirements.txt not found")
        content = req_path.read_text(encoding="utf-8")
        model_lines = [
            line for line in content.splitlines()
            if "spacy-models" in line or "en_core_web" in line or "el_core_news" in line
        ]
        self.assertTrue(
            model_lines,
            "No spaCy model entry found in requirements.txt; model must be version-pinned",
        )


if __name__ == "__main__":
    unittest.main()
