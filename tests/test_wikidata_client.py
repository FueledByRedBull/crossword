import shutil
import unittest
from pathlib import Path

from src.cache import DiskCache
from src.wikidata_client import WikidataClient


class WikidataClientTests(unittest.TestCase):
    def test_offline_cache_miss_raises(self) -> None:
        cache_dir = Path("tests") / "tmp_cache_wikidata_offline"
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            cache = DiskCache(cache_dir)
            client = WikidataClient(cache, offline=True)
            with self.assertRaisesRegex(RuntimeError, "offline_cache_miss"):
                client.search_entity("entropy")
        finally:
            shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
