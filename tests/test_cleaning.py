import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biomedical_cpt.cleaning import (  # noqa: E402
    deduplicate_records,
    filter_quality,
    mix_domain_and_replay,
    normalize_text,
    split_records,
)


class CleaningTests(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(normalize_text(" A\n\tB  "), "A B")

    def test_filter_and_deduplicate(self):
        rows, reasons = filter_quality(
            [
                {"id": "a", "text": "This is a sufficiently long biomedical abstract with treatment response and measured outcomes."},
                {"id": "short", "text": "tiny"},
            ],
            min_chars=20,
        )
        self.assertEqual(reasons["too_short"], 1)
        kept, stats = deduplicate_records(rows + [rows[0]])
        self.assertEqual(len(kept), 1)
        self.assertEqual(stats["dropped_exact"], 1)

    def test_mix_and_split_are_deterministic(self):
        domain = [{"id": f"d{i}", "text": f"domain {i}"} for i in range(3)]
        replay = [{"id": f"r{i}", "text": f"replay {i}"} for i in range(2)]
        first = mix_domain_and_replay(domain, replay, domain_ratio=0.6, seed=5, target_records=10)
        second = mix_domain_and_replay(domain, replay, domain_ratio=0.6, seed=5, target_records=10)
        self.assertEqual(first, second)
        self.assertEqual(sum(row["source"] == "domain" for row in first), 6)
        splits = split_records(first)
        self.assertEqual(sum(map(len, splits.values())), 10)


if __name__ == "__main__":
    unittest.main()

