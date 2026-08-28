import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biomedical_cpt.contamination import contamination_report  # noqa: E402
from biomedical_cpt.metrics import forgetting_delta, relative_domain_gain  # noqa: E402


class MetricTests(unittest.TestCase):
    def test_direction(self):
        self.assertAlmostEqual(relative_domain_gain(10.0, 8.0), 0.2)
        self.assertAlmostEqual(relative_domain_gain(0.5, 0.6, lower_is_better=False), 0.2)
        self.assertAlmostEqual(forgetting_delta(0.7, 0.65), -0.05)

    def test_contamination(self):
        report = contamination_report(["alpha beta gamma delta"], ["alpha beta gamma delta"], n=3)
        self.assertEqual(report["contaminated_examples"], 1)
        self.assertEqual(report["contamination_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()

