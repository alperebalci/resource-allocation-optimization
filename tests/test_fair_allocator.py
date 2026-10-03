import unittest

from src.fair_allocator import FairnessAwareResourceAllocator


class TestFairnessAwareResourceAllocator(unittest.TestCase):
    def test_service_ratio_gap_is_enforced(self):
        model = FairnessAwareResourceAllocator(max_service_ratio_gap=0.10)
        result = model.solve()

        self.assertLessEqual(result["service_ratio_gap"], 0.10 + 1e-9)
        self.assertGreaterEqual(result["total_deployed"], 700)
        self.assertLessEqual(result["total_cost"], model.total_budget)

    def test_tighter_parity_has_an_explicit_cost_tradeoff(self):
        baseline = FairnessAwareResourceAllocator(
            max_service_ratio_gap=None
        ).solve()
        fair = FairnessAwareResourceAllocator(
            max_service_ratio_gap=0.05
        ).solve()

        self.assertLessEqual(fair["service_ratio_gap"], 0.05 + 1e-9)
        self.assertGreaterEqual(fair["total_cost"], baseline["total_cost"])

    def test_zero_gap_equalizes_service_ratios(self):
        result = FairnessAwareResourceAllocator(
            max_service_ratio_gap=0.0
        ).solve()

        self.assertAlmostEqual(result["service_ratio_gap"], 0.0, places=9)

    def test_frontier_reports_policy_scenarios(self):
        model = FairnessAwareResourceAllocator()
        frontier = model.fairness_frontier(gaps=(None, 0.10, 0.0))

        self.assertEqual(len(frontier), 3)
        self.assertIn("total_cost", frontier.columns)
        self.assertIn("realized_service_ratio_gap", frontier.columns)


if __name__ == "__main__":
    unittest.main()
