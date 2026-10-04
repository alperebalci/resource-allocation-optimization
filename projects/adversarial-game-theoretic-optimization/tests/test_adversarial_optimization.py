import math
import unittest

import numpy as np

from adversarial_optimization import (
    solve_network_interdiction,
    solve_robust_adversarial_allocation,
    solve_stackelberg_security,
)


class TestAdversarialOptimization(unittest.TestCase):
    def test_stackelberg_balances_residual_loss(self):
        result = solve_stackelberg_security([10.0, 5.0], resource_budget=1.0)
        self.assertAlmostEqual(float(result.coverage.sum()), 1.0, places=7)
        self.assertTrue(np.all(result.coverage >= -1e-9))
        self.assertTrue(np.all(result.coverage <= 1.0 + 1e-9))
        self.assertLessEqual(result.attacker_value, 10.0)

    def test_network_interdiction_increases_shortest_path(self):
        result = solve_network_interdiction(
            nodes=["S", "A", "B", "T"],
            edges=[
                ("S", "A", 1.0),
                ("A", "T", 1.0),
                ("S", "B", 3.0),
                ("B", "T", 3.0),
            ],
            source="S",
            sink="T",
            interdiction_budget=1,
        )
        self.assertEqual(result.shortest_path_length, 6.0)
        self.assertEqual(len(result.interdicted_edges), 1)

    def test_interdiction_can_disconnect_network(self):
        result = solve_network_interdiction(
            nodes=["S", "T"],
            edges=[("S", "T", 1.0)],
            source="S",
            sink="T",
            interdiction_budget=1,
        )
        self.assertTrue(math.isinf(result.shortest_path_length))
        self.assertEqual(result.remaining_path, ())

    def test_robust_allocation_respects_budget(self):
        result = solve_robust_adversarial_allocation(
            baseline_demand={"a": 4, "b": 4},
            scenario_shocks={
                "a_surge": {"a": 3},
                "b_surge": {"b": 3},
            },
            total_units=8,
        )
        self.assertLessEqual(sum(result.allocation.values()), 8)
        self.assertGreaterEqual(result.worst_case_weighted_unmet, 0.0)
        self.assertIn(result.active_worst_case_scenario, {"a_surge", "b_surge"})


if __name__ == "__main__":
    unittest.main()
