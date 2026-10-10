"""Independent exact-oracle and edge-case regressions for finite bilevel pricing."""

import itertools
import unittest

from bilevel_pricing import (
    BilevelInstance,
    demo_instance,
    follower_best_responses,
    follower_utility,
    leader_profit,
    solve_bilevel,
)


class BilevelPricingTests(unittest.TestCase):
    def test_follower_optimality_against_independent_exhaustion(self):
        instance = demo_instance()
        for p in instance.price_options:
            for cap in instance.capacity_options:
                r = follower_best_responses(instance, p, cap)
                feasible = [
                    q for q in itertools.product(range(3), range(2))
                    if sum(q) <= cap
                ]
                scores = [follower_utility(instance, p, q) for q in feasible]
                self.assertEqual(r.objective, max(scores))
                self.assertEqual(set(r.allocations), {
                    q for q in feasible
                    if follower_utility(instance, p, q) == max(scores)
                })

    def test_distinct_optimistic_and_pessimistic_decisions(self):
        inst = demo_instance()
        optimistic = solve_bilevel(inst, "optimistic")
        pessimistic = solve_bilevel(inst, "pessimistic")
        self.assertEqual((optimistic.price, optimistic.capacity, optimistic.leader_profit), (6, 2, 10))
        self.assertEqual((pessimistic.price, pessimistic.capacity, pessimistic.leader_profit), (4, 2, 6))
        self.assertGreater(optimistic.leader_profit, pessimistic.leader_profit)

    def test_lower_level_tie_is_not_ignored(self):
        inst = demo_instance()
        follower = follower_best_responses(inst, 6, 2)
        self.assertIn((0, 0), follower.allocations)
        self.assertIn((2, 0), follower.allocations)
        self.assertEqual(follower.objective, 0)
        self.assertEqual(min(leader_profit(inst, 6, 2, q) for q in follower.allocations), -2)
        self.assertEqual(max(leader_profit(inst, 6, 2, q) for q in follower.allocations), 10)

    def test_feasibility(self):
        inst = demo_instance()
        for cap in inst.capacity_options:
            for allocation in follower_best_responses(inst, 1, cap).allocations:
                self.assertLessEqual(sum(allocation), cap)
                self.assertTrue(all(0 <= q <= d for q, d in zip(allocation, inst.demand_caps)))

    def test_zero_capacity(self):
        inst = BilevelInstance((0,), (1,), (0,), (3,), (0,), 2)
        result = solve_bilevel(inst)
        self.assertEqual((result.leader_profit, result.allocation), (0, (0,)))

    def test_validation_and_limit(self):
        with self.assertRaisesRegex(ValueError, "matched"):
            BilevelInstance((1,), (2, 3), (0,), (1,), (1,), 0)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            BilevelInstance((1,), (2,), (0,), (-1,), (1,), 0)
        with self.assertRaisesRegex(ValueError, "safety limit"):
            BilevelInstance((1, 2), (999, 999), (0, 0), (1,), (1,), 0)
        with self.assertRaisesRegex(ValueError, "tie_rule"):
            solve_bilevel(demo_instance(), "none")

    def test_repeatability(self):
        self.assertEqual(solve_bilevel(demo_instance()), solve_bilevel(demo_instance()))


if __name__ == "__main__":
    unittest.main()
