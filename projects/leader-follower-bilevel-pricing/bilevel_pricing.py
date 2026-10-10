"""Exact small-instance, nonzero-sum leader-follower pricing benchmark.

The leader commits to price and capacity; the follower maximizes its own
discrete surplus. All parameters are integers, so every comparison is exact.
This is an exponential reference oracle, not an industrial-size bilevel solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import prod
from typing import Literal

TieRule = Literal["optimistic", "pessimistic"]
MAX_ENUMERATED_STATES = 250_000


@dataclass(frozen=True)
class BilevelInstance:
    valuations: tuple[int, ...]
    demand_caps: tuple[int, ...]
    transport_costs: tuple[int, ...]
    price_options: tuple[int, ...]
    capacity_options: tuple[int, ...]
    capacity_cost: int
    setup_cost: int = 0

    def __post_init__(self) -> None:
        if not self.valuations:
            raise ValueError("At least one demand segment is required")
        if not (len(self.valuations) == len(self.demand_caps) == len(self.transport_costs)):
            raise ValueError("Segments must have matched value, demand, and cost lengths")
        if not self.price_options or not self.capacity_options:
            raise ValueError("Price and capacity menus must be non-empty")
        for values, label in (
            (self.valuations, "valuations"),
            (self.demand_caps, "demand_caps"),
            (self.transport_costs, "transport_costs"),
            (self.price_options, "price_options"),
            (self.capacity_options, "capacity_options"),
            ((self.capacity_cost, self.setup_cost), "capacity/setup costs"),
        ):
            if any(type(x) is not int or x < 0 for x in values):
                raise ValueError(f"{label} must contain only nonnegative integers")
        if prod(d + 1 for d in self.demand_caps) > MAX_ENUMERATED_STATES:
            raise ValueError("Instance exceeds exact-enumeration safety limit")


@dataclass(frozen=True)
class FollowerOptimum:
    objective: int
    allocations: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class LeaderDecision:
    price: int
    capacity: int
    allocation: tuple[int, ...]
    follower_surplus: int
    leader_profit: int
    follower_optimal_response_count: int
    tie_rule: TieRule


def follower_utility(instance: BilevelInstance, price: int, quantities: tuple[int, ...]) -> int:
    return sum(
        (value - price - travel) * q
        for value, travel, q in zip(instance.valuations, instance.transport_costs, quantities)
    )


def leader_profit(instance: BilevelInstance, price: int, capacity: int, quantities: tuple[int, ...]) -> int:
    return price * sum(quantities) - instance.capacity_cost * capacity - instance.setup_cost * (capacity > 0)


def follower_best_responses(instance: BilevelInstance, price: int, capacity: int) -> FollowerOptimum:
    """Enumerate ALL exactly optimal feasible lower-level integer allocations."""
    if type(price) is not int or type(capacity) is not int or price < 0 or capacity < 0:
        raise ValueError("Price and capacity must be nonnegative integers")

    best_value: int | None = None
    optima: list[tuple[int, ...]] = []
    ranges = (range(d + 1) for d in instance.demand_caps)
    for quantities in product(*ranges):
        if sum(quantities) > capacity:
            continue
        value = follower_utility(instance, price, quantities)
        if best_value is None or value > best_value:
            best_value, optima = value, [quantities]
        elif value == best_value:
            optima.append(quantities)
    assert best_value is not None and optima  # zero allocation is always feasible
    return FollowerOptimum(best_value, tuple(optima))


def solve_bilevel(instance: BilevelInstance, tie_rule: TieRule = "pessimistic") -> LeaderDecision:
    """Exact upper-level menu search with optimistic/pessimistic lower-level ties.

    Leader ties use ascending (price, capacity, allocation) for determinism.
    """
    if tie_rule not in ("optimistic", "pessimistic"):
        raise ValueError("tie_rule must be optimistic or pessimistic")

    incumbent: LeaderDecision | None = None
    for price in sorted(set(instance.price_options)):
        for capacity in sorted(set(instance.capacity_options)):
            follower = follower_best_responses(instance, price, capacity)
            choices = [(leader_profit(instance, price, capacity, q), q) for q in follower.allocations]
            target = max(p for p, _ in choices) if tie_rule == "optimistic" else min(p for p, _ in choices)
            chosen = min(q for profit, q in choices if profit == target)
            candidate = LeaderDecision(
                price=price,
                capacity=capacity,
                allocation=chosen,
                follower_surplus=follower.objective,
                leader_profit=target,
                follower_optimal_response_count=len(follower.allocations),
                tie_rule=tie_rule,
            )
            if incumbent is None or candidate.leader_profit > incumbent.leader_profit:
                incumbent = candidate
            elif candidate.leader_profit == incumbent.leader_profit:
                if (candidate.price, candidate.capacity, candidate.allocation) < (
                    incumbent.price, incumbent.capacity, incumbent.allocation
                ):
                    incumbent = candidate
    assert incumbent is not None
    return incumbent


def demo_instance() -> BilevelInstance:
    return BilevelInstance(
        valuations=(6, 2),
        demand_caps=(2, 1),
        transport_costs=(0, 0),
        price_options=(1, 4, 6),
        capacity_options=(0, 1, 2),
        capacity_cost=1,
    )


if __name__ == "__main__":
    instance = demo_instance()
    for rule in ("optimistic", "pessimistic"):
        result = solve_bilevel(instance, rule)
        print(f"{rule}: {result}")
