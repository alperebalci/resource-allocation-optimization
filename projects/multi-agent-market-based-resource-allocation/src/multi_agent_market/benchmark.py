from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Agent:
    name: str
    base_capacity: int
    worker_productivity: int
    backlog_penalty: float
    worker_cost: float
    max_workers: int

    def period_cost(self, workload: int, workers: int) -> float:
        if workers < 0 or workers > self.max_workers:
            raise ValueError("workers outside feasible range")
        effective_capacity = self.base_capacity + workers * self.worker_productivity
        backlog = max(0, workload - effective_capacity)
        return self.backlog_penalty * backlog * backlog + self.worker_cost * workers

    def marginal_value(self, workload: int, workers_already_assigned: int) -> float:
        if workers_already_assigned >= self.max_workers:
            return float("-inf")
        before = self.period_cost(workload, workers_already_assigned)
        after = self.period_cost(workload, workers_already_assigned + 1)
        return before - after


@dataclass(frozen=True)
class AllocationResult:
    allocation: tuple[int, ...]
    total_cost: float
    unserved_workload: int

    def as_dict(self, agents: Sequence[Agent]) -> dict[str, object]:
        return {
            "allocation": {a.name: x for a, x in zip(agents, self.allocation)},
            "total_cost": self.total_cost,
            "unserved_workload": self.unserved_workload,
        }


def _evaluate(
    agents: Sequence[Agent],
    workloads: Sequence[int],
    allocation: Sequence[int],
) -> AllocationResult:
    if len(agents) != len(workloads) or len(agents) != len(allocation):
        raise ValueError("agents, workloads and allocation must have the same length")
    total_cost = 0.0
    unserved = 0
    for agent, workload, workers in zip(agents, workloads, allocation):
        total_cost += agent.period_cost(workload, workers)
        effective_capacity = agent.base_capacity + workers * agent.worker_productivity
        unserved += max(0, workload - effective_capacity)
    return AllocationResult(tuple(int(x) for x in allocation), float(total_cost), int(unserved))


def centralized_exact(
    agents: Sequence[Agent],
    workloads: Sequence[int],
    shared_workers: int,
) -> AllocationResult:
    """Exact small-instance oracle by enumeration."""
    if shared_workers < 0:
        raise ValueError("shared_workers must be nonnegative")
    ranges: Iterable[range] = [range(a.max_workers + 1) for a in agents]
    best: AllocationResult | None = None
    for allocation in product(*ranges):
        if sum(allocation) > shared_workers:
            continue
        candidate = _evaluate(agents, workloads, allocation)
        if best is None or (candidate.total_cost, candidate.allocation) < (
            best.total_cost,
            best.allocation,
        ):
            best = candidate
    if best is None:
        raise RuntimeError("no feasible allocation found")
    return best


def equal_quota(
    agents: Sequence[Agent],
    workloads: Sequence[int],
    shared_workers: int,
) -> AllocationResult:
    """Static fairness baseline that ignores marginal operational value."""
    allocation = [0] * len(agents)
    remaining = shared_workers
    while remaining > 0:
        progressed = False
        for i, agent in enumerate(agents):
            if allocation[i] < agent.max_workers and remaining > 0:
                allocation[i] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return _evaluate(agents, workloads, allocation)


def marginal_value_auction(
    agents: Sequence[Agent],
    workloads: Sequence[int],
    shared_workers: int,
) -> AllocationResult:
    """
    Decentralized unit auction.

    Each production cell exposes only its current marginal value for one additional
    worker. The coordinator assigns the next worker to the highest positive bid.
    For separable discrete-convex costs with diminishing marginal values, this is
    equivalent to the centralized optimum; the exact oracle is kept to verify that
    property on every benchmark instance.
    """
    allocation = [0] * len(agents)
    for _ in range(shared_workers):
        bids = [
            agent.marginal_value(workload, allocation[i])
            for i, (agent, workload) in enumerate(zip(agents, workloads))
        ]
        best_idx = max(range(len(bids)), key=lambda i: (bids[i], -i))
        if bids[best_idx] <= 0:
            break
        allocation[best_idx] += 1
    return _evaluate(agents, workloads, allocation)


def sample_instance() -> tuple[list[Agent], list[int], int]:
    agents = [
        Agent("machining", 18, 5, 2.8, 2.0, 3),
        Agent("assembly", 15, 4, 3.4, 1.8, 3),
        Agent("inspection", 10, 3, 4.2, 1.5, 3),
    ]
    workloads = [31, 25, 19]
    shared_workers = 5
    return agents, workloads, shared_workers


def run_demo() -> dict[str, dict[str, object]]:
    agents, workloads, shared_workers = sample_instance()
    methods = {
        "centralized_exact": centralized_exact(agents, workloads, shared_workers),
        "equal_quota": equal_quota(agents, workloads, shared_workers),
        "marginal_value_auction": marginal_value_auction(
            agents, workloads, shared_workers
        ),
    }
    return {name: result.as_dict(agents) for name, result in methods.items()}


if __name__ == "__main__":
    from pprint import pprint

    pprint(run_demo())
