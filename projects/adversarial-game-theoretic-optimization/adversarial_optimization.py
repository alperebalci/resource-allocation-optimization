"""Game-theoretic and adversarial optimization reference models.

The module contains three compact, executable prescriptive-analytics models:

1. Stackelberg security allocation with a committed defender and best-responding attacker.
2. Exact small-network interdiction by enumerating interdiction sets.
3. Robust adversarial resource allocation against a finite scenario set.

The implementations favor transparency and testability over large-scale performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import combinations
from math import inf
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp


@dataclass(frozen=True)
class StackelbergResult:
    coverage: np.ndarray
    attacker_target: int
    attacker_value: float
    defender_expected_loss: float


def solve_stackelberg_security(
    losses: Sequence[float],
    resource_budget: float,
) -> StackelbergResult:
    """Solve a zero-sum Stackelberg security allocation LP."""

    loss = np.asarray(losses, dtype=float)
    if loss.ndim != 1 or len(loss) == 0:
        raise ValueError("losses must be a non-empty 1D sequence")
    if np.any(loss < 0):
        raise ValueError("losses must be non-negative")
    if not 0 <= resource_budget <= len(loss):
        raise ValueError("resource_budget must be between 0 and number of targets")

    n = len(loss)
    c = np.zeros(n + 1)
    c[-1] = 1.0

    A_ub = np.zeros((n + 1, n + 1))
    b_ub = np.zeros(n + 1)
    for i, li in enumerate(loss):
        A_ub[i, i] = -li
        A_ub[i, -1] = -1.0
        b_ub[i] = -li

    A_ub[n, :n] = 1.0
    b_ub[n] = resource_budget

    bounds = [(0.0, 1.0)] * n + [(0.0, None)]
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(f"Stackelberg LP failed: {result.message}")

    coverage = np.asarray(result.x[:n], dtype=float)
    residual_losses = loss * (1.0 - coverage)
    attacker_target = int(np.argmax(residual_losses))
    attacker_value = float(residual_losses[attacker_target])

    return StackelbergResult(
        coverage=coverage,
        attacker_target=attacker_target,
        attacker_value=attacker_value,
        defender_expected_loss=attacker_value,
    )


@dataclass(frozen=True)
class InterdictionResult:
    interdicted_edges: tuple[tuple[str, str], ...]
    shortest_path_length: float
    remaining_path: tuple[str, ...]


def _shortest_path(
    nodes: Iterable[str],
    edges: Sequence[tuple[str, str, float]],
    source: str,
    sink: str,
    removed: set[tuple[str, str]],
) -> tuple[float, tuple[str, ...]]:
    adjacency: dict[str, list[tuple[str, float]]] = {node: [] for node in nodes}
    for u, v, w in edges:
        if w < 0:
            raise ValueError("edge weights must be non-negative")
        if (u, v) in removed:
            continue
        adjacency.setdefault(u, []).append((v, float(w)))
        adjacency.setdefault(v, [])

    if source not in adjacency or sink not in adjacency:
        raise ValueError("source and sink must belong to the node set")

    dist = {node: inf for node in adjacency}
    prev: dict[str, str] = {}
    dist[source] = 0.0
    heap: list[tuple[float, str]] = [(0.0, source)]

    while heap:
        d, u = heappop(heap)
        if d != dist[u]:
            continue
        if u == sink:
            break
        for v, w in adjacency[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heappush(heap, (nd, v))

    if dist[sink] == inf:
        return inf, ()

    path = [sink]
    while path[-1] != source:
        path.append(prev[path[-1]])
    path.reverse()
    return dist[sink], tuple(path)


def solve_network_interdiction(
    nodes: Sequence[str],
    edges: Sequence[tuple[str, str, float]],
    source: str,
    sink: str,
    interdiction_budget: int,
    protected_edges: Iterable[tuple[str, str]] = (),
) -> InterdictionResult:
    """Exact interdiction for small directed networks."""

    if interdiction_budget < 0:
        raise ValueError("interdiction_budget must be non-negative")

    node_set = tuple(dict.fromkeys(nodes))
    protected = set(protected_edges)
    candidates = [(u, v) for u, v, _ in edges if (u, v) not in protected]

    best_removed: tuple[tuple[str, str], ...] = ()
    best_length, best_path = _shortest_path(node_set, edges, source, sink, set())

    max_k = min(interdiction_budget, len(candidates))
    for k in range(1, max_k + 1):
        for combo in combinations(candidates, k):
            length, path = _shortest_path(node_set, edges, source, sink, set(combo))
            if length > best_length:
                best_removed = tuple(combo)
                best_length = length
                best_path = path

    return InterdictionResult(
        interdicted_edges=best_removed,
        shortest_path_length=float(best_length),
        remaining_path=best_path,
    )


@dataclass(frozen=True)
class RobustAllocationResult:
    allocation: dict[str, int]
    scenario_unmet: dict[str, float]
    worst_case_weighted_unmet: float
    active_worst_case_scenario: str


def solve_robust_adversarial_allocation(
    baseline_demand: Mapping[str, float],
    scenario_shocks: Mapping[str, Mapping[str, float]],
    total_units: int,
    unmet_weights: Mapping[str, float] | None = None,
) -> RobustAllocationResult:
    """Minimize worst-case weighted unmet demand over finite adversarial scenarios."""

    if total_units < 0:
        raise ValueError("total_units must be non-negative")
    sites = tuple(baseline_demand.keys())
    if not sites:
        raise ValueError("baseline_demand must not be empty")
    if any(v < 0 for v in baseline_demand.values()):
        raise ValueError("baseline demand must be non-negative")
    if not scenario_shocks:
        raise ValueError("scenario_shocks must not be empty")

    weights = {site: 1.0 for site in sites}
    if unmet_weights is not None:
        for site in sites:
            weights[site] = float(unmet_weights.get(site, 1.0))
            if weights[site] < 0:
                raise ValueError("unmet weights must be non-negative")

    scenarios = tuple(scenario_shocks.keys())
    for scenario, shocks in scenario_shocks.items():
        unknown = set(shocks) - set(sites)
        if unknown:
            raise ValueError(f"scenario {scenario!r} has unknown sites: {sorted(unknown)}")

    n = len(sites)
    s_count = len(scenarios)
    u_offset = n
    z_idx = n + s_count * n
    var_count = z_idx + 1

    c = np.zeros(var_count)
    c[z_idx] = 1.0
    c[:n] = 1e-6

    lower = np.zeros(var_count)
    upper = np.full(var_count, np.inf)
    upper[:n] = total_units
    bounds = Bounds(lower, upper)

    integrality = np.zeros(var_count)
    integrality[:n] = 1

    rows = []
    lbs = []
    ubs = []

    row = np.zeros(var_count)
    row[:n] = 1.0
    rows.append(row)
    lbs.append(-np.inf)
    ubs.append(float(total_units))

    scenario_demands: dict[str, dict[str, float]] = {}
    for s_idx, scenario in enumerate(scenarios):
        scenario_demands[scenario] = {}
        shocks = scenario_shocks[scenario]
        for i, site in enumerate(sites):
            demand = float(baseline_demand[site]) + float(shocks.get(site, 0.0))
            if demand < 0:
                raise ValueError("scenario demand cannot be negative")
            scenario_demands[scenario][site] = demand

            row = np.zeros(var_count)
            row[i] = 1.0
            u_idx = u_offset + s_idx * n + i
            row[u_idx] = 1.0
            rows.append(row)
            lbs.append(demand)
            ubs.append(np.inf)

        row = np.zeros(var_count)
        row[z_idx] = 1.0
        for i, site in enumerate(sites):
            u_idx = u_offset + s_idx * n + i
            row[u_idx] = -weights[site]
        rows.append(row)
        lbs.append(0.0)
        ubs.append(np.inf)

    constraints = LinearConstraint(np.vstack(rows), np.asarray(lbs), np.asarray(ubs))
    result = milp(c, integrality=integrality, bounds=bounds, constraints=constraints)
    if not result.success:
        raise RuntimeError(f"Robust allocation MILP failed: {result.message}")

    allocation = {site: int(round(result.x[i])) for i, site in enumerate(sites)}

    scenario_unmet: dict[str, float] = {}
    for scenario in scenarios:
        weighted_unmet = 0.0
        for site in sites:
            unmet = max(scenario_demands[scenario][site] - allocation[site], 0.0)
            weighted_unmet += weights[site] * unmet
        scenario_unmet[scenario] = weighted_unmet

    active = max(scenario_unmet, key=scenario_unmet.get)
    return RobustAllocationResult(
        allocation=allocation,
        scenario_unmet=scenario_unmet,
        worst_case_weighted_unmet=float(scenario_unmet[active]),
        active_worst_case_scenario=active,
    )


def _demo() -> None:
    security = solve_stackelberg_security([100, 70, 50, 30], resource_budget=1.5)
    print("Stackelberg coverage:", np.round(security.coverage, 3))
    print("Attacker best response:", security.attacker_target)
    print("Worst expected loss:", round(security.attacker_value, 3))

    interdiction = solve_network_interdiction(
        nodes=["S", "A", "B", "T"],
        edges=[
            ("S", "A", 1),
            ("A", "T", 1),
            ("S", "B", 2),
            ("B", "T", 2),
            ("A", "B", 1),
        ],
        source="S",
        sink="T",
        interdiction_budget=1,
    )
    print("Interdicted:", interdiction.interdicted_edges)
    print("Remaining path length:", interdiction.shortest_path_length)

    robust = solve_robust_adversarial_allocation(
        baseline_demand={"north": 5, "central": 4, "south": 3},
        scenario_shocks={
            "north_surge": {"north": 4},
            "central_surge": {"central": 4},
            "south_surge": {"south": 4},
        },
        total_units=12,
        unmet_weights={"north": 1.2, "central": 1.0, "south": 0.9},
    )
    print("Robust allocation:", robust.allocation)
    print("Worst scenario:", robust.active_worst_case_scenario)
    print("Worst weighted unmet:", robust.worst_case_weighted_unmet)


if __name__ == "__main__":
    _demo()
