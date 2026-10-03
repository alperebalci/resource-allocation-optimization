# Multi-Agent and Market-Based Resource Allocation

A compact Industrial Engineering / Operations Research benchmark for comparing **centralized optimization** with **decentralized multi-agent coordination** under a shared resource constraint.

This project fills the gap between classical resource-allocation models and distributed AI/coordination methods. It is built around a falsifiable comparison: decentralized coordination is evaluated against an exact centralized oracle rather than presented as automatically superior.

## Research question

When several production cells compete for a scarce shared workforce pool, how much operational performance is lost by decentralized coordination relative to centralized optimization?

The first benchmark models three autonomous production cells. Each cell has:

- a base processing capacity;
- worker productivity;
- a maximum number of flex workers it can absorb;
- a convex backlog penalty;
- a per-worker operating cost.

A central planner has a fixed number of shared flex workers.

## Compared methods

1. **centralized_exact** — enumerates all feasible allocations and returns the minimum-cost solution. This is the small-instance correctness oracle.
2. **equal_quota** — static fairness baseline that ignores marginal operational value.
3. **marginal_value_auction** — each cell reveals only the current marginal value of one additional worker; the coordinator assigns workers one unit at a time to the highest positive bid.

For the current separable discrete-convex fixture, diminishing marginal values make the unit auction equivalent to the centralized optimum. The exact oracle is retained to verify that property explicitly rather than assuming it.

## Mathematical structure

For cell i, workload d_i, assigned workers x_i, base capacity c_i, worker productivity p_i, backlog penalty q_i, and worker cost w_i:

```text
backlog_i = max(0, d_i - c_i - p_i x_i)

cost_i(x_i) = q_i backlog_i^2 + w_i x_i

minimize    sum_i cost_i(x_i)
subject to  sum_i x_i <= shared_workers
            0 <= x_i <= max_workers_i
            x_i integer
```

The decentralized bid for one additional worker is the cell's local marginal cost reduction.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m multi_agent_market.benchmark
pytest -q
```

The reference fixture produces a centralized optimum and market allocation with the same objective value, while the equal-quota baseline is feasible but more expensive.

## Repository structure

```text
src/multi_agent_market/
  __init__.py
  benchmark.py

tests/
  test_benchmark.py

docs/
  experiment_design.md
```

## What this benchmark does not claim

- It does not claim that auctions are optimal for arbitrary coupled, non-convex, or strategic-agent problems.
- It does not model truthful bidding as a guaranteed behavioral assumption outside the stated fixture.
- It does not treat a market mechanism as a replacement for MILP/CP-SAT in tightly coupled planning problems.
- It does not yet implement MARL. Multi-agent reinforcement learning is a separate extension because it changes both the information structure and the validation problem.

## Next research extensions

- combinatorial auctions with machine-hour and labor bundles;
- capacity coupling and setup externalities that break separability;
- learned bidding policies under nonstationary workload regimes;
- VCG-style payments and incentive diagnostics;
- multi-agent RL compared against the same centralized oracle and auction baselines;
- common-random-number evaluation under stochastic workloads.
