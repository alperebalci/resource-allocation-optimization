# Experiment Design

## Objective

Measure the coordination gap between a centralized resource allocator and decentralized production-cell coordination while keeping the physical decision problem fixed.

## Fixture

Three production cells share a pool of flex workers. Workload is observed before allocation. Each cell's backlog penalty is convex and each additional worker provides a diminishing marginal cost reduction.

## Methods

- `centralized_exact`: exhaustive small-instance oracle.
- `equal_quota`: non-optimized fairness baseline.
- `marginal_value_auction`: decentralized marginal-value coordination.

## Primary metrics

- total operating cost;
- total unserved workload;
- feasibility of the shared-resource constraint;
- centralized optimality gap.

For the reference fixture, the auction's optimality gap should be zero. That result is a property of the stated separable/diminishing-return model, not a general claim about auctions.

## Extension protocol

When stochastic workloads are added:

1. freeze train/validation/final random seeds separately;
2. use common random numbers for all compared policies;
3. report mean, p90, and paired cost differences;
4. include failure/constraint-violation rates;
5. keep centralized or MILP/CP-SAT baselines whenever tractable.

When strategic or learned agents are added, report both operational performance and coordination/incentive diagnostics.
