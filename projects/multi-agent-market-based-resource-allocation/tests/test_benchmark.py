from multi_agent_market.benchmark import (
    centralized_exact,
    equal_quota,
    marginal_value_auction,
    sample_instance,
)


def test_auction_matches_exact_on_reference_fixture() -> None:
    agents, workloads, shared_workers = sample_instance()
    exact = centralized_exact(agents, workloads, shared_workers)
    auction = marginal_value_auction(agents, workloads, shared_workers)
    assert auction.total_cost == exact.total_cost
    assert sum(auction.allocation) <= shared_workers


def test_equal_quota_is_feasible() -> None:
    agents, workloads, shared_workers = sample_instance()
    result = equal_quota(agents, workloads, shared_workers)
    assert sum(result.allocation) <= shared_workers
    assert all(x <= agent.max_workers for x, agent in zip(result.allocation, agents))


def test_exact_uses_only_nonnegative_allocations() -> None:
    agents, workloads, shared_workers = sample_instance()
    result = centralized_exact(agents, workloads, shared_workers)
    assert all(x >= 0 for x in result.allocation)
