import numpy as np
from market_design import efficient_assignment, vcg_payments, deferred_acceptance, is_stable_matching


def test_assignment_maximizes_welfare():
    v = np.array([[9, 4, 1], [8, 7, 2], [3, 6, 5]], dtype=float)
    r = efficient_assignment(v)
    assert r.welfare == 21.0
    assert sorted(r.allocation.tolist()) == [0, 1, 2]


def test_vcg_payments_are_nonnegative_and_below_assigned_values():
    v = np.array([[10, 2], [9, 8]], dtype=float)
    r, p = vcg_payments(v)
    assert r.welfare == 18
    assert np.all(p >= 0)
    for i, item in enumerate(r.allocation):
        if item >= 0:
            assert p[i] <= v[i, item] + 1e-9


def test_deferred_acceptance_is_stable():
    proposers = [[0, 1, 2], [0, 2, 1], [1, 0, 2]]
    receivers = [[1, 0, 2], [0, 2, 1], [2, 1, 0]]
    m = deferred_acceptance(proposers, receivers)
    assert is_stable_matching(m, proposers, receivers)
    assert sorted(m.tolist()) == [0, 1, 2]
