from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class AssignmentResult:
    allocation: np.ndarray
    welfare: float


def efficient_assignment(valuations: np.ndarray) -> AssignmentResult:
    v = np.asarray(valuations, dtype=float)
    if v.ndim != 2:
        raise ValueError("valuations must be a 2D agent x item matrix")
    n_agents, n_items = v.shape
    size = max(n_agents, n_items)
    padded = np.zeros((size, size), dtype=float)
    padded[:n_agents, :n_items] = v
    rows, cols = linear_sum_assignment(-padded)
    allocation = np.full(n_agents, -1, dtype=int)
    welfare = 0.0
    for r, c in zip(rows, cols):
        if r < n_agents and c < n_items and v[r, c] > 0:
            allocation[r] = c
            welfare += v[r, c]
    return AssignmentResult(allocation, float(welfare))


def vcg_payments(valuations: np.ndarray) -> tuple[AssignmentResult, np.ndarray]:
    v = np.asarray(valuations, dtype=float)
    result = efficient_assignment(v)
    payments = np.zeros(v.shape[0], dtype=float)
    for i in range(v.shape[0]):
        without_i = np.delete(v, i, axis=0)
        welfare_without = efficient_assignment(without_i).welfare if len(without_i) else 0.0
        own_value = 0.0 if result.allocation[i] < 0 else v[i, result.allocation[i]]
        welfare_others_with = result.welfare - own_value
        payments[i] = max(0.0, welfare_without - welfare_others_with)
    return result, payments


def deferred_acceptance(proposer_preferences: list[list[int]], receiver_rankings: list[list[int]]) -> np.ndarray:
    n_p = len(proposer_preferences)
    n_r = len(receiver_rankings)
    rank = [{p: pos for pos, p in enumerate(pref)} for pref in receiver_rankings]
    next_choice = [0] * n_p
    match_p = np.full(n_p, -1, dtype=int)
    match_r = np.full(n_r, -1, dtype=int)
    free = list(range(n_p))
    while free:
        p = free.pop(0)
        if next_choice[p] >= len(proposer_preferences[p]):
            continue
        r = proposer_preferences[p][next_choice[p]]
        next_choice[p] += 1
        if r < 0 or r >= n_r or p not in rank[r]:
            free.append(p)
            continue
        incumbent = match_r[r]
        if incumbent == -1:
            match_r[r] = p; match_p[p] = r
        elif rank[r][p] < rank[r].get(incumbent, 10**9):
            match_r[r] = p; match_p[p] = r; match_p[incumbent] = -1; free.append(incumbent)
        else:
            free.append(p)
    return match_p


def is_stable_matching(match_p: np.ndarray, proposer_preferences: list[list[int]], receiver_rankings: list[list[int]]) -> bool:
    match_p = np.asarray(match_p, dtype=int)
    n_r = len(receiver_rankings)
    receiver_partner = np.full(n_r, -1, dtype=int)
    for p, r in enumerate(match_p):
        if r >= 0:
            receiver_partner[r] = p
    rank = [{p: pos for pos, p in enumerate(pref)} for pref in receiver_rankings]
    for p, prefs in enumerate(proposer_preferences):
        current = match_p[p]
        for r in prefs:
            if r == current:
                break
            if r < 0 or r >= n_r or p not in rank[r]:
                continue
            incumbent = receiver_partner[r]
            if incumbent == -1 or rank[r][p] < rank[r].get(incumbent, 10**9):
                return False
    return True
