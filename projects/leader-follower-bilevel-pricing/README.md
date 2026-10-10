# Finite leader–follower (bilevel) pricing and capacity

An **exact, small-instance** nonzero-sum Stackelberg benchmark for a leader that sets a posted unit price and installed capacity, anticipating how a utility-maximizing follower allocates consumption across demand segments.

This extends the sibling [adversarial optimization](../adversarial-game-theoretic-optimization/) and [market-design](../market-design-mechanism-design-and-incentives/) examples. Those address security/interdiction and allocation mechanisms; here, distinct upper and lower objectives and follower-response tie handling are explicit.

## Mathematical model

Leader decisions: posted unit price `p` in a finite menu `P` and capacity `K` in a finite menu `C`. Follower decisions: integers `q_i` with `0 <= q_i <= d_i` and `sum_i q_i <= K`.

Given `(p, K)`, the follower maximizes:

`sum_i (v_i - p - t_i) q_i`

where `v_i` is per-unit value, `t_i` is transport cost, and `d_i` is segment demand capacity.

The leader maximizes:

`p * sum_i q_i - c_K * K - f * I[K > 0]`

subject to follower optimality. All economic parameters are nonnegative **integers** in this teaching example.

**Multiple follower optima matter.** The **optimistic** version assumes a follower-optimal allocation favorable to leader profit; **pessimistic** assumes the least favorable follower optimum. They are different mathematical models. The code enumerates *all* lower-level optima and then implements the requested tie policy, rather than arbitrarily returning the first follower optimum.

## Run and verify

From this project directory:

`python bilevel_pricing.py`

From the repository root:

`python -m unittest discover -s projects/leader-follower-bilevel-pricing -p "test_*.py" -v`

With the included synthetic fixture:

| Follower response assumption | Optimal posted price | Capacity | Leader profit |
|---|---:|---:|---:|
| Optimistic | 6 | 2 | 10 |
| Pessimistic | 4 | 2 | 6 |

Regression tests independently enumerate all feasible follower allocations, verify follower optimality, assert feasibility and input validation, and verify that the response assumption changes the leader's optimal decision. No external solver or dataset.

## Scope and limitations

- Exponential enumeration of up to `product_i (d_i + 1)` lower-level states *per leader action* is capped at 250,000 states. This is a **small-instance reference oracle**, not an industrial solver.
- The finite integer price/capacity menus are not general continuous bilevel LP/NLP problems, KKT/MPEC complementarity, multi-follower games, or equilibrium algorithms.
- The follower is an aggregated surplus maximizer, not a strategic market containing multiple players.
- Optimistic profit is conditional on favorable follower tie behavior, not guaranteed when follower tie selection is unknown.
- Synthetic assumptions demonstrate model behavior, not real-world profit or calibrated demand.

A subsequent research extension could benchmark an LP follower with a validated KKT reformulation against this type of independent exact oracle.
