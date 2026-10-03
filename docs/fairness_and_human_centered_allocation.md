# Fairness-Aware and Human-Centered Resource Allocation

This extension adds two modern IE/OR curriculum concepts to the base multi-site allocation model:

1. an explicit **fairness-efficiency trade-off** that can be audited mathematically; and
2. a **human-centered policy interface** in which decision makers choose an interpretable fairness tolerance rather than accepting a hidden scoring rule.

## Operational fairness definition

For this synthetic example, each site has a declared service need. If `x_i` is allocated staff and `d_i` is service need, define

`r_i = x_i / d_i`

as the site's service ratio.

The fairness-aware model bounds the pairwise disparity:

`|r_i - r_j| <= epsilon`

for every pair of sites. The policy parameter `epsilon` is therefore directly interpretable:

- `None`: no service-parity constraint;
- `0.10`: service ratios may differ by at most 10 percentage points;
- `0.00`: exact parity of service ratios.

The objective remains total operating cost. This makes the fairness intervention visible as a constraint rather than burying it inside an opaque weighted score.

## Why this is useful pedagogically

Students can now separate four questions:

- **Efficiency:** what is the least-cost feasible allocation?
- **Equity policy:** how much disparity is acceptable?
- **Trade-off:** what additional cost or deployment is required by a tighter parity rule?
- **Governance:** who sets the rule, how is it justified, and how is it reviewed?

The `fairness_frontier()` method solves a policy sweep and reports realized disparity, total deployment, and cost. This supports Pareto-style discussion without pretending there is one universally correct fairness parameter.

## Human-centered implementation protocol

A real deployment should expose rather than hide the following decisions:

1. **Define service need with domain owners.** The denominator in a service ratio is a policy and measurement choice.
2. **Show the baseline first.** Decision makers should see the unconstrained allocation before fairness restrictions are added.
3. **Present a frontier, not one magic answer.** Show how cost and service parity change as `epsilon` changes.
4. **Allow documented overrides.** Operational exceptions should be visible and auditable.
5. **Record assumptions and provenance.** Service-need data, budget values, bounds, and policy parameters should be versioned.
6. **Monitor realized outcomes.** A fair allocation model does not guarantee fair real-world outcomes if demand, service quality, or measurement processes differ.

## Important boundary

This project uses **site-level service parity** only. It does not use or infer race, sex, disability, age, religion, or other protected characteristics, and it should not be interpreted as a legal-compliance model. In real high-stakes settings, fairness definitions must be developed with relevant domain, legal, and affected-stakeholder input.

## Run

```bash
python -m src.fair_allocator
```

## Test

```bash
python -m unittest discover -s tests
```

The tests verify the parity bound, the cost trade-off under tighter parity, exact parity at zero tolerance for the synthetic fixture, and frontier reporting.
