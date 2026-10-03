# Market Design, Mechanism Design and Incentives

A compact Management Science implementation of allocation when decision-makers can be strategic rather than passive inputs to an optimizer.

Implemented mechanisms:

- maximum-welfare unit-demand assignment;
- Vickrey-Clarke-Groves (Clarke pivot) payments for the assignment market;
- proposer-optimal Gale-Shapley deferred acceptance;
- an explicit stability checker for two-sided matching.

This complements centralized resource-allocation MILPs by making incentives and strategic participation part of the model. VCG illustrates the efficiency/incentive side; deferred acceptance illustrates stable matching when preferences, rather than transferable utility, drive allocation.

Run:

```bash
python -m pip install -r requirements.txt
pytest -q
```

Scope is intentionally narrow: no claim is made that VCG is budget balanced, that matching preferences are static, or that these mechanisms directly fit arbitrary non-convex industrial allocation problems.
