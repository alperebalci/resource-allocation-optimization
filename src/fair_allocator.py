from __future__ import annotations

from itertools import combinations
from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint

from src.optimizer import MultiSiteResourceAllocator


class FairnessAwareResourceAllocator(MultiSiteResourceAllocator):
    """Extend the base allocation MILP with an explicit service-parity constraint.

    Fairness here means parity of staffing-to-service-need ratios across the
    modeled sites. It is an operational policy metric, not a claim about legal
    or demographic fairness.
    """

    def __init__(
        self,
        total_personnel: int = 1200,
        total_budget: float = 50_000_000,
        operation_months: int = 3,
        minimum_total_deployment: int = 700,
        service_needs: Sequence[float] = (360.0, 260.0, 180.0),
        max_service_ratio_gap: float | None = 0.10,
    ) -> None:
        super().__init__(
            total_personnel=total_personnel,
            total_budget=total_budget,
            operation_months=operation_months,
            minimum_total_deployment=minimum_total_deployment,
        )
        self.service_needs = np.asarray(service_needs, dtype=float)
        self.max_service_ratio_gap = max_service_ratio_gap

        if self.service_needs.shape != (len(self.sites),):
            raise ValueError("service_needs must contain one positive value per site")
        if np.any(self.service_needs <= 0):
            raise ValueError("service_needs must be strictly positive")
        if max_service_ratio_gap is not None and max_service_ratio_gap < 0:
            raise ValueError("max_service_ratio_gap must be non-negative or None")

    def _build_constraints(self) -> LinearConstraint:
        base = super()._build_constraints()
        if self.max_service_ratio_gap is None:
            return base

        rows = [np.asarray(base.A, dtype=float)]
        lower = list(np.asarray(base.lb, dtype=float))
        upper = list(np.asarray(base.ub, dtype=float))

        # For every pair of sites i,j:
        # |x_i / need_i - x_j / need_j| <= allowed_gap
        # This keeps service ratios comparable while preserving the original
        # staffing, budget, deployment and strategic-share constraints.
        for i, j in combinations(range(len(self.sites)), 2):
            row = np.zeros(len(self.sites), dtype=float)
            row[i] = 1.0 / self.service_needs[i]
            row[j] = -1.0 / self.service_needs[j]
            rows.append(row)
            lower.append(-self.max_service_ratio_gap)
            upper.append(self.max_service_ratio_gap)

        return LinearConstraint(
            np.vstack(rows),
            np.asarray(lower, dtype=float),
            np.asarray(upper, dtype=float),
        )

    def solve(self) -> Dict[str, object]:
        solved = super().solve()
        allocation = np.asarray(solved["allocation"], dtype=float)
        ratios = allocation / self.service_needs
        unmet = np.maximum(self.service_needs - allocation, 0.0)

        table = solved["table"].copy()
        table["service_need"] = self.service_needs
        table["service_ratio"] = ratios
        table["unmet_service_need"] = unmet

        solved["table"] = table
        solved["service_ratios"] = ratios
        solved["service_ratio_gap"] = float(ratios.max() - ratios.min())
        return solved

    def fairness_frontier(
        self,
        gaps: Iterable[float | None] = (None, 0.30, 0.20, 0.10, 0.05, 0.0),
    ) -> pd.DataFrame:
        """Solve a transparent cost-vs-parity policy sweep."""

        rows = []
        for gap in gaps:
            model = FairnessAwareResourceAllocator(
                total_personnel=self.total_personnel,
                total_budget=self.total_budget,
                operation_months=self.operation_months,
                minimum_total_deployment=self.minimum_total_deployment,
                service_needs=self.service_needs,
                max_service_ratio_gap=gap,
            )
            result = model.solve()
            rows.append(
                {
                    "allowed_service_ratio_gap": gap,
                    "realized_service_ratio_gap": result["service_ratio_gap"],
                    "total_deployed": result["total_deployed"],
                    "total_cost": result["total_cost"],
                }
            )

        return pd.DataFrame(rows)


def main() -> None:
    model = FairnessAwareResourceAllocator(max_service_ratio_gap=0.10)
    result = model.solve()

    print("=== FAIRNESS-AWARE RESOURCE ALLOCATION ===")
    print(f"Total deployed: {result['total_deployed']}")
    print(f"Total cost: {result['total_cost']:,.2f}")
    print(f"Service-ratio gap: {result['service_ratio_gap']:.4f}")
    print()
    print(result["table"].to_string(index=False))

    print("\n=== COST / SERVICE-PARITY FRONTIER ===")
    print(model.fairness_frontier().to_string(index=False))


if __name__ == "__main__":
    main()
