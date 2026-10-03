"""Input-oriented CCR (CRS) and BCC (VRS) DEA models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


@dataclass(frozen=True)
class DEAResult:
    dmu: int
    efficiency: float
    lambdas: np.ndarray
    input_slacks: np.ndarray
    output_slacks: np.ndarray
    returns_to_scale: str

    @property
    def efficient(self) -> bool:
        return self.efficiency >= 1.0 - 1e-7


class DataEnvelopmentAnalysis:
    """Input-oriented envelopment DEA with nonnegative inputs and outputs."""

    def __init__(self, inputs: np.ndarray, outputs: np.ndarray):
        x = np.asarray(inputs, dtype=float)
        y = np.asarray(outputs, dtype=float)
        if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
            raise ValueError("inputs and outputs must be 2-D with the same DMU count")
        if x.shape[0] < 2:
            raise ValueError("DEA requires at least two DMUs")
        if np.any(x <= 0) or np.any(y < 0):
            raise ValueError("inputs must be positive and outputs nonnegative")
        if np.any(y.sum(axis=0) <= 0):
            raise ValueError("every output dimension must contain positive production")
        self.inputs = x
        self.outputs = y

    def solve(self, dmu: int, *, returns_to_scale: str = "crs") -> DEAResult:
        """Solve one radial input-oriented DEA model."""
        if not 0 <= dmu < self.inputs.shape[0]:
            raise IndexError("dmu index out of range")
        if returns_to_scale not in {"crs", "vrs"}:
            raise ValueError("returns_to_scale must be 'crs' or 'vrs'")

        n, m = self.inputs.shape
        s = self.outputs.shape[1]
        # z = [theta, lambda_1, ..., lambda_n]
        c = np.zeros(n + 1)
        c[0] = 1.0

        rows = []
        rhs = []

        # Sum_j lambda_j x_ji <= theta x_oi
        for i in range(m):
            row = np.zeros(n + 1)
            row[0] = -self.inputs[dmu, i]
            row[1:] = self.inputs[:, i]
            rows.append(row)
            rhs.append(0.0)

        # Sum_j lambda_j y_jr >= y_or
        for r in range(s):
            row = np.zeros(n + 1)
            row[1:] = -self.outputs[:, r]
            rows.append(row)
            rhs.append(-self.outputs[dmu, r])

        a_eq = None
        b_eq = None
        if returns_to_scale == "vrs":
            a_eq = np.zeros((1, n + 1))
            a_eq[0, 1:] = 1.0
            b_eq = np.array([1.0])

        result = linprog(
            c,
            A_ub=np.asarray(rows),
            b_ub=np.asarray(rhs),
            A_eq=a_eq,
            b_eq=b_eq,
            bounds=[(0.0, 1.0)] + [(0.0, None)] * n,
            method="highs",
        )
        if not result.success:
            raise RuntimeError(f"DEA LP failed: {result.message}")

        theta = float(result.x[0])
        lambdas = np.asarray(result.x[1:])
        projected_inputs = lambdas @ self.inputs
        projected_outputs = lambdas @ self.outputs
        input_slacks = theta * self.inputs[dmu] - projected_inputs
        output_slacks = projected_outputs - self.outputs[dmu]

        return DEAResult(
            dmu=dmu,
            efficiency=theta,
            lambdas=lambdas,
            input_slacks=np.maximum(input_slacks, 0.0),
            output_slacks=np.maximum(output_slacks, 0.0),
            returns_to_scale=returns_to_scale,
        )

    def frontier(self, *, returns_to_scale: str = "crs") -> list[DEAResult]:
        return [
            self.solve(i, returns_to_scale=returns_to_scale)
            for i in range(self.inputs.shape[0])
        ]
