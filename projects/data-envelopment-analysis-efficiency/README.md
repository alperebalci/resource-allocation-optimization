# Data Envelopment Analysis: CCR and BCC Efficiency

This project adds a classical Operations Research / Management Science methodology that is not a prescriptive allocation model.

Given peer decision-making units (DMUs) with multiple inputs and outputs, DEA asks:

> How much could this DMU radially reduce its inputs while still producing at least its current outputs, relative to the observed peer frontier?

Implemented models:

- **CCR / CRS** — constant returns to scale;
- **BCC / VRS** — variable returns to scale through `sum(lambda)=1`;
- peer intensity weights `lambda`;
- radial input efficiency `theta`;
- post-solve input and output slacks.

For target DMU `o`, the input-oriented model is

```text
min theta

sum_j lambda_j x_ij <= theta x_io
sum_j lambda_j y_rj >= y_ro
lambda_j >= 0
```

with `sum_j lambda_j = 1` added for BCC/VRS.

## Validation fixture

The tests include a DMU that is technically efficient under VRS but scale-inefficient under CRS. This explicitly demonstrates why CCR and BCC scores answer different questions rather than merely producing two implementations of the same score.

## Run

```bash
python -m pip install -e '.[dev]'
pytest
```

## Interpretation boundary

DEA measures relative efficiency against the observed comparison set. It does not establish causal productivity, does not automatically correct for environmental heterogeneity or measurement error, and can be sensitive to outliers and variable selection.
