# Quantitative Module

This module computes portfolio risk and performance metrics for use in an agent-based financial analysis system. It supports full analysis, rebalancing, and specific metric queries.

## Functionality

**Risk Metrics**
* Portfolio volatility
* Value at Risk (VaR, 95%)
* Conditional Value at Risk (CVaR, 95%)
* Maximum drawdown
* Volatility of volatility

**Performance Metrics**
* Sharpe ratio
* Sortino ratio

**Distribution Metrics**
* Skewness
* Excess kurtosis

**Market and Diversification**
* Beta (vs SPY)
* HHI concentration
* Average pairwise correlation
* Risk contribution per asset (LLM context only)

**Interpretation Layer**
* `metric_benchmarks()` maps raw metrics to labels and comments for LLM explanations

## Usage

```python
from quant_module import calculate_all_metrics, metric_benchmarks

metrics = calculate_all_metrics(
    returns=returns_df,
    prices=price_df,
    weights=[0.6, 0.3, 0.1],
    spy_returns=spy_returns,
    cov_matrix=cov_matrix
)

benchmarks = metric_benchmarks(metrics)
```

## Inputs

* `returns`: DataFrame of daily returns
* `prices`: DataFrame of prices
* `weights`: list of asset weights (must sum to 1)
* `spy_returns`: Series of benchmark returns
* `cov_matrix`: precomputed covariance matrix

All inputs must be aligned in terms of assets and dates.

## Outputs

* `calculate_all_metrics()`: dictionary of computed metrics
* `metric_benchmarks()`: dictionary of value, label, and comment per metric

## Notes

* Covariance matrix should be computed once and reused
* Benchmark thresholds are heuristic and intended for interpretability
* Some metrics require sufficient data history
* Missing data should be handled before calling this module
