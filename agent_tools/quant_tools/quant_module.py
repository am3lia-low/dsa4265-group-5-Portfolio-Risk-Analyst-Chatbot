"""
Quantitative Module (Portfolio Risk Analyst Chatbot)

Computes risk metrics for a given portfolio.
Called by the agent layer as a tool.

Dependencies:
    pip install numpy pandas scipy yfinance

Usage:
    from quant_module import calculate_all_metrics, metric_benchmarks

    metrics = calculate_all_metrics(
        returns=returns_df,
        prices=price_df,
        weights=[0.60, 0.30, 0.10],
        spy_returns=spy_returns_series,
        cov_matrix=cov_matrix_df
    )
    benchmarks = metric_benchmarks(metrics)
"""

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis


# =============================================================================
# 1. COVARIANCE MATRIX
# Compute once — shared by volatility, risk contribution, avg correlation.
# Do NOT recompute inside each individual function.
# =============================================================================


def calculate_covariance_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the annualised covariance matrix from daily returns.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Daily simple returns. Rows = dates, columns = tickers.

    Returns
    -------
    pd.DataFrame
        Annualised covariance matrix (n_assets x n_assets).
    """
    return returns_df.cov() * 252


# 2. PORTFOLIO VOLATILITY
def calculate_portfolio_volatility(cov_matrix: pd.DataFrame, weights: list) -> float:
    """
    Compute annualised portfolio volatility using the covariance matrix.
    More accurate than simple weighted average — accounts for correlation.

    Parameters
    ----------
    cov_matrix : pd.DataFrame
        Annualised covariance matrix from calculate_covariance_matrix().
    weights : list
        Asset weights. Must sum to 1.0.

    Returns
    -------
    float
        Annualised portfolio volatility (e.g. 0.22 = 22%).
    """
    w = np.array(weights)
    variance = w.T @ cov_matrix.values @ w
    return float(np.sqrt(variance))


# 3. VOLATILITY OF VOLATILITY
def calculate_vol_of_vol(
    returns: pd.DataFrame, weights: list, window: int = 21
) -> float:
    """
    Compute the stability of portfolio volatility over time.
    High vol-of-vol = the risk level is erratic and unpredictable.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily simple returns. Rows = dates, columns = tickers.
    weights : list
        Asset weights. Must sum to 1.0.
    window : int
        Rolling window in days. Default 21 (approx 1 month).

    Returns
    -------
    float
        Std dev of rolling portfolio volatility (annualised).
    """
    port_returns = (returns * np.array(weights)).sum(axis=1)
    rolling_vol = port_returns.rolling(window).std() * np.sqrt(252)
    return float(rolling_vol.std())


# 4. VALUE AT RISK (VaR)
def calculate_var(returns: pd.DataFrame, weights: list, conf: float = 0.95) -> float:
    """
    Compute historical Value at Risk at a given confidence level.
    Uses historical simulation — no normality assumption.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily simple returns. Rows = dates, columns = tickers.
    weights : list
        Asset weights. Must sum to 1.0.
    conf : float
        Confidence level. Default 0.95 (95%).

    Returns
    -------
    float
        VaR as a negative decimal (e.g. -0.05 = 5% loss threshold).
        Interpretation: with (conf)% confidence, daily loss won't exceed this.
    """
    port_returns = (returns * np.array(weights)).sum(axis=1)
    return float(np.percentile(port_returns, (1 - conf) * 100))


# 5. CONDITIONAL VALUE AT RISK (CVaR)
def calculate_cvar(returns: pd.DataFrame, weights: list, conf: float = 0.95) -> float:
    """
    Compute Conditional Value at Risk (Expected Shortfall).
    Average loss on days that exceed the VaR threshold.
    Captures tail severity that VaR alone misses.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily simple returns. Rows = dates, columns = tickers.
    weights : list
        Asset weights. Must sum to 1.0.
    conf : float
        Confidence level. Default 0.95 (95%).

    Returns
    -------
    float
        CVaR as a negative decimal (e.g. -0.072 = 7.2% avg tail loss).
        Always more negative than VaR.
    """
    port_returns = (returns * np.array(weights)).sum(axis=1)
    var = calculate_var(returns, weights, conf)
    tail = port_returns[port_returns <= var]
    return float(tail.mean())


# 6. MAXIMUM DRAWDOWN (MDD)
def calculate_mdd(returns: pd.DataFrame, weights: list) -> float:
    """
    Compute the maximum peak-to-trough decline in portfolio value.
    Most emotionally impactful metric for users — worst-case loss ever.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily closing prices. Rows = dates, columns = tickers.
    weights : list
        Asset weights. Must sum to 1.0.

    Returns
    -------
    float
        Maximum drawdown as a negative decimal (e.g. -0.40 = 40% drop).
    """
    w = np.array(weights)
    port_returns = (returns * w).sum(axis=1)
    wealth = (1 + port_returns).cumprod()
    rolling_max = wealth.cummax()
    drawdown = (wealth - rolling_max) / rolling_max
    return float(drawdown.min())


# 7. SHARPE RATIO
def calculate_sharpe_ratio(
    returns: pd.DataFrame, weights: list, rf: float = 0.04
) -> float:
    """
    Compute the Sharpe ratio — return per unit of total volatility.
    Treats upside and downside volatility equally.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily simple returns. Rows = dates, columns = tickers.
    weights : list
        Asset weights. Must sum to 1.0.
    rf : float
        Annualised risk-free rate. Default 0.04 (4%).

    Returns
    -------
    float
        Sharpe ratio (e.g. 0.32). Higher is better.
        Rule of thumb: <1 poor, 1-2 good, >2 excellent.
    """
    port_returns = (returns * np.array(weights)).sum(axis=1)
    annualised_return = port_returns.mean() * 252
    annualised_vol = port_returns.std() * np.sqrt(252)
    return float((annualised_return - rf) / annualised_vol)


# 8. SORTINO RATIO
def calculate_sortino_ratio(
    returns: pd.DataFrame, weights: list, rf: float = 0.04
) -> float:
    """
    Compute the Sortino ratio — return per unit of downside volatility only.
    More honest than Sharpe for asymmetric portfolios (e.g. Tesla-heavy).

    Parameters
    ----------
    returns : pd.DataFrame
        Daily simple returns. Rows = dates, columns = tickers.
    weights : list
        Asset weights. Must sum to 1.0.
    rf : float
        Annualised risk-free rate. Default 0.04 (4%).

    Returns
    -------
    float
        Sortino ratio. Higher is better.
        If higher than Sharpe, the portfolio has meaningful upside volatility.
    """
    port_returns = (returns * np.array(weights)).sum(axis=1)
    annualised_return = port_returns.mean() * 252
    downside_returns = port_returns[port_returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(252)
    if downside_vol == 0:
        return float("inf")
    return float((annualised_return - rf) / downside_vol)


# 9. SKEWNESS
def calculate_skewness(returns: pd.DataFrame, weights: list) -> float:
    """
    Compute the skewness of portfolio return distribution.
    Detects whether crashes are more likely than rallies.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily simple returns. Rows = dates, columns = tickers.
    weights : list
        Asset weights. Must sum to 1.0.

    Returns
    -------
    float
        Skewness value.
        Negative = left tail is fatter = more crash risk than upside.
        Positive = right tail is fatter = more upside surprise potential.
    """
    port_returns = (returns * np.array(weights)).sum(axis=1)
    return float(skew(port_returns.dropna()))


# 10. EXCESS KURTOSIS
def calculate_excess_kurtosis(returns: pd.DataFrame, weights: list) -> float:
    """
    Compute excess kurtosis of the portfolio return distribution.
    Measures how frequently extreme events occur vs a normal distribution.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily simple returns. Rows = dates, columns = tickers.
    weights : list
        Asset weights. Must sum to 1.0.

    Returns
    -------
    float
        Excess kurtosis (normal distribution = 0).
        High positive value = fat tails = black swan events more likely.
    """
    port_returns = (returns * np.array(weights)).sum(axis=1)
    return float(kurtosis(port_returns.dropna(), fisher=True))


# 11. BETA
def calculate_beta(
    returns: pd.DataFrame, weights: list, spy_returns
) -> float:
    """
    Compute portfolio beta relative to the S&P 500 (SPY).
    Measures how much market moves are amplified or dampened.

    Parameters
    ----------
    returns : pd.DataFrame -> from fetch_price_data
        Daily simple returns. Rows = dates, columns = tickers.
    weights : list -> from st session
        Asset weights. Must sum to 1.0.
    spy_returns: series -> column from returns

    Returns
    -------
    float
        Beta value.
        1.0 = moves with market. >1.0 = amplified. <1.0 = dampened.
    """

    port_returns = (returns * np.array(weights)).sum(axis=1)
    aligned = pd.concat([port_returns, spy_returns], axis=1, join="inner").dropna()
    aligned.columns = ["portfolio", "spy"]
    cov_matrix = np.cov(aligned["portfolio"], aligned["spy"])
    return float(cov_matrix[0, 1] / cov_matrix[1, 1])


# 12. HHI CONCENTRATION
def calculate_hhi(weights: list) -> float:
    """
    Compute the Herfindahl-Hirschman Index (HHI) for portfolio concentration.
    Flags dangerous single-asset over-concentration.

    Parameters
    ----------
    weights : list
        Asset weights. Must sum to 1.0.

    Returns
    -------
    float
        HHI value between 1/n (perfectly diversified) and 1.0 (single asset).
        e.g. [0.6, 0.3, 0.1] → 0.46, perfectly equal 3-asset → 0.33.
    """
    return float(sum(w**2 for w in weights))


# 13. AVERAGE PAIRWISE CORRELATION
def calculate_avg_pairwise_correlation(cov_matrix: pd.DataFrame) -> float:
    """
    Compute mean pairwise correlation across all asset pairs.
    NN-friendly single-number summary of the full correlation matrix.

    Parameters
    ----------
    cov_matrix : pd.DataFrame
        Annualised covariance matrix from calculate_covariance_matrix().

    Returns
    -------
    float
        Average pairwise correlation between -1 and 1.
        High value = assets move together = less diversification benefit.
    """
    std = np.sqrt(np.diag(cov_matrix.values))
    outer = np.outer(std, std)
    corr_matrix = cov_matrix.values / outer
    n = corr_matrix.shape[0]
    if n < 2:
        return 0.0  # single asset — no pairs to correlate
    # Extract upper triangle excluding diagonal
    upper = corr_matrix[np.triu_indices(n, k=1)]
    return float(upper.mean())


# 14. RISK CONTRIBUTION PER ASSET
def calculate_risk_contribution(cov_matrix: pd.DataFrame, weights: list) -> dict:
    """
    Compute each asset's percentage contribution to total portfolio volatility.
    Reveals which asset is actually driving the risk.

    Parameters
    ----------
    cov_matrix : pd.DataFrame
        Annualised covariance matrix from calculate_covariance_matrix().
    weights : list
        Asset weights. Must sum to 1.0.

    Returns
    -------
    dict
        {ticker: contribution_percentage} for each asset.
        e.g. {"AAPL": 0.38, "TSLA": 0.55, "TLT": 0.07}
        Values sum to 1.0. For LLM context only — do NOT feed raw into NN.
    """
    w = np.array(weights)
    port_vol = np.sqrt(w.T @ cov_matrix.values @ w)
    marginal_contrib = cov_matrix.values @ w
    risk_contrib = w * marginal_contrib / port_vol
    pct_contrib = risk_contrib / risk_contrib.sum()
    return {
        ticker: float(round(pct, 4))
        for ticker, pct in zip(cov_matrix.columns, pct_contrib)
    }


# 15. CALCULATE ALL METRICS — wrapper
def calculate_all_metrics(
    returns: pd.DataFrame,
    # prices: pd.DataFrame, # does not seem to be used!
    weights: list,
    cov_matrix: pd.DataFrame,
    rf: float = 0.04,
) -> dict:
    """
    Wrapper that runs all 12 quant functions and returns a single flat dict.
    Called by the agent for FULL_ANALYSIS, SPECIFIC_METRIC, and REBALANCE.
    Covariance matrix is pre-computed and passed in — not recomputed here.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily simple returns. Rows = dates, columns = tickers.
    weights : list
        Asset weights. Must sum to 1.0. Order must match columns in returns/prices.
    cov_matrix : pd.DataFrame
        Pre-computed annualised covariance matrix.
    rf : float
        Annualised risk-free rate. Default 0.04 (4%).

    Returns
    -------
    dict
        Flat dictionary of all 12 metrics. Keys match Wen Xin's NN feature names.
        Risk contribution is nested dict — for LLM context only, not NN input.
    """

    # note:
    # using fetch_price_data in data_tools, spy is added automatically
    # but whether or not it is actually inside the portfolio is another thing

    # checking if SPY in portfolio
    #   (if yes, keep returns as is)
    #   (if no, drop the column)
    spy_returns = returns['SPY']
    if len(weights) == (len(returns.columns)-1): # assuming the missing weight, is infact, bcos of SPY
        returns = returns.drop(columns="SPY")

    metrics = {
        "portfolio_volatility": calculate_portfolio_volatility(cov_matrix, weights),
        "var_95": calculate_var(returns, weights, conf=0.95),
        "sharpe_ratio": calculate_sharpe_ratio(returns, weights, rf),
        "hhi_concentration": calculate_hhi(weights),
        "avg_pairwise_correlation": calculate_avg_pairwise_correlation(cov_matrix),
        "cvar_95": calculate_cvar(returns, weights, conf=0.95),
        "vol_of_vol": calculate_vol_of_vol(returns, weights),
        "max_drawdown": calculate_mdd(returns, weights),
        "sortino_ratio": calculate_sortino_ratio(returns, weights, rf),
        "skewness": calculate_skewness(returns, weights),
        "excess_kurtosis": calculate_excess_kurtosis(returns, weights),
        "beta": calculate_beta(returns, weights, spy_returns),
        # --- Per-asset breakdown — LLM context only, not NN input ---
        "risk_contribution": calculate_risk_contribution(cov_matrix, weights),
    }
    return metrics


# 16. METRIC BENCHMARKS
def metric_benchmarks(metrics: dict) -> dict:
    """
    Map computed metric values to human-readable interpretation tags.
    Helps the LLM generate confident, grounded explanations without
    having to infer what "good" or "bad" means for each metric.

    Parameters
    ----------
    metrics : dict
        Output from calculate_all_metrics().

    Returns
    -------
    dict
        {metric_name: {"value": float, "label": str, "comment": str}}
        label is one of: "low" | "moderate" | "high" | "good" | "poor"
    """

    def _tag(metric, value):

        if metric == "portfolio_volatility":
            if value < 0.10:
                return "low", "Below 10% — relatively stable portfolio"
            elif value < 0.20:
                return "moderate", "10–20% — moderate volatility"
            else:
                return "high", "Above 20% — high volatility, significant price swings"

        elif metric == "var_95":
            if value > -0.02:
                return (
                    "low",
                    "Less than 2% daily loss at 95% confidence — low tail risk",
                )
            elif value > -0.05:
                return (
                    "moderate",
                    "2–5% daily loss at 95% confidence — moderate tail risk",
                )
            else:
                return (
                    "high",
                    "More than 5% daily loss at 95% confidence — high tail risk",
                )

        elif metric == "cvar_95":
            if value > -0.03:
                return "low", "Avg tail loss below 3% — tail risk is contained"
            elif value > -0.07:
                return "moderate", "Avg tail loss 3–7% — moderate tail severity"
            else:
                return "high", "Avg tail loss above 7% — severe tail risk"

        elif metric == "sharpe_ratio":
            if value >= 1.5:
                return "good", "Above 1.5 — excellent risk-adjusted return"
            elif value >= 0.8:
                return "moderate", "0.8–1.5 — acceptable risk-adjusted return"
            else:
                return "poor", "Below 0.8 — poor risk-adjusted return"

        elif metric == "sortino_ratio":
            if value >= 1.5:
                return "good", "Above 1.5 — strong downside-adjusted return"
            elif value >= 0.8:
                return "moderate", "0.8–1.5 — acceptable downside-adjusted return"
            else:
                return "poor", "Below 0.8 — poor downside-adjusted return"

        elif metric == "max_drawdown":
            if value > -0.10:
                return "low", "Less than 10% peak-to-trough — resilient portfolio"
            elif value > -0.25:
                return "moderate", "10–25% peak-to-trough — moderate historical loss"
            else:
                return "high", "More than 25% peak-to-trough — severe historical loss"

        elif metric == "skewness":
            if value > 0.5:
                return (
                    "good",
                    "Positive skew — upside surprises more likely than crashes",
                )
            elif value > -0.5:
                return (
                    "moderate",
                    "Near-zero skew — roughly symmetric return distribution",
                )
            else:
                return "poor", "Negative skew — crashes more likely than rallies"

        elif metric == "excess_kurtosis":
            if value < 1:
                return "low", "Below 1 — tail events close to normal frequency"
            elif value < 3:
                return "moderate", "1–3 — moderately fat tails"
            else:
                return (
                    "high",
                    "Above 3 — very fat tails, extreme events more frequent than expected",
                )

        elif metric == "beta":
            if value < 0.8:
                return "low", "Beta below 0.8 - less sensitive than the market"
            elif value <= 1.2:
                return "moderate", "Beta 0.8 to 1.2 — moves roughly with the market"
            else:
                return (
                    "high",
                    f"Beta above 1.2 — amplifies market moves by {round(value * 100 - 100)}%",
                )

        elif metric == "hhi_concentration":
            if value is None or np.isnan(value) or value == 0:
                return "n/a", "Invalid HHI value — unable to assess concentration"

            if value < 0.20:
                return "low", "Well diversified — no single asset dominates"
            elif value < 0.35:
                return "moderate", "Moderate concentration — some single-asset risk"
            else:
                return (
                    "high",
                    "High concentration — portfolio dominated by one or two assets",
                )

        elif metric == "avg_pairwise_correlation":
            if value < 0.3:
                return "low", "Low correlation — strong diversification between assets"
            elif value < 0.6:
                return "moderate", "Moderate correlation — some diversification benefit"
            else:
                return (
                    "high",
                    "High correlation — assets move together, limited diversification",
                )

        elif metric == "vol_of_vol":
            if value < 0.05:
                return "stable", "Vol-of-vol below 5% — risk level is consistent"
            elif value < 0.10:
                return "moderate", "Vol-of-vol 5–10% — some instability in risk level"
            else:
                return (
                    "unstable",
                    "Vol-of-vol above 10% — risk level is erratic and hard to predict",
                )

        return "n/a", "No benchmark defined for this metric"

    result = {}
    for metric, value in metrics.items():
        if metric == "risk_contribution":
            continue  # skip nested dict
        label, comment = _tag(metric, value)
        result[metric] = {"value": value, "label": label, "comment": comment}
    return result
