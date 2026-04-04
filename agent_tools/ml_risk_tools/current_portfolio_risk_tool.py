
import pandas as pd
import numpy as np
import datetime
from agent_tools.data_tools.fetch_price_data import fetch_price_data
from agent_tools.data_tools.calculate_returns import calculate_returns

def risk_metrics_tool(returns: pd.DataFrame, weights: np.ndarray) -> dict:
    """
    Compute core portfolio risk metrics from asset return series and weights.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset return series (rows = dates, columns = assets).
    weights : np.ndarray
        Portfolio weights aligned with the columns of `returns`.

    Returns
    -------
    dict
        Dictionary of raw portfolio risk metrics:
        - volatility : float
            Annualized standard deviation of portfolio returns.
        - VaR : float
            5th percentile Value at Risk (historical).
        - sharpe : float
            Annualized Sharpe ratio (assumes risk-free rate = 0).
        - max_drawdown : float
            Maximum drawdown of cumulative returns.
        - correlation : float
            Average pairwise correlation between assets.
        - concentration : float
            Herfindahl index (sum of squared weights).

    """

    weights = weights / weights.sum()
    portfolio_returns = returns.dot(weights)

    volatility = np.std(portfolio_returns) * np.sqrt(252)
    VaR = np.percentile(portfolio_returns, 5)
    sharpe = np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252)

    cumulative = (1 + portfolio_returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()

    corr_matrix = returns.corr()
    avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)].mean()

    concentration = np.sum(weights**2)

    return {
        "volatility": volatility,
        "VaR": VaR,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "correlation": avg_corr,
        "concentration": concentration
    }

def risk_scoring_tool(raw_metrics: dict) -> dict:
    """
    Convert raw portfolio risk metrics into a normalized risk score and label.

    Parameters
    ----------
    raw_metrics : dict
        Dictionary of raw risk metrics from `risk_metrics_tool`.

    Returns
    -------
    dict
        Dictionary containing:
        - risk_score : float
            Weighted composite score (0 = low risk, 1 = high risk).
        - risk_level : str
            Categorical risk label ("Low", "Medium", "High").
    """

    def clamp(x, min_val=0, max_val=1):
        return max(min(x, max_val), min_val)

    vol_norm = clamp(raw_metrics["volatility"] / 0.4)
    var_norm = clamp(abs(raw_metrics["VaR"]) / 0.1)
    sharpe_norm = clamp((1 - raw_metrics["sharpe"]) / 2)
    conc_norm = clamp(raw_metrics["concentration"])
    corr_norm = clamp((raw_metrics["correlation"] + 1) / 2)
    dd_norm = clamp(abs(raw_metrics["max_drawdown"]) / 0.5)

    risk_score = (
        0.25 * vol_norm +
        0.20 * var_norm +
        0.20 * sharpe_norm +
        0.15 * conc_norm +
        0.10 * corr_norm +
        0.10 * dd_norm
    )

    if risk_score < 0.35:
        label = "Low"
    elif risk_score <= 0.65:
        label = "Medium"
    else:
        label = "High"

    return {
        "risk_score": risk_score,
        "risk_level": label
    }


def current_portfolio_risk_tool(portfolios: list[dict]):
    """
    Evaluate risk for one or multiple portfolios.

    Parameters
    ----------
    portfolios : list[dict]
        List of portfolio objects. Each must contain:
        - tickers : list[str]
        - weights : list[float]
        - (optional) id, investment_amount, currency

    Returns
    -------
    list[dict]
        Risk results for each portfolio.
    """

    results = []

    for portfolio in portfolios:
        tickers = portfolio["tickers"]
        weights = np.array(portfolio["weights"], dtype=float)

        # Normalize weights (handles % like 50,30,20)
        weights = weights / weights.sum()

        # Convert to dict format if needed elsewhere
        tickers_weights = dict(zip(tickers, weights))

        # Fetch + compute
        prices = fetch_price_data(
            tickers,
            start="2020-01-01",
            end=str(datetime.date.today())
        )

        returns = calculate_returns(prices, method="simple")

        raw_metrics = risk_metrics_tool(returns, weights)
        scoring = risk_scoring_tool(raw_metrics)

        results.append({
            "portfolio_id": portfolio.get("id"),
            "risk_score": scoring,
            "metrics": raw_metrics
        })

    return results