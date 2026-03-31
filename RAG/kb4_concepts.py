"""
kb4_concepts.py
===============
Static curated concept definitions — Intent 4 (Concept Explanation).
 
Output
------
  knowledge_base/concepts/concepts.json   — persisted concept dict
  kb4_concepts.txt                        — human-readable audit
 
Folder
------
  knowledge_base/concepts/
"""

from __future__ import annotations

import os
import re
import json
import datetime

CONCEPTS_DIR  = "knowledge_base/concepts"
CONCEPTS_FILE = os.path.join(CONCEPTS_DIR, "concepts.json")
OUTPUT_TXT = "output/kb4_concepts.txt"

os.makedirs(CONCEPTS_DIR, exist_ok=True)

# =============================================================================
# DATA CLEANING
# =============================================================================
 
def _clean_text(text: str) -> str:
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return re.sub(r"\s+", " ", text).strip()
 
 
def _convert_numbers(text: str) -> str:
    """
    Concept-specific number normalisation.
 
    Rules:
      • $X.XXB/M/K and plain $NNN → expanded prose
      • "X%"        → "X percent"
      • "1/N"       → kept (already clear prose in concept text)
    """
    def _dollar_scale(m):
        s = m.group().replace(",", "")[1:]
        if s.endswith("B"):   return f"{float(s[:-1])} billion dollars"
        if s.endswith("M"):   return f"{float(s[:-1])} million dollars"
        if s.endswith("K"):   return f"{float(s[:-1])} thousand dollars"
        try:
            v = float(s)
            if v >= 1_000_000: return f"{v/1_000_000:.2f} million dollars"
            if v >= 1_000:     return f"{v/1_000:.2f} thousand dollars"
            return f"{v:.2f} dollars"
        except: return m.group()
 
    text = re.sub(r"\$\d[\d,]*\.?\d*[BMK]?", _dollar_scale, text)
 
    def _pct(m):
        s = m.group().replace(",", "").rstrip("%")
        try:    return f"{float(s):.4g} percent"
        except: return m.group()
 
    text = re.sub(r"-?\d[\d,]*\.?\d*%", _pct, text)
    return re.sub(r"\s+", " ", text).strip()
 
 
def _clean_chunk(text: str) -> str:
    return _convert_numbers(_clean_text(text))

# =============================================================================
# CONCEPT DEFINITIONS
# =============================================================================

CONCEPTS: dict[str, dict] = {
 
    # ── Core Risk Metrics ─────────────────────────────────────────────────────
 
    "volatility": {
        "label": "Volatility (Standard Deviation)",
        "definition": "Volatility measures how much an asset's returns fluctuate over time, expressed as the annualised standard deviation of daily returns.",
        "detail": "A stock with 30% annualised volatility is expected to move roughly ±30% from its current price in one year with 68% probability (one standard deviation). Volatility is symmetric — it penalises both upside and downside moves equally, which is why the Sortino ratio was developed as an improvement for risk assessment.",
        "worked_example": "Daily returns with standard deviation of 1% → annualised vol = 1% × √252 ≈ 15.87%. A $100,000 portfolio can expect to fluctuate roughly ±$15,870 in a typical year.",
        "risk_relevance": "Risk classification: below 10% is low, 10-18% is moderate, 18-28% is high and above 28% is very high. Portfolios with volatility above 20% are generally considered high-risk for most retail investors.",
        "related": ["portfolio_volatility", "volatility_21d", "sharpe_ratio", "sortino_ratio", "var", "beta"],
    },
 
    "portfolio_volatility": {
        "label": "Portfolio Volatility",
        "definition": "Portfolio volatility is the annualised standard deviation of the portfolio's daily weighted returns, accounting for the correlations between all holdings.",
        "detail": "It is not simply the weighted average of individual asset volatilities. Two highly correlated assets (correlation near 1.0) contribute nearly additive volatility. Two uncorrelated or negatively correlated assets reduce total portfolio volatility below the weighted average — this is the mathematical basis of diversification. The formula of portfolio volatility is the square root of the transpose weight vector w multiplied by the covariance matrix and multiplied by w.",
        "worked_example": "Asset A has a volatility of 30%, a weight of 50%. Asset B has a volatility of 20%, a weight of 50%. Weighted average volatility is 25%. If correlation is equal to 0, then the portfolio vol is equal to the square root of (the square of 0.5, multiplied by the square of 0.3, adding the square of 0.5 multiplied by the square of 0.2, which is approximately 18%. If correlation is equal to 1.0, then the portfolio volatility will be equal to 25%, which has no diversification benefit.",
        "risk_relevance": "Portfolio volatility is the primary number for overall risk classification. It should be reviewed alongside individual-asset volatilities and the correlation matrix to understand what is driving it.",
        "related": ["volatility", "covariance_matrix", "correlation", "diversification"],
    },
 
    "volatility_21d": {
        "label": "21-Day Rolling Volatility",
        "definition": "The annualised volatility computed from only the most recent 21 trading days (approximately one calendar month), giving a short-term view of current market conditions.",
        "detail": "21-day vol reacts quickly to regime changes and recent market shocks, whereas 252-day (annual) vol is slower to reflect new information. When 21-day vol is significantly above annual vol, the portfolio has recently entered a higher-risk environment. When it is below, conditions have calmed relative to the historical norm.",
        "worked_example": "Annual vol is 20%. Recent 21-day vol is 35%. The portfolio is experiencing elevated short-term risk, likely due to a recent market event. The 21-day number is more relevant for near-term decisions like position sizing or stop-loss levels.",
        "risk_relevance": "Use 21-day vol for tactical decisions (near-term risk sizing) and annual vol for strategic decisions (long-run allocation). A large gap between the two signals a regime transition.",
        "related": ["volatility", "portfolio_volatility", "var", "max_drawdown"],
    },
 
    "var": {
        "label": "Value at Risk — Loss Threshold at 95% Confidence (VaR)",
        "definition": "VaR is the maximum expected daily loss at a given confidence level under normal market conditions. A 95% daily VaR of 2% means that; on 95 of every 100 trading days, the loss will not exceed 2%.",
        "detail":        "The remaining 5% of days (roughly 12 per year) can produce losses exceeding the VaR threshold. VaR does NOT say how large those tail losses can be — that is CVaR (Conditional VaR). Calculated here as the negative 5th percentile of the historical daily return distribution (historical simulation method).",
        "worked_example": "Portfolio value has an amount of $500,000. Daily VaR 95% is 1.5%. Dollar VaR is equal to $7,500. On approximately 12 trading days per year, losses are expected to exceed $7,500.",
        "risk_relevance": "VaR is a regulatory standard (Basel III) and practical risk budget tool. Use alongside max drawdown for a complete picture — VaR captures typical tail days, MDD captures the worst sustained loss path.",
        "related": ["max_drawdown", "volatility", "skewness", "excess_kurtosis"],
    },

    "cvar": {
    "label": "Conditional Value at Risk — Expected Tail Loss (CVaR / Expected Shortfall)",
    "definition": "CVaR is the average loss on the worst-case days beyond the VaR threshold. At 95% confidence, it measures the expected loss on the worst 5% of trading days.",
    "detail": "While VaR gives a cutoff point, CVaR goes further by quantifying how severe losses are in the tail. Calculated here as the average of returns below the 5th percentile of the historical daily return distribution (historical simulation method). CVaR is always greater than or equal to VaR in magnitude.",
    "worked_example": "Portfolio value has an amount of $500,000. Daily VaR 95% is 1.5% ($7,500). CVaR 95%: 2.3% ($11,500). This means that on the worst 5% of days, the average loss is about $11,500 — significantly larger than the VaR threshold.",
    "risk_relevance": "CVaR is a more conservative and informative tail risk measure than VaR, capturing the severity of extreme losses. Widely used in risk management and increasingly preferred in regulatory frameworks (Basel III/IV). Especially important for portfolios exposed to fat tails or market shocks.",
    "related": ["var", "max_drawdown", "volatility", "excess_kurtosis"],
    },
 
    "max_drawdown": {
        "label": "Maximum Drawdown (MDD)",
        "definition": "Maximum drawdown is the largest peak-to-trough decline in portfolio value over a given period, expressed as a percentage of the peak value.",
        "detail": "MDD = (Trough Value − Peak Value) / Peak Value. It measures the worst-case loss an investor would have experienced if they bought at the peak and sold at the trough. A 50% drawdown requires a 100% subsequent gain just to return to breakeven — the asymmetry makes MDD one of the most psychologically important risk metrics.",
        "worked_example": "Portfolio grows from $100,000 to $150,000 (peak), then falls to $90,000 (trough). MDD is equal to ($90,000 − $150,000) divide by $150,000, which is equal to negative 40%. Note that this is a 40% loss from peak, not a 10% loss from the original $100,000 starting value.",
        "risk_relevance": "MDD above 20% signals historically painful drawdowns. High-concentration tech portfolios have experienced 40-70% MDD in bear markets. Diversification and rebalancing are the primary tools to reduce MDD.",
        "related": ["var", "volatility_21d", "sharpe_ratio", "rebalancing"],
    },
 
    "sharpe_ratio": {
        "label": "Sharpe Ratio",
        "definition": "The Sharpe ratio measures risk-adjusted return: excess return earned per unit of total volatility (risk) taken.",
        "detail": "Formula: (Portfolio Return − Risk-Free Rate) / Portfolio Volatility. Values above 1.0 are generally considered good; above 2.0 is excellent. The risk-free rate is typically the current T-bill rate or Fed Funds rate. Sharpe is symmetric — it penalises upside volatility equally to downside volatility.",
        "worked_example": "The return is 12%, risk-free rate is 5% and the volatility is 18%. Sharpe will be equal to (12% − 5%) divided by 18% = 0.39. This is below 1.0 — the portfolio is not well-compensated for the risk it carries.",
        "risk_relevance": "A falling Sharpe over time signals deteriorating risk-adjusted performance. Use Sharpe to compare portfolios with similar return goals — a portfolio with lower return but higher Sharpe is taking more efficient risk.",
        "related": ["sortino_ratio", "volatility", "max_drawdown", "portfolio_volatility"],
    },
 
    "sortino_ratio": {
        "label": "Sortino Ratio",
        "definition": "The Sortino ratio modifies the Sharpe ratio to penalise only downside volatility (losses), not upside volatility (gains).",
        "detail": "Formula: (Portfolio Return − Risk-Free Rate) divided by the Downside Deviation. Downside deviation is the standard deviation of only negative-return days. Sortino is more appropriate than Sharpe for portfolios with positively skewed returns (options, trend-following) where upside volatility should not be penalised.",
        "worked_example": "The return is 14%, the risk-free rate is 5%, and the downside deviation is 6%. Sortino will be equal to (14% − 5%) divided by 6%, which is equal to 1.5. Healthy ratio — the portfolio earns 1.5 units of excess return per unit of downside risk.",
        "risk_relevance": "Use Sortino when assets have asymmetric return distributions. If Sharpe is low but Sortino is high, the portfolio's volatility is mostly upside — a less concerning situation than if both are low.",
        "related": ["sharpe_ratio", "skewness", "volatility", "max_drawdown"],
    },
 
    "skewness": {
        "label": "Skewness",
        "definition": "Skewness measures the asymmetry of the return distribution. Positive skew means occasional large gains; negative skew means occasional large losses.",
        "detail": "A normal distribution has skewness of 0. Positive skewness, which has a value above 0.5, means that the distribution has a long right tail — returns occasionally spike far above average. Negative skewness, which has a value below 0.5, means that the distribution has a long left tail — returns occasionally plunge far below average. Most equity portfolios exhibit slight negative skew due to crash risk.",
        "worked_example": "Portfolio skewness of −0.8, which is a negative skew, indicates that crash risk is present. The portfolio occasionally experiences large negative daily returns that are far worse than the average bad day. This elevates VaR and makes the Sharpe ratio less representative of true risk.",
        "risk_relevance": "Negative skew combined with high excess kurtosis (fat tails) indicates significant tail risk not captured by volatility or Sharpe alone. Always review skewness alongside VaR for a complete risk picture.",
        "related":       ["excess_kurtosis", "var", "sortino_ratio"],
    },
 
    "excess_kurtosis": {
        "label":         "Excess Kurtosis",
        "definition":    "Excess kurtosis measures how much fatter the tails of the return distribution are compared to a normal distribution. Positive excess kurtosis means extreme returns (very large gains or losses) occur more often than normal.",
        "detail":        "A normal distribution has excess kurtosis of 0. Leptokurtic (excess kurtosis above 1): fat tails — extreme market events occur more frequently than models assuming normality predict. This is the 'fat tails' problem that caused failures in quantitative models during the 2008 crisis. Platykurtic (below −1): thin tails — extreme events are rarer than normal.",
        "worked_example": "Portfolio excess kurtosis of 3.2, shows that it's significantly leptokurtic. A model assuming normal returns would underestimate the probability of a 3-standard-deviation daily loss by several times. VaR calculated with a normality assumption would significantly understate true risk.",
        "risk_relevance": "High excess kurtosis means VaR and volatility-based risk measures underestimate true tail risk. Historical simulation VaR (used in this system) captures this better than parametric VaR, but it requires sufficient history to have sampled the tails.",
        "related":       ["skewness", "var", "volatility"],
    },
 
    "beta": {
        "label": "Beta (Market Sensitivity)",
        "definition": "Beta measures how much a portfolio or asset moves relative to the market (S&P 500). Beta of 1.0 means it moves in line with the market.",
        "detail": "Beta above 1 means that amplifies market moves. Beta below 1: dampens market moves (defensive). Beta below 0 means that moves inversely to the market. Formula: covariance(asset returns, market returns) / variance(market returns). Beta is estimated from historical data and is not stable — it shifts with market regimes.",
        "worked_example": "Portfolio beta 1.3, shows that if SPY drops 10%, the portfolio is expected to drop approximately 13%. If SPY gains 15%, the portfolio gains approximately 19.5%. High beta amplifies both gains and losses.",
        "risk_relevance": "High-beta portfolios, which have a beta value above 1.2, are appropriate in bull markets but dangerous in corrections. Risk-sensitive investors should target beta near 0.8-1.0. Adding bonds, gold, or low-beta defensive stocks reduces portfolio beta.",
        "related": ["volatility", "correlation", "rebalancing", "sharpe_ratio"],
    },
 
    "covariance_matrix": {
        "label": "Covariance Matrix",
        "definition": "The covariance matrix captures the pairwise covariances between all assets in a portfolio — how much each pair of assets moves together in absolute terms.",
        "detail": "Each off-diagonal element Σ_ij = covariance(asset i, asset j) = correlation_ij × vol_i × vol_j. The diagonal elements are the variances of individual assets. Portfolio variance = wᵀ Σ w (the quadratic form of weights and the covariance matrix). The covariance matrix is the mathematical foundation of Modern Portfolio Theory — it determines both portfolio volatility and the efficient frontier.",
        "worked_example": "I have two assets; AAPL (vol having 35% and weight having 50%) and TLT (vol having 12% and weight having 50%), correlation is equal to −0.15. Covariance(AAPL, TLT) is equal to −0.15 x 0.35 × 0.12 = −0.0063. Portfolio variance is equal to 0.5² × 0.35² + 0.5² × 0.12² + 2 × 0.5 × 0.5 × (−0.0063) is then equal to 0.035. Portfolio vol is equal to √0.035 = 18.7%, well below the weighted average of 23.5%.",
        "risk_relevance": "The covariance matrix is what allows diversification to reduce risk. Assets with negative covariance offset each other's volatility. In market crises, covariances spike toward positive values — previously uncorrelated assets move together — which is why diversification can fail precisely when it is most needed.",
        "related":       ["correlation", "pairwise_correlations", "portfolio_volatility", "diversification"],
    },
 
    "pairwise_correlations": {
        "label":         "Pairwise Correlations",
        "definition":    "Pairwise correlations are the correlation coefficients between every pair of assets in the portfolio, ranging from −1 (perfectly opposite) to +1 (perfectly in sync).",
        "detail":        "Correlation of 0 means the assets are independent — holding both gives maximum diversification benefit. Correlation above 0.70 means the assets behave almost identically — holding both offers limited risk reduction. The correlation matrix is the normalised version of the covariance matrix: correlation_ij = Σ_ij / (vol_i × vol_j). Critically, correlations are not stable — they tend to spike toward 1.0 during market crises.",
        "worked_example": "AAPL and MSFT have a correlation of 0.85, showing near-identical behaviour. Holding 50% each is almost the same as holding 100% in either — concentration risk remains. AAPL and TLT correlation −0.15: slightly negative — TLT provides mild diversification benefit against equity drawdowns.",
        "risk_relevance": "Review pairwise correlations to identify clusters of highly correlated assets. A portfolio with all pairs above 0.6 has essentially one concentrated bet regardless of how many tickers it holds. The correlation breakdown in crises is why static allocation is insufficient — dynamic monitoring is needed.",
        "related":       ["covariance_matrix", "portfolio_volatility", "concentration_risk", "diversification"],
    },
 
    "risk_contribution": {
        "label":         "Risk Contribution (% of Portfolio Volatility)",
        "definition":    "Risk contribution measures what percentage of total portfolio volatility each asset is responsible for, accounting for both its weight and its correlation with other holdings.",
        "detail":        "A position with a small weight can have a disproportionately large risk contribution if it is highly volatile or highly correlated with other large positions. Formula: RC_i = w_i × (Σw)_i / σ_p, where (Σw)_i is the i-th element of the product of the covariance matrix and the weight vector, and σ_p is portfolio volatility. Risk contributions sum to 100% of total portfolio variance.",
        "worked_example": "My Portfolio has a share of 50% of NVDA, 30% of MSFT and 20% of TLT. NVDA risk contribution might be 75% despite being only 50% by weight — because NVDA is highly volatile and correlated with MSFT. TLT risk contribution might be negative (it reduces portfolio risk) or near zero due to low correlation with equities.",
        "risk_relevance": "Risk contribution is more informative than weight for understanding true portfolio concentration. Two portfolios with identical weights but different correlations can have completely different risk contribution profiles. Use risk contribution to identify the single largest driver of portfolio volatility and consider trimming it.",
        "related":       ["covariance_matrix", "pairwise_correlations", "concentration_risk", "portfolio_volatility"],
    },
 
    "herfindahl_index": {
        "label":         "Herfindahl-Hirschman Index (HHI)",
        "definition":    "The HHI measures portfolio concentration as the sum of squared weights. It ranges from 1/N (perfectly equal-weight, maximum diversification) to 1.0 (single position, maximum concentration).",
        "detail":        "HHI = Σ w_i². For a 5-asset equal-weight portfolio: HHI = 5 × (0.2²) = 0.20. For a single position: HHI = 1.0. A useful normalised version: (HHI − 1/N) / (1 − 1/N) maps the range to [0, 1] regardless of portfolio size. Originally used in antitrust economics to measure market concentration; adapted for portfolio analysis.",
        "worked_example": "My portfolio has a share of AAPL of 50%, MSFT of 30% and NVDA of 20%. HHI = 0.5² + 0.3² + 0.2² is equal to 0.25 + 0.09 + 0.04 is equal to 0.38. Equal-weight 3-stock benchmark HHI is then equal to 0.33. The concentrated portfolio has HHI 0.38 vs benchmark 0.33 — moderately more concentrated than equal-weight.",
        "risk_relevance": "HHI above 0.25 in a 10-asset portfolio signals meaningful concentration. HHI above 0.5 in any portfolio signals very high single-name risk. Rebalancing is the remedy — trim the largest weights to bring HHI down.",
        "related":       ["concentration_risk", "risk_contribution", "pairwise_correlations", "rebalancing"],
    },
 
    "concentration_risk": {
        "label":         "Concentration Risk",
        "definition":    "Concentration risk is the excess risk arising from holding too much in one asset, sector, or correlated group — creating dependence on a single source of return or loss.",
        "detail":        "Concentration risk has three dimensions: (1) single-name concentration (one stock too large), (2) sector concentration (too much in technology, for example), (3) factor concentration (all holdings are growth-style or all are rate-sensitive). All three can cause correlated losses even when the portfolio appears diversified by ticker count.",
        "worked_example": "A 20-stock portfolio where 18 stocks are US technology: single-name HHI may be reasonable, but sector concentration is 90%. In a tech selloff (like 2022), all 18 positions fall simultaneously — the apparent diversification across tickers provides no protection.",
        "risk_relevance": "Monitor HHI, risk contribution, pairwise correlations, and sector weights together. No single metric captures all forms of concentration. The remedy is always the same: diversify across low-correlated asset classes, not just across tickers within the same sector.",
        "related":       ["herfindahl_index", "risk_contribution", "pairwise_correlations", "diversification"],
    },
 
    # ── Portfolio Strategy Concepts ───────────────────────────────────────────
 
    "diversification": {
        "label":         "Diversification",
        "definition":    "Diversification is the practice of spreading investments across assets with low correlations to reduce portfolio volatility without proportionally reducing expected return.",
        "detail":        "True diversification requires low or negative pairwise correlations — it is not just about holding many tickers. A 20-stock portfolio of US technology stocks is not well-diversified. Effective diversification spans: asset class (equities, bonds, commodities), geography, sector, and factor (growth vs value, cyclical vs defensive).",
        "worked_example": "2022: SPY −18%, short-term bonds (SHY) −3.5%, commodities (DJP) +20%. A 60% equity / 30% bond / 10% commodity portfolio would have lost approximately −8% — far less than pure equity.",
        "risk_relevance": "Diversification is the only free lunch in finance (Markowitz). It reduces volatility without proportionally reducing return. It is the primary defence against concentration risk and single-event losses.",
        "related":       ["correlation", "pairwise_correlations", "concentration_risk", "rebalancing", "asset_allocation"],
    },
 
    "rebalancing": {
        "label":         "Portfolio Rebalancing",
        "definition":    "Rebalancing is the process of realigning portfolio weights back to target allocations after market movements have caused drift.",
        "detail":        "Two main approaches: (1) Calendar rebalancing — at fixed intervals (monthly, quarterly). (2) Threshold rebalancing — when any position drifts beyond a set band (e.g. ±5pp). Research shows threshold rebalancing outperforms calendar on a risk-adjusted basis. Rebalancing sells winners and buys laggards — disciplined buy-low-sell-high that many investors struggle with emotionally.",
        "worked_example": "Target: 60% equity / 40% bond. After rally, drifts to 72%/28%. Threshold rule triggers: sell 12pp equity, buy 12pp bond. On a $500,000 portfolio: sell $60,000 equity, buy $60,000 bond.",
        "risk_relevance": "Unmanaged drift can materially increase risk. A 60/40 drifting to 80/20 has substantially higher equity risk than intended. Rebalancing also prevents emotional bias — investors naturally want to hold more of what has risen.",
        "related":       ["concentration_risk", "diversification", "asset_allocation", "beta"],
    },
 
    "asset_allocation": {
        "label":         "Asset Allocation",
        "definition":    "Asset allocation is the strategic distribution of a portfolio across asset classes (equities, bonds, commodities, cash) to match an investor's risk tolerance, time horizon, and return objectives.",
        "detail":        "Research (Brinson, Hood & Beebower 1986) shows asset allocation explains approximately 90% of long-run portfolio return variation — far more than individual stock selection. Common frameworks: 60/40 (equities/bonds), 80/20 (growth), Risk Parity (equal risk contribution), All-Weather (performs across all macro regimes).",
        "worked_example": "Risk Parity: equities vol 20%, bonds vol 6% → bonds get roughly 3.3× more weight than equities so both contribute equally to portfolio risk. Result: approximately 25% equities, 75% bonds (often levered).",
        "risk_relevance": "The right asset allocation is the single most consequential portfolio construction decision. An allocation too aggressive for an investor's time horizon forces selling at market lows to meet cash needs — permanently realising losses.",
        "related":       ["diversification", "rebalancing", "beta", "correlation"],
    },
 
    "high_risk_definition": {
        "label":         "What 'High Risk' Means in a Portfolio Context",
        "definition":    "A portfolio is high risk when multiple risk dimensions are simultaneously elevated: high volatility, high beta, high concentration, high drawdown potential, and poor diversification.",
        "detail":        "Risk is multidimensional — a single metric is never sufficient. A portfolio can have moderate volatility but extremely high concentration (one stock is 60% of the portfolio). High risk in practice means: annualised vol above 18%, beta above 1.3, max drawdown potential above 25%, HHI above 0.25, and fewer than 5 distinct sector exposures. All five dimensions should be assessed together.",
        "worked_example": "Portfolio A: 100% NVDA. Vol 55%, beta 1.8, MDD (2022) −66%, HHI 1.0, sectors 1 — extremely high risk. Portfolio B: 40% SPY, 30% TLT, 20% GLD, 10% cash. Vol ~10%, beta ~0.6, MDD ~−15% — moderate to low risk.",
        "risk_relevance": "Understanding the components of high risk enables targeted remediation: too-high vol → add bonds or defensive equities; too-high beta → add uncorrelated assets; too-high concentration → rebalance into broader index exposure.",
        "related":       ["volatility", "beta", "concentration_risk", "max_drawdown", "var", "diversification"],
    },
 
    # ── Market Concepts ───────────────────────────────────────────────────────
 
    "yield_curve": {
        "label":         "Yield Curve and Inversion",
        "definition":    "The yield curve plots Treasury interest rates across maturities. An inverted curve (short rates above long rates) is a historically reliable recession predictor.",
        "detail":        "The 10Y-2Y spread is the most watched indicator. Inversion has preceded every US recession since 1955, with a lag of 6-18 months. The mechanism: when short rates exceed long rates, banks' borrowing costs exceed lending income — credit tightens. However, equities have historically continued rising for 12+ months after initial inversion.",
        "worked_example": "2022-2023: 10Y-2Y spread inverted to −1.07%, the deepest since the 1980s. Equities sold off materially despite the economy technically staying out of recession into 2023.",
        "risk_relevance": "Inverted yield curve signals: reduce equity risk, particularly in rate-sensitive sectors (real estate, utilities, banks). High-quality short-duration bonds become attractive as they yield more than long-term bonds without duration risk.",
        "related":       ["interest_rate_risk", "recession_risk", "asset_allocation"],
    },
 
    "interest_rate_risk": {
        "label":         "Interest Rate Risk (Duration)",
        "definition":    "Interest rate risk is the risk that rising rates reduce the market value of existing fixed-income holdings. Duration measures this sensitivity.",
        "detail":        "Duration (years) approximates how much a bond's price falls for a 1% rise in rates. A bond with duration 7 will lose approximately 7% in price if rates rise by 1%. Equities face indirect rate risk: high-multiple growth stocks (with value in distant future cash flows) are particularly sensitive — rising rates reduce the present value of future earnings.",
        "worked_example": "TLT (20+ year Treasury ETF) has duration approximately 17 years. When rates rose by roughly 2% in 2022, TLT fell approximately 34% — from interest rate risk alone, not credit risk.",
        "risk_relevance": "In a rising-rate environment, long-duration bonds and growth stocks both suffer — the traditional 60/40 fails to diversify. Reducing duration (move to shorter-term bonds) and tilting toward value stocks mitigates this.",
        "related":       ["yield_curve", "asset_allocation", "diversification"],
    },
 
    "recession_risk": {
        "label":         "Recession Risk",
        "definition":    "Recession risk is the probability the economy enters a period of declining GDP for two or more consecutive quarters, typically causing broad equity market declines.",
        "detail":        "Key leading indicators: inverted yield curve, rising unemployment claims, falling PMI, declining consumer confidence, and tightening credit. Equities typically peak 6-12 months before a recession begins and trough 6-12 months before it ends. Defensive sectors (Staples, Healthcare, Utilities) historically outperform; cyclicals (Energy, Financials, Industrials) underperform.",
        "worked_example": "2008 recession: S&P 500 fell −57% peak to trough. Consumer Staples ETF (XLP) fell only −29%. A portfolio with 40% defensive sector allocation would have significantly outperformed market-cap-weighted allocation.",
        "risk_relevance": "Recession risk warrants rotating toward defensive sectors, increasing cash or short-duration bonds, and reducing beta below 1.0. It is NOT a signal to exit markets entirely — timing recessions precisely is nearly impossible.",
        "related":       ["yield_curve", "beta", "asset_allocation"],
    },
 
    "liquidity_risk": {
        "label":         "Liquidity Risk",
        "definition":    "Liquidity risk is the risk of not being able to buy or sell an asset quickly at a fair price without significantly moving the market.",
        "detail":        "Applies to: small-cap and micro-cap stocks (wide bid-ask spreads), illiquid ETFs (low trading volume), real estate, and private equity. Liquidity risk becomes critical during market stress — when you most need to sell, liquidity evaporates and prices gap down. As a rule of thumb, keep individual positions below 5% of a stock's average daily volume.",
        "worked_example": "Stock with average daily volume 50,000 shares; you hold 100,000 shares (2× average daily volume). Selling your full position would take multiple days and likely move the price significantly against you.",
        "risk_relevance": "Ensure at least 20-30% of the portfolio is in highly liquid assets (large-cap ETFs, Treasuries) for crisis resilience. Illiquid positions may show low volatility on paper but carry hidden risk in forced-sale scenarios.",
        "related":       ["concentration_risk", "var", "diversification"],
    },
}


# =============================================================================
# CONCEPT STORE
# =============================================================================

class ConceptStore:
    """
    Manages the static concept definitions knowledge base.

    Provides:
      generate_chunks() — for RAG ingestion
      lookup(key)       — for Intent 3 (specific metric) and Intent 6 (follow-up)
      search(query)     — keyword search across all concepts
    """

    def __init__(self):
        self._chunks: list[dict] = []
 
    def lookup(self, key: str) -> dict | None:
        return CONCEPTS.get(key)
 
    def search(self, query: str) -> list[dict]:
        q_words = set(query.lower().split())
        results = []
        for key, c in CONCEPTS.items():
            text = (c["label"] + " " + c["definition"] + " " +
                    c.get("detail", "") + " " + " ".join(c.get("related", []))).lower()
            score = sum(1 for w in q_words if w in text)
            if score > 0:
                results.append({"key": key, "score": score, **c})
        return sorted(results, key=lambda x: x["score"], reverse=True)
 
    # ── Chunk generation + cleaning ───────────────────────────────────────────
 
    def generate_chunks(self) -> list[dict]:
        if self._chunks:
            return self._chunks
 
        now    = datetime.datetime.now().isoformat()
        chunks = []
 
        for key, c in CONCEPTS.items():
            related_str = (
                "Related concepts: " + ", ".join(c.get("related", [])) + "."
                if c.get("related") else ""
            )
            raw_text = (
                f"{c['label']}: {c['definition']} "
                f"{c.get('detail', '')} "
                f"Example: {c.get('worked_example', '')} "
                f"Risk relevance: {c.get('risk_relevance', '')} "
                f"{related_str}"
            )
            chunks.append({
                "citation_id":  f"concept-{key}",
                "kb_source":    "concepts",
                "intent_tags":  ["concept_explanation", "full_analysis", "rebalance"],
                "source_url":   "internal://concepts",
                "source_type":  "curated_static",
                "updated_at":   now,
                "concept_key":  key,
                "label":        c["label"],
                "text":         _clean_chunk(raw_text),
            })
 
        self._chunks = chunks
        return chunks
 
    # ── TXT export ────────────────────────────────────────────────────────────
 
    def export_txt(self, output_path: str = OUTPUT_TXT) -> str:
        chunks = self.generate_chunks()
        now    = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
 
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"# kb4 CONCEPTS — {len(chunks)} definitions — generated {now}\n\n")
            for c in chunks:
                f.write(f"--- {c['concept_key']} ---\n")
                f.write(f"Label: {c['label']}\n")
                f.write(c["text"] + "\n\n")
 
        print(f"  ✓ kb4 TXT → {output_path}  ({len(chunks)} concepts)")
        return output_path
 
    def save_json(self) -> str:
        with open(CONCEPTS_FILE, "w", encoding="utf-8") as f:
            json.dump(CONCEPTS, f, indent=2)
        print(f"  ✓ kb4 JSON → {CONCEPTS_FILE}")
        return CONCEPTS_FILE
 
 
# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    cs = ConceptStore()
    chunks = cs.generate_chunks()
    cs.export_txt()
    cs.save_json()
    print(f"\n✅ {len(chunks)} concept chunks ready")
    for c in chunks:
        print(f"  [{c['concept_key']}] {c['label']}")