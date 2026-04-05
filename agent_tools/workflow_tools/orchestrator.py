"""Run portfolio workflows after intent classification."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional
import pandas as pd

import sys
import os

import datetime
from agent_tools.data_tools import fetch_price_data, calculate_returns
from agent_tools.quant_tools import calculate_all_metrics, metric_benchmarks, calculate_risk_contribution, calculate_covariance_matrix
from agent_tools.workflow_tools import classify_intent, Intent, IntentResult
from agent_tools.ml_risk_tools import current_portfolio_risk_tool, future_portfolio_risk

logger = logging.getLogger(__name__)

# Intent strings accepted by RAG_utils.retrieve_context
_RAG_INTENT_FULL = "full_analysis"
_RAG_INTENT_CONCEPT = "concept_explanation"
_RAG_INTENT_TREND = "trend_prediction"

# Silent mode for RAG outputs
RAG_SILENT_MODE = False

def set_rag_silent_mode(silent: bool = True):
    """Set the global RAG silent mode."""
    global RAG_SILENT_MODE
    RAG_SILENT_MODE = silent

# extracting data & quant stuff
def data_and_metrics(tickers, weights, year=5):
    END = datetime.date.today()
    START = END.replace(year=END.year - year)

    # extract data
    prices = fetch_price_data(
        tickers=tickers,
        start=str(START), end=str(END),
        include_spy=True
    )
    returns = calculate_returns(prices, method="simple")

    # calculate metrics
    all_metrics = calculate_all_metrics(returns=returns, weights=weights)
    metric_analysis = metric_benchmarks(all_metrics)

    return (all_metrics, metric_analysis)

# Maps quant_module metric keys to their proper quantitative finance names.
metric_key_map = {
        "portfolio_volatility":     "Annualised Portfolio Volatility",
        "var_95":                   "Value at Risk (95% confidence, historical simulation)",
        "cvar_95":                  "Conditional Value at Risk / Expected Shortfall (95%)",
        "max_drawdown":             "Maximum Drawdown (peak-to-trough)",
        "sharpe_ratio":             "Sharpe Ratio (rf = 4%)",
        "sortino_ratio":            "Sortino Ratio (rf = 4%)",
        "beta":                     "Market Beta (vs. SPY)",
        "hhi_concentration":        "Herfindahl-Hirschman Index (HHI)",
        "avg_pairwise_correlation": "Average Pairwise Correlation",
        "vol_of_vol":               "Volatility of Volatility (21-day rolling)",
        "skewness":                 "Return Distribution Skewness",
        "excess_kurtosis":          "Excess Kurtosis (Fisher)",
        "risk_contribution":        "Per-Asset Risk Contribution (%)",
    }

# lstm window
FUTURE_VOL_WINDOW = 60

@dataclass
# outputs whatever ui's update_cache will take in + content to put into generate_explanation
class WorkflowResult:
    content: str
    intent: Intent
    secondary_intent: Optional[Intent] = None
    portfolio: dict
    returns_df: pd.DataFrame
    cov_matrix: pd.DataFrame
    metrics: dict
    risk_contributions: dict
    risk_level: dict
    trend_forecast: dict
    rag_context: str


def _append_secondary_line(base: str, secondary: Optional[Intent]) -> str:
    if secondary is None:
        return base
    return f"{base.rstrip()}\n\nAlso relevant: **{secondary.value.replace('_', ' ')}**."


def _rag_block(intent: str, query: str, top_k: int = 6) -> Optional[str]:
    try:
        from agent_tools.rag_tools.RAG_utils import retrieve_context
    except ImportError:
        logger.debug("RAG_utils not available", exc_info=True)
        return None

    try:
        result = retrieve_context(
            intent=intent,
            query=query,
            top_k=top_k,
            save_log=False,
            silent=RAG_SILENT_MODE,
        )
    except Exception as exc:
        logger.warning("retrieve_context failed: %s", exc)
        return None

    if not result.chunks:
        return None

    parts: list[str] = ["**Reference material**"]
    for i, chunk in enumerate(result.chunks[:top_k], 1):
        text = (chunk.text or "").strip().replace("\n", " ")
        if len(text) > 450:
            text = text[:447] + "…"
        parts.append(f"{i}. ({chunk.kb_source}) {text}")
    return "\n".join(parts)


def _full_analysis_markdown(portfolio: dict, portfolio_changed: bool, all_metrics: dict, metric_analysis: dict) -> str:
    """
    what needs to go in
    [BEFORE THAT] calculate_all_metrics -> metric_analysis
    [ADDED] current_portfolio_risk_tool
    [ADDED] predict_volatility_trend
    [ADDED] retrieve_context
    """

    print("     classifying portfolio risk...")
    current_risk = current_portfolio_risk_tool(portfolio, all_metrics)

    print("     organising metrics...")
    lines = [
        "## Risk summary",
        f"- **Level:** {current_risk["risk_level"]} (composite {float(current_risk["risk_score"]):.2f})",
        "## Metrics",
    ]
    for metric, details in metric_analysis.items():
        lines.append(
            f"- {metric_key_map[metric]}: {details["value"]:.2%}, {details["label"]}, {details["comment"]}"
        )

    print("     predicting future volatility...")
    future_risk = future_portfolio_risk(portfolio, FUTURE_VOL_WINDOW)
    lines.append (
        "## Future volatility for the next {FUTURE_VOL_WINDOW} days",
        f"Predicted Volatility: {future_risk["predicted_volatility"]}",
        f"Predicted Direction: {future_risk["predicted_direction"]} with a probability of {future_risk["prob_up"]}.",
        f"Overall Confidence of this prediction: {future_risk["confidence"]}"
    )

    print("     running RAG retrieval...")
    rag = _rag_block(_RAG_INTENT_FULL, "portfolio risk diversification rebalancing")
    if rag:
        lines.extend(["", rag])

    if portfolio_changed:
        lines.insert(1, "- **Note:** Holdings changed; side-by-side vs. prior snapshot is not implemented yet.")

    print("     gathering all the answers...")
    WorkflowResult(
        content="\n".join(lines),
        inetnt=Intent.FULL_ANALYSIS,
        portfolio=portfolio
    )

    return "\n".join(lines)


def _specific_metric_markdown(
    portfolio: dict,
    extracted: Optional[list[str]],
) -> str:
    rows = current_portfolio_risk_tool([portfolio])
    if not rows:
        return "Could not load metrics for this portfolio."

    metrics: dict = rows[0].get("metrics") or {}

    if not extracted:
        return (
            "Say which measure you want (e.g. Sharpe, VaR, drawdown, volatility), "
            "or ask for a full risk summary."
        )

    lines: list[str] = ["## Requested metrics"]
    pending: list[str] = []

    for name in extracted[:8]:
        if name in mapping:
            key, label = mapping[name]
            lines.append(f"- **{label}:** {metrics.get(key)}")
        else:
            pending.append(name)

    if pending:
        lines.append("")
        lines.append(
            "Not available in the current engine: "
            + ", ".join(f"`{n}`" for n in pending)
        )

    q = " ".join(extracted[:3])
    rag = _rag_block(_RAG_INTENT_CONCEPT, f"define explain portfolio metrics {q}")
    if rag:
        lines.extend(["", rag])

    return "\n".join(lines)


def _last_assistant_text(history: list[dict]) -> Optional[str]:
    for msg in reversed(history):
        if msg.get("role") != "assistant":
            continue
        if msg.get("type") == "status":
            continue
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
    return None


def _follow_up_markdown(history: list[dict]) -> str:
    prior = _last_assistant_text(history)
    if not prior:
        return (
            "There’s no earlier answer to refer to. "
            "Ask for a metric or a full risk summary for this portfolio."
        )

    head = prior[:900] + ("…" if len(prior) > 900 else "")
    return (
        "## Last response\n\n"
        f"{head}\n\n"
        "What would you like clarified or expanded? "
        "Naming a metric or risk angle helps."
    )


def _concept_markdown(concept: Optional[str]) -> str:
    label = (concept or "risk concept").strip()
    base = f"## {label}\n\n"

    rag = _rag_block(_RAG_INTENT_CONCEPT, label)
    if rag:
        return base + rag

    return base + (
        "No knowledge-base hits for this query yet. "
        "Try rephrasing the term, or ask how it applies to your current portfolio numbers."
    )


def _trend_markdown() -> str:
    lines = [
        "## Outlook",
        "Volatility forecasting is not enabled in this build.",
        "",
        "You can still use the risk summary for where the portfolio stands today.",
    ]
    rag = _rag_block(_RAG_INTENT_TREND, "market volatility regime forward outlook")
    if rag:
        lines.extend(["", rag])
    return "\n".join(lines)


def _general_chat_markdown() -> str:
    return (
        "I analyze this portfolio’s risk: full summary, individual metrics, "
        "definitions, and (when available) context from your knowledge base."
    )


def route_and_execute(
    intent_result: IntentResult,
    portfolio: dict,
    portfolio_changed: bool = False,
    recent_history: Optional[list[dict]] = None,
) -> WorkflowResult:
    history = recent_history or []
    primary = intent_result.primary_intent
    secondary = intent_result.secondary_intent

    # 1. if portfolio change has been detected, data and metrics will be recalibrated here
    if portfolio_changed:
        print("calibrating portfolio data & calculating metrics...")
        all_metrics, metric_analysis = data_and_metrics(portfolio["tickers"], portfolio["weights"])

    if primary == Intent.FULL_ANALYSIS:
        print("executing full analysis workflow...")
        body = _full_analysis_markdown(portfolio, portfolio_changed, all_metrics, metric_analysis)
    elif primary == Intent.SPECIFIC_METRIC:
        body = _specific_metric_markdown(portfolio, intent_result.extracted_metrics)
    elif primary == Intent.CONCEPT_EXPLANATION:
        body = _concept_markdown(intent_result.extracted_concept)
    elif primary == Intent.TREND_PREDICTION:
        body = _trend_markdown()
    elif primary == Intent.FOLLOW_UP:
        body = _follow_up_markdown(history)
    elif primary == Intent.GENERAL_CHAT:
        body = _general_chat_markdown()
    else:
        body = "Unsupported intent."

    content = _append_secondary_line(body, secondary)

    return WorkflowResult(
        content=content.strip(),
        intent=primary,
        secondary_intent=secondary,
        metadata={
            "reasoning": intent_result.reasoning,
            "confidence": intent_result.confidence,
        },
    )
