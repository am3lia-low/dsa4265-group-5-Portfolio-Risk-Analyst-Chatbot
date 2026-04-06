"""Run portfolio workflows after intent classification."""

from __future__ import annotations

import logging
import datetime
from dataclasses import dataclass, field
from typing import Optional

from agent_tools.data_tools import fetch_price_data, calculate_returns
from agent_tools.quant_tools import calculate_all_metrics, metric_benchmarks
from agent_tools.workflow_tools import classify_intent, Intent, IntentResult
from agent_tools.ml_risk_tools import current_portfolio_risk_tool, future_portfolio_risk

logger = logging.getLogger(__name__)

# RAG intent strings accepted by RAG_utils.retrieve_context
_RAG_INTENT_FULL    = "full_analysis"
_RAG_INTENT_CONCEPT = "concept_explanation"
_RAG_INTENT_TREND   = "trend_prediction"

# LSTM forecast window (days)
FUTURE_VOL_WINDOW = 60

# Which intents require which computations
_NEEDS_METRICS = {Intent.FULL_ANALYSIS, Intent.SPECIFIC_METRIC, Intent.TREND_PREDICTION}
_NEEDS_RISK    = {Intent.FULL_ANALYSIS}
_NEEDS_LSTM    = {Intent.FULL_ANALYSIS, Intent.TREND_PREDICTION}

# Maps all_metrics keys to human-readable labels
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

# Dimensionless ratio metrics — formatted as floats, not percentages
_RATIO_METRICS = {
    "sharpe_ratio", "sortino_ratio", "beta",
    "skewness", "excess_kurtosis", "vol_of_vol",
}


# ---------------------------------------------------------------------------
# WorkflowResult
# ---------------------------------------------------------------------------

@dataclass
class WorkflowResult:
    content: str
    intent: Intent
    secondary_intent: Optional[Intent] = None
    cache: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# RAG helper
# ---------------------------------------------------------------------------

def _rag_block(intent: str, query: str, top_k: int = 6) -> Optional[str]:
    try:
        from agent_tools.rag_tools.RAG_utils import retrieve_context
    except ImportError:
        logger.debug("RAG_utils not available", exc_info=True)
        return None

    try:
        result = retrieve_context(intent=intent, query=query, top_k=top_k, save_log=False, silent=True)
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


# ---------------------------------------------------------------------------
# Fuzzy metric matching
# ---------------------------------------------------------------------------

_METRIC_ALIASES: dict[str, str] = {
    "vol":                      "portfolio_volatility",
    "volatility":               "portfolio_volatility",
    "var":                      "var_95",
    "value_at_risk":            "var_95",
    "cvar":                     "cvar_95",
    "expected_shortfall":       "cvar_95",
    "drawdown":                 "max_drawdown",
    "mdd":                      "max_drawdown",
    "max_drawdown":             "max_drawdown",
    "sharpe":                   "sharpe_ratio",
    "sharpe_ratio":             "sharpe_ratio",
    "sortino":                  "sortino_ratio",
    "sortino_ratio":            "sortino_ratio",
    "beta":                     "beta",
    "hhi":                      "hhi_concentration",
    "concentration":            "hhi_concentration",
    "hhi_concentration":        "hhi_concentration",
    "correlation":              "avg_pairwise_correlation",
    "avg_pairwise_correlation": "avg_pairwise_correlation",
    "skew":                     "skewness",
    "skewness":                 "skewness",
    "kurtosis":                 "excess_kurtosis",
    "excess_kurtosis":          "excess_kurtosis",
    "risk_contribution":        "risk_contribution",
    "vol_of_vol":               "vol_of_vol",
}


def _fuzzy_match_metric(name: str) -> Optional[str]:
    """Map a user-provided metric name to a metric_key_map key."""
    normalised = name.lower().replace(" ", "_").replace("-", "_")

    if normalised in metric_key_map:
        return normalised

    if normalised in _METRIC_ALIASES:
        return _METRIC_ALIASES[normalised]

    for key in metric_key_map:
        if normalised in key or key in normalised:
            return key

    return None


def _format_metric_value(key: str, value) -> str:
    if not isinstance(value, (int, float)):
        return str(value)
    if key in _RATIO_METRICS:
        return f"{value:.4f}"
    return f"{value:.2%}"


# ---------------------------------------------------------------------------
# Shared metric lines builder (reused by full_analysis and comparison block)
# ---------------------------------------------------------------------------

def _non_rag_lines(cache: dict) -> list[str]:
    metric_analysis = cache["metrics"]["metric_analysis"]
    all_metrics     = cache["metrics"]["all_metrics"]
    current_risk    = cache["risk_level"]
    future_risk     = cache.get("trend_forecast")

    lines = [
        "## Risk Summary",
        f"- **Level:** {current_risk['label']} "
        f"(composite score: {float(current_risk['confidence']):.2f})",
        "",
        "## Metrics",
    ]

    for metric, details in metric_analysis.items():
        if metric == "risk_contribution":
            continue
        label = metric_key_map.get(metric, metric)
        value = _format_metric_value(metric, details["value"])
        lines.append(f"- **{label}:** {value} — {details['label']}, {details['comment']}")

    risk_contrib = all_metrics.get("risk_contribution")
    if risk_contrib and isinstance(risk_contrib, dict):
        lines.append("")
        lines.append("## Risk Contribution by Asset")
        for ticker, pctg in risk_contrib.items():
            lines.append(f"- {ticker}: {pctg:.2%}")

    if future_risk:
        lines.extend([
            "",
            f"## Volatility Forecast (next {FUTURE_VOL_WINDOW} days)",
            f"- **Predicted Direction:** {future_risk['predicted_direction']}",
            f"- **Predicted Volatility:** {future_risk['predicted_volatility']:.4f}",
            f"- **Probability (Up):** {future_risk['prob_up']:.2%}",
            f"- **Model Confidence:** {future_risk['confidence']:.2%}",
        ])

    return lines


# ---------------------------------------------------------------------------
# Intent-specific markdown builders
# ---------------------------------------------------------------------------

def _full_analysis_markdown(
    portfolio: dict,
    portfolio_changed: bool,
    is_first_portfolio: bool,
    old_cache: dict,
    working_cache: dict,
) -> str:
    logger.info("     building full analysis output...")
    lines = ["# Portfolio Risk Analysis"]
    lines.extend(_non_rag_lines(working_cache))

    if portfolio_changed and not is_first_portfolio and old_cache.get("metrics"):
        logger.info("     appending previous portfolio comparison...")
        lines.extend([
            "",
            "---",
            "# Previous Portfolio (Before Change)",
        ])
        lines.extend(_non_rag_lines(old_cache))

    logger.info("     running RAG retrieval for full analysis...")
    tickers_str = " ".join(portfolio["tickers"])
    rag = _rag_block(
        _RAG_INTENT_FULL,
        f"{tickers_str} portfolio risk concentration rebalancing strategy",
    )
    working_cache["rag_context"] = rag
    if rag:
        lines.extend(["", rag])

    return "\n".join(lines)


def _specific_metric_markdown(
    portfolio: dict,
    cache: dict,
    extracted_metrics: Optional[list[str]],
) -> str:
    if not extracted_metrics:
        return (
            "Please specify which metric(s) you'd like to know about — "
            "e.g. Sharpe ratio, VaR, max drawdown, volatility."
        )

    all_metrics     = cache["metrics"]["all_metrics"]
    metric_analysis = cache["metrics"]["metric_analysis"]

    lines: list[str] = ["## Requested Metrics"]
    unmatched: list[str] = []

    for name in extracted_metrics:
        key = _fuzzy_match_metric(name)
        if key == "risk_contribution":
            risk_contrib = all_metrics.get("risk_contribution", {})
            if risk_contrib and isinstance(risk_contrib, dict):
                lines.append("- **Per-Asset Risk Contribution:**")
                for ticker, pctg in risk_contrib.items():
                    lines.append(f"  - {ticker}: {pctg:.2%}")
            else:
                lines.append("- **Per-Asset Risk Contribution:** not available")
        elif key and key in all_metrics:
            label   = metric_key_map.get(key, key)
            value   = _format_metric_value(key, all_metrics[key])
            details = metric_analysis.get(key, {})
            comment = f" — {details['label']}, {details['comment']}" if details else ""
            lines.append(f"- **{label}:** {value}{comment}")
        else:
            unmatched.append(name)

    if unmatched:
        lines.append(
            "\nCould not find: " + ", ".join(f"`{n}`" for n in unmatched) + ". "
            "Available metrics: " + ", ".join(metric_key_map.keys()) + "."
        )

    logger.info("     running RAG retrieval for specific metrics...")
    q = " ".join(extracted_metrics[:3])
    rag = _rag_block(_RAG_INTENT_CONCEPT, f"define explain {q}")
    if rag:
        lines.extend(["", rag])

    return "\n".join(lines)


def _trend_markdown(portfolio: dict, cache: dict) -> str:
    future_risk = cache.get("trend_forecast")
    if not future_risk:
        return (
            "No volatility forecast is available yet. "
            "Please submit your portfolio for analysis first."
        )

    lines = [
        "## Volatility Outlook",
        f"- **Predicted Direction:** {future_risk['predicted_direction']}",
        f"- **Predicted Volatility:** {future_risk['predicted_volatility']:.4f}",
        f"- **Probability (Up):** {future_risk['prob_up']:.2%}",
        f"- **Model Confidence:** {future_risk['confidence']:.2%}",
        "",
        "> **Disclaimer:** This is a model-based estimate, not a guarantee. "
        "Sudden market events or regime changes may invalidate the prediction.",
    ]

    logger.info("     running RAG retrieval for trend prediction...")
    tickers_str = " ".join(portfolio["tickers"])
    rag = _rag_block(
        _RAG_INTENT_TREND,
        f"{tickers_str} volatility regime outlook forecast",
    )
    if rag:
        lines.extend(["", rag])

    return "\n".join(lines)


def _concept_markdown(concept: Optional[str]) -> str:
    label = (concept or "risk concept").strip()
    logger.info("     running RAG retrieval for concept: %s...", label)
    rag = _rag_block(_RAG_INTENT_CONCEPT, label)

    base = f"## {label}\n\n"
    if rag:
        return base + rag

    return base + (
        "No knowledge-base entries found for this concept. "
        "Try rephrasing, or ask how it applies to your portfolio numbers."
    )


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
            "There's no earlier answer to refer to. "
            "Ask for a metric or a full risk summary for this portfolio."
        )
    head = prior[:900] + ("…" if len(prior) > 900 else "")
    return (
        "## Last Response\n\n"
        f"{head}\n\n"
        "What would you like clarified or expanded? "
        "Naming a specific metric or risk angle helps."
    )


def _general_chat_markdown() -> str:
    return (
        "I can help you analyze your portfolio's risk. Here's what I can do:\n"
        "- **Full analysis** — overall risk level, all key metrics, and volatility forecast\n"
        "- **Specific metrics** — Sharpe, VaR, drawdown, beta, and more\n"
        "- **Trend prediction** — LSTM-based volatility outlook\n"
        "- **Concept explanations** — plain-language definitions with portfolio context"
    )


# ---------------------------------------------------------------------------
# Dispatcher — calls the right markdown builder for a given intent
# ---------------------------------------------------------------------------

def _body_for_intent(
    intent: Intent,
    portfolio: dict,
    portfolio_changed: bool,
    is_first_portfolio: bool,
    old_cache: dict,
    working_cache: dict,
    history: list[dict],
    extracted_metrics: Optional[list[str]],
    extracted_concept: Optional[str],
) -> str:
    if intent == Intent.FULL_ANALYSIS:
        return _full_analysis_markdown(
            portfolio, portfolio_changed, is_first_portfolio, old_cache, working_cache,
        )
    elif intent == Intent.SPECIFIC_METRIC:
        return _specific_metric_markdown(portfolio, working_cache, extracted_metrics)
    elif intent == Intent.TREND_PREDICTION:
        return _trend_markdown(portfolio, working_cache)
    elif intent == Intent.CONCEPT_EXPLANATION:
        return _concept_markdown(extracted_concept)
    elif intent == Intent.FOLLOW_UP:
        return _follow_up_markdown(history)
    elif intent == Intent.GENERAL_CHAT:
        return _general_chat_markdown()
    else:
        return "Unsupported intent."


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def route_and_execute(
    intent_result: IntentResult,
    portfolio: dict,
    is_first_portfolio: bool,
    portfolio_changed: bool = False,
    recent_history: Optional[list[dict]] = None,
    cache: Optional[dict] = None,
) -> WorkflowResult:
    history   = recent_history or []
    old_cache = dict(cache) if cache else {}
    primary   = intent_result.primary_intent
    secondary = intent_result.secondary_intent

    logger.info(
        "route_and_execute | primary=%s | secondary=%s | portfolio_changed=%s | is_first=%s",
        primary, secondary, portfolio_changed, is_first_portfolio,
    )
    print(f"\n[route_and_execute]")
    print(f"  intent:            {primary.value}" + (f" + {secondary.value}" if secondary else ""))
    print(f"  portfolio:         {portfolio['tickers']}")
    print(f"  portfolio_changed: {portfolio_changed} | is_first_portfolio: {is_first_portfolio}")

    # Working cache starts from existing cache so unchanged data is preserved
    working_cache = dict(old_cache)

    # Determine what computations are needed (union of primary + secondary intents)
    intents = {primary}
    if secondary:
        intents.add(secondary)

    needs_metrics = bool(intents & _NEEDS_METRICS)
    needs_risk    = bool(intents & _NEEDS_RISK)
    needs_lstm    = bool(intents & _NEEDS_LSTM)

    metrics_stale = portfolio_changed or working_cache.get("metrics") is None
    risk_stale    = portfolio_changed or working_cache.get("risk_level") is None
    lstm_stale    = portfolio_changed or working_cache.get("trend_forecast") is None

    print(f"  tools needed:      metrics={needs_metrics} | risk={needs_risk} | lstm={needs_lstm}")
    print(f"  cache stale:       metrics={metrics_stale} | risk={risk_stale} | lstm={lstm_stale}")

    # ------------------------------------------------------------------
    # 1. Fetch price data + compute metrics (if needed and stale)
    # ------------------------------------------------------------------
    all_metrics     = None
    metric_analysis = None

    if needs_metrics and metrics_stale:
        print(f"\n  [1/3] fetching price data ({portfolio['tickers']}, 5yr)...")
        logger.info("Fetching price data and computing metrics...")
        END    = datetime.date.today()
        START  = END.replace(year=END.year - 5)
        prices  = fetch_price_data(
            tickers=portfolio["tickers"],
            start=str(START), end=str(END),
            include_spy=True,
        )
        returns = calculate_returns(prices, method="simple")
        working_cache["returns_df"] = returns
        print(f"        price data fetched: {returns.shape[0]} rows x {returns.shape[1]} assets")

        print(f"        calculating quantitative metrics...")
        all_metrics     = calculate_all_metrics(returns=returns, weights=portfolio["weights"])
        print(f"        metrics computed: {list(all_metrics.keys())}")

        print(f"        benchmarking metrics...")
        metric_analysis = metric_benchmarks(all_metrics)
        working_cache["metrics"] = {
            "all_metrics":     all_metrics,
            "metric_analysis": metric_analysis,
        }
        logger.info("Metrics computed successfully.")
        print(f"        done.")

    elif needs_metrics and not metrics_stale:
        print(f"\n  [1/3] using cached metrics (skipping fetch + computation)")
        logger.info("Using cached metrics.")
        all_metrics     = working_cache["metrics"]["all_metrics"]
        metric_analysis = working_cache["metrics"]["metric_analysis"]

    else:
        print(f"\n  [1/3] metrics not needed for intent: {primary.value}")

    # ------------------------------------------------------------------
    # 2. Risk classification (if needed and stale)
    # ------------------------------------------------------------------
    if needs_risk and risk_stale and all_metrics is not None:
        print(f"\n  [2/3] running risk classification (neural network scoring)...")
        logger.info("Classifying portfolio risk...")
        raw_risk = current_portfolio_risk_tool(portfolio, all_metrics)
        # Align keys to ExplanationContext expectations: label + confidence
        working_cache["risk_level"] = {
            "label":      raw_risk.get("risk_level"),
            "confidence": raw_risk.get("risk_score"),
        }
        logger.info("Risk classification: %s", working_cache["risk_level"]["label"])
        print(f"        risk level: {working_cache['risk_level']['label']} "
              f"(score: {working_cache['risk_level']['confidence']:.4f})")

    elif needs_risk and not risk_stale:
        print(f"\n  [2/3] using cached risk level: {working_cache['risk_level']['label']}")
        logger.info("Using cached risk level.")

    else:
        print(f"\n  [2/3] risk classification not needed for intent: {primary.value}")

    # ------------------------------------------------------------------
    # 3. LSTM volatility forecast (if needed and stale)
    # ------------------------------------------------------------------
    if needs_lstm and lstm_stale:
        print(f"\n  [3/3] running LSTM volatility forecast (window={FUTURE_VOL_WINDOW} days)...")
        logger.info("Running LSTM volatility forecast...")
        working_cache["trend_forecast"] = future_portfolio_risk(portfolio, FUTURE_VOL_WINDOW)
        forecast = working_cache["trend_forecast"]
        logger.info("LSTM forecast complete: direction=%s", forecast.get("predicted_direction"))
        print(f"        direction: {forecast['predicted_direction']} | "
              f"prob_up: {forecast['prob_up']:.2%} | "
              f"confidence: {forecast['confidence']:.2%}")

    elif needs_lstm and not lstm_stale:
        forecast = working_cache["trend_forecast"]
        print(f"\n  [3/3] using cached LSTM forecast: "
              f"{forecast['predicted_direction']} (confidence: {forecast['confidence']:.2%})")
        logger.info("Using cached LSTM forecast.")

    else:
        print(f"\n  [3/3] LSTM forecast not needed for intent: {primary.value}")

    # ------------------------------------------------------------------
    # 4. Build primary response body
    # ------------------------------------------------------------------
    print(f"\n  [response] building primary response ({primary.value})...")
    logger.info("Generating primary response for intent: %s", primary)
    primary_body = _body_for_intent(
        intent=primary,
        portfolio=portfolio,
        portfolio_changed=portfolio_changed,
        is_first_portfolio=is_first_portfolio,
        old_cache=old_cache,
        working_cache=working_cache,
        history=history,
        extracted_metrics=intent_result.extracted_metrics,
        extracted_concept=intent_result.extracted_concept,
    )

    # ------------------------------------------------------------------
    # 5. Build secondary response body (if present)
    # ------------------------------------------------------------------
    content = primary_body
    if secondary:
        print(f"  [response] building secondary response ({secondary.value})...")
        logger.info("Generating secondary response for intent: %s", secondary)
        secondary_body = _body_for_intent(
            intent=secondary,
            portfolio=portfolio,
            portfolio_changed=portfolio_changed,
            is_first_portfolio=is_first_portfolio,
            old_cache=old_cache,
            working_cache=working_cache,
            history=history,
            extracted_metrics=intent_result.extracted_metrics,
            extracted_concept=intent_result.extracted_concept,
        )
        content = f"{primary_body}\n\n---\n\n## Additionally\n\n{secondary_body}"

    print(f"  [done] route_and_execute complete\n")
    logger.info("route_and_execute complete.")

    return WorkflowResult(
        content=content.strip(),
        intent=primary,
        secondary_intent=secondary,
        cache=working_cache,
    )
