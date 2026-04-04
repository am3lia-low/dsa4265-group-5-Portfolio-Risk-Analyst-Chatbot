"""Run portfolio workflows after intent classification."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import sys
import os


from agent_tools.workflow_tools.agent_llm import classify_intent, Intent, IntentResult
from agent_tools.ml_risk_tools import current_portfolio_risk_tool

logger = logging.getLogger(__name__)

# Intent strings accepted by RAG_utils.retrieve_context
_RAG_INTENT_FULL = "full_analysis"
_RAG_INTENT_CONCEPT = "concept_explanation"
_RAG_INTENT_TREND = "trend_prediction"


@dataclass
class WorkflowResult:
    content: str
    intent: Intent
    secondary_intent: Optional[Intent] = None
    metadata: Optional[dict[str, Any]] = None


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


def _full_analysis_markdown(portfolio: dict, portfolio_changed: bool) -> str:
    rows = current_portfolio_risk_tool([portfolio])
    if not rows:
        return "Risk computation returned no data. Check tickers and date range."

    row = rows[0]
    metrics: dict = row.get("metrics") or {}
    score_bundle: dict = row.get("risk_score") or {}
    level = score_bundle.get("risk_level", "—")
    score = score_bundle.get("risk_score", 0.0)

    lines = [
        "## Risk summary",
        f"- **Level:** {level} (composite {float(score):.2f})",
        "",
        "## Numbers",
        f"- Volatility (annualized): **{metrics.get('volatility', 0):.2%}**",
        f"- VaR (5th pct. daily return): **{metrics.get('VaR', 0):.4f}**",
        f"- Sharpe (rf ≈ 0): **{metrics.get('sharpe', 0):.2f}**",
        f"- Max drawdown: **{metrics.get('max_drawdown', 0):.2%}**",
        f"- Avg. pairwise correlation: **{metrics.get('correlation', 0):.2f}**",
        f"- Concentration (HHI): **{metrics.get('concentration', 0):.2f}**",
    ]

    rag = _rag_block(_RAG_INTENT_FULL, "portfolio risk diversification rebalancing")
    if rag:
        lines.extend(["", rag])

    if portfolio_changed:
        lines.insert(1, "- **Note:** Holdings changed; side-by-side vs. prior snapshot is not implemented yet.")

    return "\n".join(lines)


def _metric_key_map() -> dict[str, tuple[str, str]]:
    return {
        "portfolio_volatility": ("volatility", "Volatility (annualized)"),
        "var_95": ("VaR", "VaR (5th pct. daily return)"),
        "max_drawdown": ("max_drawdown", "Max drawdown"),
        "sharpe_ratio": ("sharpe", "Sharpe ratio"),
        "hhi_concentration": ("concentration", "HHI concentration"),
        "avg_pairwise_correlation": ("correlation", "Avg. pairwise correlation"),
    }


def _specific_metric_markdown(
    portfolio: dict,
    extracted: Optional[list[str]],
) -> str:
    rows = current_portfolio_risk_tool([portfolio])
    if not rows:
        return "Could not load metrics for this portfolio."

    metrics: dict = rows[0].get("metrics") or {}
    mapping = _metric_key_map()

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

    if primary == Intent.FULL_ANALYSIS:
        body = _full_analysis_markdown(portfolio, portfolio_changed)
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
