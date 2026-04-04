"""Shared tools for the Portfolio Risk Analyst Agent."""

# Data tools (these should always be available)
from .data_tools import (
    fetch_price_data,
    calculate_returns,
    valid_tickers,
    valid_weights
)

# RAG: Retrieve_context()
from agent_tools.rag_tools.RAG_utils import retrieve_context  # Added: Import retrieve_context 

# ML risk tools
from .ml_risk_tools import (
    current_portfolio_risk_tool,
    future_portfolio_risk
)

# Workflow tools
from .workflow_tools import (
    classify_intent,
    Intent,
    IntentResult,
    route_and_execute,
    WorkflowResult
)

# Export everything
__all__ = [
    "fetch_price_data",
    "calculate_returns",
    "valid_tickers",
    "valid_weights",
    "classify_intent",
    "Intent",
    "IntentResult",
    "retrieve_context",
    "current_portfolio_risk_tool",
    "future_portfolio_risk",
    "route_and_execute",
    "WorkflowResult"
]

# Only export workflow tools if they're available
# if _has_workflow_tools:
#     __all__.extend([
#         "classify_intent",
#         "Intent",
#         "IntentResult",
#         "retrieve_context", # I've changed the final result into a Pydantic model (View RAG.ipynb for examples!)
#     ])
