"""Shared tools for the Portfolio Risk Analyst Agent."""

# Data tools (these should always be available)
from .data_tools import (
    fetch_price_data,
    calculate_returns,
    valid_tickers,
    valid_weights
)

# RAG: Retrieve_context()
from .rag_tools import resolve_tickers_from_query
from .rag_tools import (
    generate_tickers,
    convert_tickers_into_txt,
    _build_ticker_meta,
    OUTPUT_DIR_HTML, OUTPUT_TXT
)
from .rag_tools import MacroStore
from .rag_tools import ConceptStore
from .rag_tools import StrategyStore
from .rag_tools import retrieve_context  # Added: Import retrieve_context 

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
    # data
    "fetch_price_data",
    "calculate_returns",
    "valid_tickers",
    "valid_weights",

    # agent
    "classify_intent",
    "Intent",
    "IntentResult",
    "route_and_execute",
    "WorkflowResult"

    # rag
    "resolve_tickers_from_query",
    "generate_tickers",
    "convert_tickers_into_txt",
    "OUTPUT_DIR_HTML",
    "OUTPUT_TXT",
    "MacroStore",
    "ConceptStore",
    "StrategyStore",
    "retrieve_context",

    # ml
    "current_portfolio_risk_tool",
    "future_portfolio_risk",
]

# Only export workflow tools if they're available
# if _has_workflow_tools:
#     __all__.extend([
#         "classify_intent",
#         "Intent",
#         "IntentResult",
#         "retrieve_context", # I've changed the final result into a Pydantic model (View RAG.ipynb for examples!)
#     ])
