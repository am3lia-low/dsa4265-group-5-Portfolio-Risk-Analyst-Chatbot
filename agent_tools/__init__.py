"""Shared tools for the Portfolio Risk Analyst Agent."""

# Data tools (these should always be available)
from .data_tools import (
    fetch_price_data,
    calculate_returns,
    valid_tickers,
    valid_weights
)

# Workflow tools (these may have optional dependencies)
try:
    from .workflow_tools import (
        classify_intent,
        Intent,
        IntentResult
    )
    _has_workflow_tools = True
except ImportError:
    # Workflow tools require additional dependencies (Google Generative AI)
    _has_workflow_tools = False
    classify_intent = None
    Intent = None
    IntentResult = None

# Export everything
__all__ = [
    "fetch_price_data",
    "calculate_returns",
    "valid_tickers",
    "valid_weights"
]

# Only export workflow tools if they're available
if _has_workflow_tools:
    __all__.extend([
        "classify_intent",
        "Intent",
        "IntentResult"
    ])
