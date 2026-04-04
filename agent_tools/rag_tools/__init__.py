"""
RAG Tools for Portfolio Risk Analyst Agent
=========================================

Unified RAG pipeline with multiple knowledge bases for risk analysis.
"""

# Knowledge Base modules
from .kb0_ticker_resolver import resolve_tickers_from_query
from .kb1_generate_tickers import (
    generate_tickers,
    convert_tickers_into_txt,
    build_ticker_meta as _build_ticker_meta,
    OUTPUT_DIR_HTML, OUTPUT_TXT
)
from .kb2_macro_regime import MacroStore
from .kb3_concepts import ConceptStore
from .kb4_strategies import StrategyStore

# RAG Utilities
from .RAG_utils import retrieve_context

__all__ = [
    # KB0 - Ticker Resolver
    "resolve_tickers_from_query",

    # KB1 - Ticker Profiles
    "generate_tickers",
    "convert_tickers_into_txt",
    "OUTPUT_DIR_HTML",
    "OUTPUT_TXT",

    # KB2 - Macro Store
    "MacroStore",

    # KB3 - Concept Store
    "ConceptStore",

    # KB4 - Strategy Store
    "StrategyStore",

    # RAG Utilities
    "retrieve_context",
]