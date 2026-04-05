"""
Test suite for the orchestrator's route_and_execute function.
Tests all workflow paths without calling external APIs or RAG modules.

Run: python test_orchestrator.py
"""

import sys
import os
from unittest.mock import patch, MagicMock
import pytest

# Add the project root to the Python path
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from agent_tools import route_and_execute, WorkflowResult
from agent_tools import Intent, IntentResult


# Mock portfolio data
MOCK_PORTFOLIO = {
    "tickers": ["AAPL", "GOOGL", "MSFT"],
    "weights": [0.4, 0.3, 0.3],
    "id": "test_portfolio"
}

# Mock risk calculation result
MOCK_RISK_RESULT = [{
    "portfolio_id": "test_portfolio",
    "risk_score": {
        "risk_score": 0.45,
        "risk_level": "Medium"
    },
    "metrics": {
        "volatility": 0.18,
        "VaR": -0.025,
        "sharpe": 1.2,
        "max_drawdown": -0.15,
        "correlation": 0.6,
        "concentration": 0.35
    }
}]


def create_intent_result(primary_intent, secondary_intent=None, extracted_metrics=None, extracted_concept=None):
    """Helper to create IntentResult objects for testing."""
    return IntentResult(
        primary_intent=primary_intent,
        confidence=0.95,
        reasoning="Test reasoning",
        secondary_intent=secondary_intent,
        extracted_metrics=extracted_metrics,
        extracted_concept=extracted_concept
    )


def test_full_analysis_workflow():
    """Test the full analysis workflow with mocked risk calculations."""
    intent_result = create_intent_result(Intent.FULL_ANALYSIS)

    with patch('agent_tools.workflow_tools.orchestrator.current_portfolio_risk_tool') as mock_risk_tool:
        mock_risk_tool.return_value = MOCK_RISK_RESULT

        result = route_and_execute(intent_result, MOCK_PORTFOLIO)

        assert isinstance(result, WorkflowResult)
        assert result.intent == Intent.FULL_ANALYSIS
        assert "Risk summary" in result.content
        assert "Medium" in result.content
        assert "Volatility (annualized): **18.00%**" in result.content
        assert "Reference material" in result.content


def test_specific_metric_workflow():
    """Test the specific metric workflow."""
    intent_result = create_intent_result(
        Intent.SPECIFIC_METRIC,
        extracted_metrics=["sharpe_ratio", "volatility"]
    )

    with patch('agent_tools.workflow_tools.orchestrator.current_portfolio_risk_tool') as mock_risk_tool:
        mock_risk_tool.return_value = MOCK_RISK_RESULT

        result = route_and_execute(intent_result, MOCK_PORTFOLIO)

        assert isinstance(result, WorkflowResult)
        assert result.intent == Intent.SPECIFIC_METRIC
        assert "Requested metrics" in result.content
        assert "**Sharpe ratio:** 1.2" in result.content
        assert "**Volatility (annualized):** 0.18" in result.content


def test_concept_explanation_workflow_with_rag():
    """Test concept explanation workflow with RAG available."""
    intent_result = create_intent_result(
        Intent.CONCEPT_EXPLANATION,
        extracted_concept="sharpe ratio"
    )

    # Mock RAG response
    mock_rag_response = MagicMock()
    mock_rag_response.chunks = [
        MagicMock(text="The Sharpe ratio measures risk-adjusted return.", kb_source="wiki"),
        MagicMock(text="Higher Sharpe ratios indicate better risk-adjusted performance.", kb_source="finance_book")
    ]

    with patch('agent_tools.retrieve_context', return_value=mock_rag_response):
        result = route_and_execute(intent_result, MOCK_PORTFOLIO)

        assert isinstance(result, WorkflowResult)
        assert result.intent == Intent.CONCEPT_EXPLANATION
        assert "## sharpe ratio" in result.content
        assert "Sharpe ratio measures risk-adjusted return" in result.content
        assert "Reference material" in result.content


def test_concept_explanation_workflow_no_rag():
    """Test concept explanation workflow when RAG is not available."""
    intent_result = create_intent_result(
        Intent.CONCEPT_EXPLANATION,
        extracted_concept="volatility"
    )

    with patch('agent_tools.rag_tools.RAG_utils.retrieve_context', side_effect=ImportError("RAG not available")):
        result = route_and_execute(intent_result, MOCK_PORTFOLIO)

        assert isinstance(result, WorkflowResult)
        assert result.intent == Intent.CONCEPT_EXPLANATION
        assert "## volatility" in result.content
        assert "No knowledge-base hits for this query yet" in result.content


def test_trend_prediction_workflow_with_rag():
    """Test trend prediction workflow with RAG available."""
    intent_result = create_intent_result(Intent.TREND_PREDICTION)

    # Mock RAG response
    mock_rag_response = MagicMock()
    mock_rag_response.chunks = [
        MagicMock(text="Market volatility tends to revert to the mean over time.", kb_source="research_paper"),
    ]

    with patch('agent_tools.rag_tools.RAG_utils.retrieve_context', return_value=mock_rag_response):
        result = route_and_execute(intent_result, MOCK_PORTFOLIO)

        assert isinstance(result, WorkflowResult)
        assert result.intent == Intent.TREND_PREDICTION
        assert "## Outlook" in result.content
        assert "Volatility forecasting is not enabled" in result.content
        assert "Reference material" in result.content


def test_trend_prediction_workflow_no_rag():
    """Test trend prediction workflow when RAG is not available."""
    intent_result = create_intent_result(Intent.TREND_PREDICTION)

    with patch('agent_tools.rag_tools.RAG_utils.retrieve_context', side_effect=Exception("RAG error")):
        result = route_and_execute(intent_result, MOCK_PORTFOLIO)

        assert isinstance(result, WorkflowResult)
        assert result.intent == Intent.TREND_PREDICTION
        assert "## Outlook" in result.content
        assert "Volatility forecasting is not enabled" in result.content


def test_follow_up_workflow_with_history():
    """Test follow-up workflow with conversation history."""
    intent_result = create_intent_result(Intent.FOLLOW_UP)

    history = [
        {"role": "assistant", "content": "Your portfolio has a Medium risk level with Sharpe ratio of 1.2."},
        {"role": "user", "content": "Can you explain that?"}
    ]

    result = route_and_execute(intent_result, MOCK_PORTFOLIO, recent_history=history)

    assert isinstance(result, WorkflowResult)
    assert result.intent == Intent.FOLLOW_UP
    assert "Last response" in result.content
    assert "Sharpe ratio of 1.2" in result.content


def test_follow_up_workflow_no_history():
    """Test follow-up workflow with no conversation history."""
    intent_result = create_intent_result(Intent.FOLLOW_UP)

    result = route_and_execute(intent_result, MOCK_PORTFOLIO)

    assert isinstance(result, WorkflowResult)
    assert result.intent == Intent.FOLLOW_UP
    assert "There's no earlier answer to refer to" in result.content


def test_general_chat_workflow():
    """Test general chat workflow."""
    intent_result = create_intent_result(Intent.GENERAL_CHAT)

    result = route_and_execute(intent_result, MOCK_PORTFOLIO)

    assert isinstance(result, WorkflowResult)
    assert result.intent == Intent.GENERAL_CHAT
    assert "I analyze this portfolio's risk" in result.content


def test_secondary_intent_handling():
    """Test that secondary intents are properly appended to the output."""
    intent_result = create_intent_result(
        Intent.FULL_ANALYSIS,
        secondary_intent=Intent.CONCEPT_EXPLANATION
    )

    with patch('agent_tools.workflow_tools.orchestrator.current_portfolio_risk_tool') as mock_risk_tool:
        mock_risk_tool.return_value = MOCK_RISK_RESULT

        result = route_and_execute(intent_result, MOCK_PORTFOLIO)

        assert isinstance(result, WorkflowResult)
        assert result.intent == Intent.FULL_ANALYSIS
        assert result.secondary_intent == Intent.CONCEPT_EXPLANATION
        assert "Also relevant: **concept explanation**" in result.content


def test_portfolio_changed_flag():
    """Test that portfolio_changed flag is handled correctly."""
    intent_result = create_intent_result(Intent.FULL_ANALYSIS)

    with patch('agent_tools.workflow_tools.orchestrator.current_portfolio_risk_tool') as mock_risk_tool:
        mock_risk_tool.return_value = MOCK_RISK_RESULT

        result = route_and_execute(intent_result, MOCK_PORTFOLIO, portfolio_changed=True)

        assert isinstance(result, WorkflowResult)
        assert result.intent == Intent.FULL_ANALYSIS
        assert "Note: Holdings changed" in result.content


def test_empty_risk_results():
    """Test handling of empty risk calculation results."""
    intent_result = create_intent_result(Intent.FULL_ANALYSIS)

    with patch('agent_tools.workflow_tools.orchestrator.current_portfolio_risk_tool') as mock_risk_tool:
        mock_risk_tool.return_value = []

        result = route_and_execute(intent_result, MOCK_PORTFOLIO)

        assert isinstance(result, WorkflowResult)
        assert result.intent == Intent.FULL_ANALYSIS
        assert "Risk computation returned no data" in result.content


def test_missing_metrics_in_risk_results():
    """Test handling of missing metrics in risk calculation results."""
    intent_result = create_intent_result(Intent.SPECIFIC_METRIC, extracted_metrics=["sharpe_ratio"])

    incomplete_risk_result = [{
        "portfolio_id": "test_portfolio",
        "risk_score": {"risk_score": 0.45, "risk_level": "Medium"},
        "metrics": {}  # Missing metrics
    }]

    with patch('agent_tools.workflow_tools.orchestrator.current_portfolio_risk_tool') as mock_risk_tool:
        mock_risk_tool.return_value = incomplete_risk_result

        result = route_and_execute(intent_result, MOCK_PORTFOLIO)

        assert isinstance(result, WorkflowResult)
        assert result.intent == Intent.SPECIFIC_METRIC
        assert "Requested metrics" in result.content


def test_unsupported_intent():
    """Test handling of unsupported intent types."""
    # Create a mock unsupported intent
    class UnsupportedIntent:
        value = "unsupported"

    intent_result = IntentResult(
        primary_intent=UnsupportedIntent(),  # type: ignore
        confidence=0.95,
        reasoning="Test reasoning"
    )

    result = route_and_execute(intent_result, MOCK_PORTFOLIO)

    assert isinstance(result, WorkflowResult)
    assert "Unsupported intent" in result.content


if __name__ == "__main__":
    # Run all test functions
    test_functions = [
        test_full_analysis_workflow,
        test_specific_metric_workflow,
        test_concept_explanation_workflow_with_rag,
        test_concept_explanation_workflow_no_rag,
        test_trend_prediction_workflow_with_rag,
        test_trend_prediction_workflow_no_rag,
        test_follow_up_workflow_with_history,
        test_follow_up_workflow_no_history,
        test_general_chat_workflow,
        test_secondary_intent_handling,
        test_portfolio_changed_flag,
        test_empty_risk_results,
        test_missing_metrics_in_risk_results,
        test_unsupported_intent,
    ]

    passed = 0
    failed = 0

    for test_func in test_functions:
        try:
            test_func()
            print(f"PASS: {test_func.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL: {test_func.__name__}: {e}")
            failed += 1

    print(f"\nTest Results: {passed} passed, {failed} failed")

    if failed > 0:
        sys.exit(1)