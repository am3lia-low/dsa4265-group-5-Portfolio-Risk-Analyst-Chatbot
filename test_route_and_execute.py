"""
Tests for route_and_execute.

Checks that the right items flow into and out of route_and_execute,
and that the returned cache matches what update_cache in state.py expects.

Run: python test_route_and_execute.py
"""

import sys
import pprint

from agent_tools import route_and_execute, WorkflowResult
from agent_tools import Intent, IntentResult

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

MOCK_PORTFOLIO = {
    "tickers": ["AAPL", "GOOGL", "MSFT"],
    "weights": [0.4, 0.3, 0.3],
    "investment_amount": 10000,
}

# Keys that update_cache in state.py expects
EXPECTED_CACHE_KEYS = {"returns_df", "metrics", "risk_level", "trend_forecast", "rag_context"}

# Keys that metrics dict should have (set by data_and_metrics)
EXPECTED_METRICS_KEYS = {"all_metrics", "metric_analysis"}

# Keys that risk_level dict should have (aligned to ExplanationContext)
EXPECTED_RISK_KEYS = {"label", "confidence"}

# Keys that trend_forecast dict should have
EXPECTED_FORECAST_KEYS = {"predicted_volatility", "predicted_direction", "confidence", "prob_up"}


def make_intent(primary, secondary=None, metrics=None, concept=None):
    return IntentResult(
        primary_intent=primary,
        confidence=0.95,
        reasoning="test",
        secondary_intent=secondary,
        extracted_metrics=metrics,
        extracted_concept=concept,
    )


def check_cache_keys(cache: dict, computed: bool, label: str):
    """Verify cache has the right top-level keys and internal structure."""
    print(f"\n  [cache check — {label}]")

    missing = EXPECTED_CACHE_KEYS - set(cache.keys())
    if missing:
        print(f"  WARN: cache missing keys: {missing}")
    else:
        print(f"  OK: all expected cache keys present")

    if computed:
        # metrics structure
        metrics = cache.get("metrics")
        if not isinstance(metrics, dict):
            print(f"  FAIL: cache['metrics'] is not a dict (got {type(metrics)})")
        else:
            missing_m = EXPECTED_METRICS_KEYS - set(metrics.keys())
            print(f"  {'OK' if not missing_m else 'FAIL'}: metrics keys: {set(metrics.keys())}")

        # risk_level structure
        risk = cache.get("risk_level")
        if not isinstance(risk, dict):
            print(f"  FAIL: cache['risk_level'] is not a dict (got {type(risk)})")
        else:
            missing_r = EXPECTED_RISK_KEYS - set(risk.keys())
            print(f"  {'OK' if not missing_r else 'FAIL'}: risk_level keys: {set(risk.keys())} | label={risk.get('label')} confidence={risk.get('confidence'):.4f}")

        # trend_forecast structure
        forecast = cache.get("trend_forecast")
        if not isinstance(forecast, dict):
            print(f"  FAIL: cache['trend_forecast'] is not a dict (got {type(forecast)})")
        else:
            missing_f = EXPECTED_FORECAST_KEYS - set(forecast.keys())
            print(f"  {'OK' if not missing_f else 'FAIL'}: trend_forecast keys: {set(forecast.keys())}")
            print(f"         direction={forecast.get('predicted_direction')} | prob_up={forecast.get('prob_up'):.4f} | confidence={forecast.get('confidence'):.4f}")


def check_result(result: WorkflowResult, expected_intent: Intent, expected_secondary=None):
    """Verify WorkflowResult fields."""
    print(f"\n  [result check]")
    print(f"  intent:           {result.intent}")
    print(f"  secondary_intent: {result.secondary_intent}")
    print(f"  content length:   {len(result.content)} chars")
    print(f"  content preview:  {result.content[:200]}...")

    assert isinstance(result, WorkflowResult),          "result is not a WorkflowResult"
    assert result.intent == expected_intent,            f"expected intent {expected_intent}, got {result.intent}"
    assert result.secondary_intent == expected_secondary, f"expected secondary {expected_secondary}, got {result.secondary_intent}"
    assert isinstance(result.content, str),             "content is not a string"
    assert result.content.strip(),                      "content is empty"
    assert isinstance(result.cache, dict),              "cache is not a dict"

    if expected_secondary:
        assert "---" in result.content and "Additionally" in result.content, \
            "secondary intent separator missing from content"
        print(f"  OK: secondary intent separator present in content")

    print(f"  OK: WorkflowResult structure valid")


# ---------------------------------------------------------------------------
# Test 1: First portfolio, full_analysis
#   - portfolio_changed=True, is_first_portfolio=True
#   - Expected: metrics + risk + LSTM computed from scratch
#   - No comparison block
# ---------------------------------------------------------------------------

def test_first_portfolio_full_analysis():
    print("\n" + "="*60)
    print("TEST 1: first portfolio, full_analysis (portfolio_changed=True)")
    print("="*60)

    intent = make_intent(Intent.FULL_ANALYSIS)
    print(f"  Input intent:     {intent.primary_intent}")
    print(f"  portfolio_changed: True | is_first_portfolio: True")
    print(f"  cache:            empty")

    result = route_and_execute(
        intent,
        user_query="Give me a full analysis of my portfolio",
        portfolio=MOCK_PORTFOLIO,
        is_first_portfolio=True,
        portfolio_changed=True,
        cache={},
    )

    check_result(result, Intent.FULL_ANALYSIS)
    check_cache_keys(result.cache, computed=True, label="first portfolio")

    assert "Previous Portfolio" not in result.content, \
        "Comparison block should NOT appear for first portfolio"
    print("  OK: no comparison block for first portfolio")

    return result.cache   # return for reuse in test 2


# ---------------------------------------------------------------------------
# Test 2: Same portfolio, full_analysis, cache warm
#   - portfolio_changed=False, pre-populated cache from test 1
#   - Expected: nothing recomputed, cache unchanged
# ---------------------------------------------------------------------------

def test_cache_hit_full_analysis(warm_cache: dict):
    print("\n" + "="*60)
    print("TEST 2: same portfolio, full_analysis, cache warm (portfolio_changed=False)")
    print("="*60)

    intent = make_intent(Intent.FULL_ANALYSIS)
    print(f"  Input intent:     {intent.primary_intent}")
    print(f"  portfolio_changed: False | is_first_portfolio: False")
    print(f"  cache:            pre-populated from test 1")

    result = route_and_execute(
        intent,
        user_query="Give me a full analysis of my portfolio",
        portfolio=MOCK_PORTFOLIO,
        is_first_portfolio=False,
        portfolio_changed=False,
        cache=warm_cache,
    )

    check_result(result, Intent.FULL_ANALYSIS)

    # Cache should be preserved, not wiped
    assert result.cache.get("metrics") is not None, "cache['metrics'] was wiped on cache hit"
    assert result.cache.get("risk_level") is not None, "cache['risk_level'] was wiped on cache hit"
    assert result.cache.get("trend_forecast") is not None, "cache['trend_forecast'] was wiped on cache hit"
    print("  OK: existing cache preserved (nothing recomputed)")


# ---------------------------------------------------------------------------
# Test 3: Portfolio changed, specific_metric (no LSTM needed)
#   - portfolio_changed=True, is_first_portfolio=False
#   - Expected: metrics + risk computed; LSTM NOT run
# ---------------------------------------------------------------------------

def test_portfolio_changed_specific_metric():
    print("\n" + "="*60)
    print("TEST 3: portfolio changed, specific_metric (LSTM should be skipped)")
    print("="*60)

    intent = make_intent(Intent.SPECIFIC_METRIC, metrics=["sharpe_ratio", "var_95"])
    print(f"  Input intent:      {intent.primary_intent}")
    print(f"  extracted_metrics: {intent.extracted_metrics}")
    print(f"  portfolio_changed: True | is_first_portfolio: False")

    result = route_and_execute(
        intent,
        user_query="What is my Sharpe ratio and max drawdown?",
        portfolio=MOCK_PORTFOLIO,
        is_first_portfolio=False,
        portfolio_changed=True,
        cache={},
    )

    check_result(result, Intent.SPECIFIC_METRIC)

    assert result.cache.get("metrics") is not None,   "metrics should have been computed"
    assert result.cache.get("trend_forecast") is None, "LSTM should NOT run for specific_metric"
    print("  OK: metrics computed, LSTM skipped")

    assert "Sharpe" in result.content or "sharpe" in result.content.lower(), \
        "Sharpe not found in content"
    assert "Value at Risk" in result.content or "var" in result.content.lower(), \
        "VaR not found in content"
    print("  OK: requested metrics present in content")


# ---------------------------------------------------------------------------
# Test 4: Concept explanation — no computation at all
#   - portfolio_changed=True but intent only needs RAG
#   - Expected: metrics/risk/LSTM all skipped
# ---------------------------------------------------------------------------

def test_concept_no_computation():
    print("\n" + "="*60)
    print("TEST 4: concept_explanation — no computation even when portfolio_changed=True")
    print("="*60)

    intent = make_intent(Intent.CONCEPT_EXPLANATION, concept="Sharpe ratio")
    print(f"  Input intent:      {intent.primary_intent}")
    print(f"  extracted_concept: {intent.extracted_concept}")
    print(f"  portfolio_changed: True (should still skip computation)")

    result = route_and_execute(
        intent,
        user_query="What is the Sharpe ratio?",
        portfolio=MOCK_PORTFOLIO,
        is_first_portfolio=True,
        portfolio_changed=True,
        cache={},
    )

    check_result(result, Intent.CONCEPT_EXPLANATION)

    assert result.cache.get("metrics") is None,       "metrics should NOT be computed for concept"
    assert result.cache.get("risk_level") is None,    "risk should NOT be computed for concept"
    assert result.cache.get("trend_forecast") is None, "LSTM should NOT run for concept"
    print("  OK: no computation performed")

    assert "Sharpe" in result.content or "sharpe" in result.content.lower(), \
        "concept name not in content"
    print("  OK: concept name present in content")


# ---------------------------------------------------------------------------
# Test 5: Dual intent — specific_metric (primary) + concept_explanation (secondary)
#   - Expected: metrics computed (for primary), content has separator + "Additionally"
# ---------------------------------------------------------------------------

def test_dual_intent_specific_metric_and_concept():
    print("\n" + "="*60)
    print("TEST 5: dual intent — specific_metric + concept_explanation")
    print("="*60)

    intent = make_intent(
        Intent.SPECIFIC_METRIC,
        secondary=Intent.CONCEPT_EXPLANATION,
        metrics=["max_drawdown"],
        concept="beta",
    )
    print(f"  primary:           {intent.primary_intent}")
    print(f"  secondary:         {intent.secondary_intent}")
    print(f"  extracted_metrics: {intent.extracted_metrics}")
    print(f"  extracted_concept: {intent.extracted_concept}")

    result = route_and_execute(
        intent,
        user_query="What is my max drawdown and what does beta mean?",
        portfolio=MOCK_PORTFOLIO,
        is_first_portfolio=True,
        portfolio_changed=True,
        cache={},
    )

    check_result(result, Intent.SPECIFIC_METRIC, expected_secondary=Intent.CONCEPT_EXPLANATION)

    assert result.cache.get("metrics") is not None, "metrics should be computed for specific_metric primary"
    assert result.cache.get("trend_forecast") is None, "LSTM not needed for this dual intent"
    print("  OK: metrics computed, LSTM skipped")

    print("  Content sections:")
    for i, section in enumerate(result.content.split("---")):
        print(f"    [{i}] {section[:120].strip()}...")


# ---------------------------------------------------------------------------
# Test 6: follow_up with prior history
#   - chat_history stored in cache
#   - Full Previous Response and Last 6 Turns sections present in content
#   - Existing cache (metrics/risk) preserved
# ---------------------------------------------------------------------------

PRIOR_ASSISTANT_RESPONSE = (
    "Your portfolio has a Medium risk level with annualised volatility of 18.3%. "
    "The Sharpe ratio is 0.82, indicating moderate risk-adjusted returns. "
    "AAPL contributes the most to overall risk at 52%."
)

FOLLOW_UP_HISTORY = [
    {"role": "user", "content": "Give me a full analysis of my portfolio"},
    {"role": "assistant", "content": PRIOR_ASSISTANT_RESPONSE},
    {"role": "user", "content": "Can you elaborate on what you said?"},
]


def test_follow_up_with_history(warm_cache: dict):
    print("\n" + "="*60)
    print("TEST 6: follow_up with prior history — chat_history cached, content labelled")
    print("="*60)

    intent = make_intent(Intent.FOLLOW_UP)
    query = "Can you elaborate on what you said?"
    print(f"  query:   \"{query}\"")
    print(f"  history: {len(FOLLOW_UP_HISTORY)} messages (last assistant: {PRIOR_ASSISTANT_RESPONSE[:60]}...)")

    result = route_and_execute(
        intent,
        user_query=query,
        portfolio=MOCK_PORTFOLIO,
        is_first_portfolio=False,
        portfolio_changed=False,
        recent_history=FOLLOW_UP_HISTORY,
        cache=warm_cache,
    )

    check_result(result, Intent.FOLLOW_UP)

    assert result.cache.get("chat_history") is not None, \
        "chat_history missing from cache for follow_up"
    assert len(result.cache["chat_history"]) == len(FOLLOW_UP_HISTORY), \
        f"chat_history length mismatch: expected {len(FOLLOW_UP_HISTORY)}, got {len(result.cache['chat_history'])}"
    print(f"  OK: chat_history stored ({len(result.cache['chat_history'])} turns)")

    assert "## Full Previous Response" in result.content, \
        "'## Full Previous Response' section missing from content"
    assert "## Last 6 Turns of Conversation" in result.content, \
        "'## Last 6 Turns of Conversation' section missing from content"
    print("  OK: content sections labelled correctly")

    assert "volatility" in result.content.lower() or "sharpe" in result.content.lower(), \
        "prior assistant response not surfaced in content"
    print("  OK: prior response content present")

    # Existing cache should be preserved
    assert result.cache.get("metrics") is not None,    "metrics wiped on follow_up"
    assert result.cache.get("risk_level") is not None, "risk_level wiped on follow_up"
    print("  OK: existing cache preserved")


# ---------------------------------------------------------------------------
# Test 7: follow_up with no history
#   - Expected: fallback message, no chat_history in cache
# ---------------------------------------------------------------------------

def test_follow_up_no_history():
    print("\n" + "="*60)
    print("TEST 7: follow_up with empty history — fallback message returned")
    print("="*60)

    intent = make_intent(Intent.FOLLOW_UP)
    query = "Can you explain what you just said?"
    print(f"  query:   \"{query}\"")
    print(f"  history: empty")

    result = route_and_execute(
        intent,
        user_query=query,
        portfolio=MOCK_PORTFOLIO,
        is_first_portfolio=False,
        portfolio_changed=False,
        recent_history=[],
        cache={},
    )

    check_result(result, Intent.FOLLOW_UP)
    assert "no earlier answer" in result.content.lower(), \
        "expected fallback message when history is empty"
    print("  OK: fallback message returned when no prior history")


# ---------------------------------------------------------------------------
# Test 8: follow_up with concept-signal query — RAG should trigger
#   - Query contains "what" → _CONCEPT_SIGNALS match
#   - Expected: rag_context populated in cache, Reference material in content
# ---------------------------------------------------------------------------

def test_follow_up_concept_rag():
    print("\n" + "="*60)
    print("TEST 8: follow_up with concept-signal query — RAG triggered")
    print("="*60)

    history = [
        {"role": "assistant", "content": "Your Sharpe ratio is 0.82."},
        {"role": "user", "content": "What does the Sharpe ratio mean exactly?"},
    ]
    intent = make_intent(Intent.FOLLOW_UP)
    query = "What does the Sharpe ratio mean exactly?"
    print(f"  query:   \"{query}\"")
    print(f"  concept signals present: {set(query.lower().split()) & {'what', 'mean', 'means'}}")

    result = route_and_execute(
        intent,
        user_query=query,
        portfolio=MOCK_PORTFOLIO,
        is_first_portfolio=False,
        portfolio_changed=False,
        recent_history=history,
        cache={},
    )

    check_result(result, Intent.FOLLOW_UP)
    assert "rag_context" in result.cache, \
        "rag_context key missing from cache — RAG should have been attempted for concept-signal query"
    print(f"  OK: rag_context key present in cache (value={'chunks returned' if result.cache['rag_context'] else 'no chunks found'})")
    if result.cache["rag_context"]:
        assert "Reference material" in result.content, \
            "RAG chunks returned but not surfaced in content"
        print("  OK: Reference material present in content")
    else:
        print("  NOTE: RAG returned no chunks for this query (knowledge base may not cover it)")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    passed = 0
    failed = 0

    tests = [
        "test_first_portfolio_full_analysis",
        "test_cache_hit_full_analysis",
        "test_portfolio_changed_specific_metric",
        "test_concept_no_computation",
        "test_dual_intent_specific_metric_and_concept",
        "test_follow_up_with_history",
        "test_follow_up_no_history",
        "test_follow_up_concept_rag",
    ]

    warm_cache = {}

    for name in tests:
        try:
            if name == "test_first_portfolio_full_analysis":
                warm_cache = test_first_portfolio_full_analysis()
            elif name == "test_cache_hit_full_analysis":
                test_cache_hit_full_analysis(warm_cache)
            elif name == "test_portfolio_changed_specific_metric":
                test_portfolio_changed_specific_metric()
            elif name == "test_concept_no_computation":
                test_concept_no_computation()
            elif name == "test_dual_intent_specific_metric_and_concept":
                test_dual_intent_specific_metric_and_concept()
            elif name == "test_follow_up_with_history":
                test_follow_up_with_history(warm_cache)
            elif name == "test_follow_up_no_history":
                test_follow_up_no_history()
            elif name == "test_follow_up_concept_rag":
                test_follow_up_concept_rag()

            print(f"\nPASS: {name}")
            passed += 1
        except Exception as e:
            import traceback
            print(f"\nFAIL: {name}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
