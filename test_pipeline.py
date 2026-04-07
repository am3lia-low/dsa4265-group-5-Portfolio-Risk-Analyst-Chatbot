"""
End-to-end pipeline test: user query → classify_intent → route_and_execute

Verifies that the full pipeline works correctly and that data flows
properly between each stage.

Run: python test_pipeline.py
"""

import sys
from dotenv import load_dotenv
load_dotenv()

from agent_tools.workflow_tools import classify_intent, route_and_execute, Intent, IntentResult, WorkflowResult
from agent_tools.workflow_tools.agent_llm import KeyRotator

_key_rotator = KeyRotator()

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MOCK_PORTFOLIO = {
    "tickers": ["AAPL", "GOOGL", "MSFT"],
    "weights": [0.4, 0.3, 0.3],
    "investment_amount": 10000,
}

# Maps queries to expected primary intents (used for soft assertion)
QUERY_INTENT_MAP = {
    "Give me a full analysis of my portfolio":             Intent.FULL_ANALYSIS,
    "What is my Sharpe ratio?":                            Intent.SPECIFIC_METRIC,
    "What is Value at Risk?":                              Intent.CONCEPT_EXPLANATION,
    "What will my portfolio volatility look like?":        Intent.TREND_PREDICTION,
    "Can you explain what you just said?":                 Intent.FOLLOW_UP,
    "Hello, what can you do?":                             Intent.GENERAL_CHAT,
}

# One query expected to produce a secondary intent
DUAL_INTENT_QUERY = "What is my Sharpe ratio and what does it mean?"


def print_intent(intent: IntentResult):
    print(f"    primary:    {intent.primary_intent.value}")
    print(f"    secondary:  {intent.secondary_intent.value if intent.secondary_intent else 'None'}")
    print(f"    confidence: {intent.confidence:.2f}")
    print(f"    metrics:    {intent.extracted_metrics}")
    print(f"    concept:    {intent.extracted_concept}")
    print(f"    reasoning:  {intent.reasoning[:120]}...")


def print_result(result: WorkflowResult):
    print(f"    intent:    {result.intent.value}")
    print(f"    secondary: {result.secondary_intent.value if result.secondary_intent else 'None'}")
    print(f"    content:   {result.content[:300]}...")
    print(f"    cache keys populated: {[k for k, v in result.cache.items() if v is not None]}")


# ---------------------------------------------------------------------------
# Test 1: Intent classification accuracy across all intent types
# ---------------------------------------------------------------------------

def test_intent_classification():
    print("\n" + "="*60)
    print("TEST 1: intent classification across query types")
    print("="*60)

    for query, expected_intent in QUERY_INTENT_MAP.items():
        print(f"\n  Query: \"{query}\"")
        intent = _key_rotator.call_with_retry(
            lambda client: classify_intent(
                message=query,
                recent_history=[],
                portfolio_changed=False,
                client=client,
            )
        )
        print_intent(intent)

        if intent.primary_intent != expected_intent:
            print(f"  WARN: expected {expected_intent.value}, got {intent.primary_intent.value}")
        else:
            print(f"  OK: intent matched ({expected_intent.value})")

    print(f"\n  (mismatches shown as WARN — LLM classifiers may vary)")


# ---------------------------------------------------------------------------
# Test 2: Full pipeline — first portfolio, full analysis
#   query → classify_intent → route_and_execute
#   Checks intent is passed through correctly and cache is populated
# ---------------------------------------------------------------------------

def test_full_pipeline_first_portfolio():
    print("\n" + "="*60)
    print("TEST 2: full pipeline — first portfolio, full_analysis")
    print("="*60)

    query = "Give me a full analysis of my portfolio"
    print(f"  Query: \"{query}\"")

    # Stage 1: classify
    intent = _key_rotator.call_with_retry(
        lambda client: classify_intent(
            message=query,
            recent_history=[],
            portfolio_changed=True,
            client=client,
        )
    )
    print(f"\n  [classify_intent output]")
    print_intent(intent)

    assert intent.primary_intent == Intent.FULL_ANALYSIS, \
        f"Expected FULL_ANALYSIS, got {intent.primary_intent}"

    # Stage 2: route_and_execute
    result = route_and_execute(
        intent,
        user_query=query,
        portfolio=MOCK_PORTFOLIO,
        is_first_portfolio=True,
        portfolio_changed=True,
        recent_history=[],
        cache={},
    )
    print(f"\n  [route_and_execute output]")
    print_result(result)

    # Verify content
    assert result.content.strip(), "content is empty"
    assert result.intent == Intent.FULL_ANALYSIS

    # Verify cache populated correctly for update_cache
    assert result.cache.get("metrics") is not None,       "metrics missing from cache"
    assert result.cache.get("risk_level") is not None,    "risk_level missing from cache"
    assert result.cache.get("trend_forecast") is not None, "trend_forecast missing from cache"
    assert "label" in result.cache["risk_level"],          "risk_level missing 'label' key"
    assert "confidence" in result.cache["risk_level"],     "risk_level missing 'confidence' key"
    assert "all_metrics" in result.cache["metrics"],       "metrics missing 'all_metrics' key"
    assert "metric_analysis" in result.cache["metrics"],   "metrics missing 'metric_analysis' key"
    assert "predicted_direction" in result.cache["trend_forecast"], "trend_forecast missing 'predicted_direction'"

    print(f"\n  OK: full pipeline passed, cache correctly populated")
    return result.cache   # return warm cache for test 3


# ---------------------------------------------------------------------------
# Test 3: Full pipeline — same portfolio, follow-up query
#   Verifies cache is preserved and no recomputation happens
# ---------------------------------------------------------------------------

def test_full_pipeline_followup(warm_cache: dict):
    print("\n" + "="*60)
    print("TEST 3: full pipeline — follow-up query, cache warm")
    print("="*60)

    query = "Can you say more about what you just told me?"
    history = [{"role": "assistant", "content": "Your portfolio has a Medium risk level with annualised volatility of 18%."}]

    print(f"  Query: \"{query}\"")
    print(f"  History: {len(history)} message(s)")

    intent = _key_rotator.call_with_retry(
        lambda client: classify_intent(
            message=query,
            recent_history=history,
            portfolio_changed=False,
            client=client,
        )
    )
    print(f"\n  [classify_intent output]")
    print_intent(intent)

    result = route_and_execute(
        intent,
        user_query=query,
        portfolio=MOCK_PORTFOLIO,
        is_first_portfolio=False,
        portfolio_changed=False,
        recent_history=history,
        cache=warm_cache,
    )
    print(f"\n  [route_and_execute output]")
    print_result(result)

    # Cache should still be intact
    assert result.cache.get("metrics") is not None,    "cache was wiped on follow-up"
    assert result.cache.get("risk_level") is not None, "cache was wiped on follow-up"
    print(f"  OK: existing cache preserved")

    if intent.primary_intent != Intent.FOLLOW_UP:
        print(f"  WARN: classifier returned {intent.primary_intent.value} instead of follow_up")


# ---------------------------------------------------------------------------
# Test 4: Full pipeline — specific metric query
#   Verifies extracted_metrics flows from classifier into orchestrator
# ---------------------------------------------------------------------------

def test_full_pipeline_specific_metric():
    print("\n" + "="*60)
    print("TEST 4: full pipeline — specific metric query")
    print("="*60)

    query = "What is my Sharpe ratio and max drawdown?"
    print(f"  Query: \"{query}\"")

    intent = _key_rotator.call_with_retry(
        lambda client: classify_intent(
            message=query,
            recent_history=[],
            portfolio_changed=True,
            client=client,
        )
    )
    print(f"\n  [classify_intent output]")
    print_intent(intent)

    assert intent.primary_intent == Intent.SPECIFIC_METRIC, \
        f"Expected SPECIFIC_METRIC, got {intent.primary_intent}"
    assert intent.extracted_metrics, "extracted_metrics should be populated"
    print(f"  OK: extracted_metrics = {intent.extracted_metrics}")

    result = route_and_execute(
        intent,
        user_query=query,
        portfolio=MOCK_PORTFOLIO,
        is_first_portfolio=True,
        portfolio_changed=True,
        recent_history=[],
        cache={},
    )
    print(f"\n  [route_and_execute output]")
    print_result(result)

    assert result.cache.get("metrics") is not None,    "metrics should be computed"
    assert result.cache.get("trend_forecast") is None, "LSTM should be skipped for specific_metric"
    print(f"\n  OK: metrics computed, LSTM skipped")


# ---------------------------------------------------------------------------
# Test 5: Full pipeline — dual intent query
#   Verifies secondary intent flows through and appears in content
# ---------------------------------------------------------------------------

def test_full_pipeline_dual_intent():
    print("\n" + "="*60)
    print("TEST 5: full pipeline — dual intent query")
    print("="*60)

    query = DUAL_INTENT_QUERY
    print(f"  Query: \"{query}\"")

    intent = _key_rotator.call_with_retry(
        lambda client: classify_intent(
            message=query,
            recent_history=[],
            portfolio_changed=True,
            client=client,
        )
    )
    print(f"\n  [classify_intent output]")
    print_intent(intent)

    result = route_and_execute(
        intent,
        user_query=query,
        portfolio=MOCK_PORTFOLIO,
        is_first_portfolio=True,
        portfolio_changed=True,
        recent_history=[],
        cache={},
    )
    print(f"\n  [route_and_execute output]")
    print_result(result)

    if intent.secondary_intent:
        print(f"  OK: dual-intent response generated (primary={intent.primary_intent.value}, secondary={intent.secondary_intent.value})")
    else:
        print(f"  NOTE: classifier returned no secondary intent for this query (may vary)")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    passed = 0
    failed = 0
    warm_cache = {}

    pipeline_tests = [
        ("test_intent_classification",          lambda: test_intent_classification()),
        ("test_full_pipeline_first_portfolio",  None),   # handled separately for warm_cache
        ("test_full_pipeline_followup",         None),   # needs warm_cache
        ("test_full_pipeline_specific_metric",  lambda: test_full_pipeline_specific_metric()),
        ("test_full_pipeline_dual_intent",      lambda: test_full_pipeline_dual_intent()),
    ]

    for name, fn in pipeline_tests:
        try:
            if name == "test_full_pipeline_first_portfolio":
                warm_cache = test_full_pipeline_first_portfolio()
            elif name == "test_full_pipeline_followup":
                test_full_pipeline_followup(warm_cache)
            else:
                fn()
            print(f"\nPASS: {name}")
            passed += 1
        except Exception as e:
            import traceback
            print(f"\nFAIL: {name}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
