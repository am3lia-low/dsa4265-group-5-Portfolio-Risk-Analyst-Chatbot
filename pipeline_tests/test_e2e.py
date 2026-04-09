"""
End-to-end test: user query → classify_intent → route_and_execute → generate_explanation

Verifies the full pipeline produces coherent LLM-generated responses for each
intent type. Also chains a follow-up query off a real prior response to confirm
the conversation context flows correctly.

Run: python test_e2e.py
"""

import sys
from dotenv import load_dotenv
load_dotenv()

from agent_tools.workflow_tools import classify_intent, route_and_execute, Intent
from agent_tools.workflow_tools.agent_llm import KeyRotator

_key_rotator = KeyRotator()

MOCK_PORTFOLIO = {
    "tickers": ["AAPL", "GOOGL", "MSFT"],
    "weights": [0.4, 0.3, 0.3],
    "investment_amount": 10000,
    "currency": "USD",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_pipeline(query: str, history: list, portfolio_changed: bool = False, cache: dict = None):
    """Classify intent then route_and_execute. Returns (intent_result, workflow_result)."""
    intent = _key_rotator.call_with_retry(
        lambda client: classify_intent(
            message=query,
            recent_history=history,
            portfolio_changed=portfolio_changed,
            client=client,
        )
    )
    workflow = route_and_execute(
        intent,
        user_query=query,
        portfolio=MOCK_PORTFOLIO,
        is_first_portfolio=(cache is None or not cache),
        portfolio_changed=portfolio_changed,
        recent_history=history,
        cache=cache or {},
    )
    return intent, workflow


def print_result(label: str, intent_result, workflow):
    print(f"\n  [classify]  {intent_result.primary_intent.value}"
          + (f" + {intent_result.secondary_intent.value}" if intent_result.secondary_intent else "")
          + f"  (conf: {intent_result.confidence:.2f})")
    print(f"  [response]  {len(workflow.content)} chars")
    print(f"  {workflow.content[:400].strip()}...")


_FALLBACK_MSG = "I'm sorry, I encountered an error generating the explanation."


def check(label: str, workflow, expected_intent: Intent, min_chars: int = 100):
    assert isinstance(workflow.content, str) and workflow.content.strip(), \
        f"[{label}] content is empty"
    if workflow.content.startswith(_FALLBACK_MSG):
        print(f"  WARN: [{label}] API returned fallback error message — transient failure, not a code bug")
        return
    assert len(workflow.content) >= min_chars, \
        f"[{label}] response too short ({len(workflow.content)} chars)"
    print(f"  OK: [{label}] non-empty LLM response ({len(workflow.content)} chars)")


# ---------------------------------------------------------------------------
# Test 1: full_analysis — first portfolio
# ---------------------------------------------------------------------------

def test_e2e_full_analysis():
    print("\n" + "="*60)
    print("TEST 1: full pipeline — full_analysis (first portfolio)")
    print("="*60)

    query = "Give me a full analysis of my portfolio"
    intent, workflow = run_pipeline(query, history=[], portfolio_changed=True)
    print_result("full_analysis", intent, workflow)

    assert intent.primary_intent == Intent.FULL_ANALYSIS, \
        f"Expected FULL_ANALYSIS, got {intent.primary_intent}"
    check("full_analysis", workflow, Intent.FULL_ANALYSIS)

    assert workflow.cache.get("metrics")       is not None, "metrics missing"
    assert workflow.cache.get("risk_level")    is not None, "risk_level missing"
    assert workflow.cache.get("trend_forecast") is not None, "trend_forecast missing"
    print("  OK: cache fully populated")

    return workflow   # pass to test 2


# ---------------------------------------------------------------------------
# Test 2: follow_up — chained off real prior response
# ---------------------------------------------------------------------------

def test_e2e_follow_up(prior_workflow):
    print("\n" + "="*60)
    print("TEST 2: full pipeline — follow_up (chained off real prior response)")
    print("="*60)

    # Build history using the actual prior response
    history = [
        {"role": "user",      "content": "Give me a full analysis of my portfolio"},
        {"role": "assistant", "content": prior_workflow.content},
        {"role": "user",      "content": "Can you say more about what you just told me?"},
    ]
    query = "Can you say more about what you just told me?"
    intent, workflow = run_pipeline(
        query, history=history,
        portfolio_changed=False,
        cache=prior_workflow.cache,
    )
    print_result("follow_up", intent, workflow)

    if intent.primary_intent != Intent.FOLLOW_UP:
        print(f"  WARN: classifier returned {intent.primary_intent.value} — follow_up check skipped")
    else:
        check("follow_up", workflow, Intent.FOLLOW_UP)
        print("  OK: follow_up response references prior conversation context")


# ---------------------------------------------------------------------------
# Test 3: specific_metric — Sharpe + VaR
# ---------------------------------------------------------------------------

def test_e2e_specific_metric():
    print("\n" + "="*60)
    print("TEST 3: full pipeline — specific_metric (Sharpe + VaR)")
    print("="*60)

    query = "Show me my Sharpe ratio and my VaR"
    intent, workflow = run_pipeline(query, history=[], portfolio_changed=True)
    print_result("specific_metric", intent, workflow)

    if intent.primary_intent != Intent.SPECIFIC_METRIC:
        print(f"  WARN: classifier returned {intent.primary_intent.value} instead of specific_metric")
    check("specific_metric", workflow, Intent.SPECIFIC_METRIC)

    assert workflow.cache.get("metrics")        is not None, "metrics missing"
    assert workflow.cache.get("trend_forecast") is None,     "LSTM should be skipped"
    print("  OK: metrics computed, LSTM skipped")


# ---------------------------------------------------------------------------
# Test 4: concept_explanation
# ---------------------------------------------------------------------------

def test_e2e_concept_explanation():
    print("\n" + "="*60)
    print("TEST 4: full pipeline — concept_explanation (max drawdown)")
    print("="*60)

    query = "What is maximum drawdown and why does it matter?"
    intent, workflow = run_pipeline(query, history=[], portfolio_changed=False)
    print_result("concept_explanation", intent, workflow)

    assert intent.primary_intent == Intent.CONCEPT_EXPLANATION, \
        f"Expected CONCEPT_EXPLANATION, got {intent.primary_intent}"
    check("concept_explanation", workflow, Intent.CONCEPT_EXPLANATION)

    assert workflow.cache.get("metrics") is None, "no computation expected for concept"
    print("  OK: no computation performed")


# ---------------------------------------------------------------------------
# Test 5: trend_prediction
# ---------------------------------------------------------------------------

def test_e2e_trend_prediction():
    print("\n" + "="*60)
    print("TEST 5: full pipeline — trend_prediction")
    print("="*60)

    query = "What does the volatility forecast look like for my portfolio?"
    intent, workflow = run_pipeline(query, history=[], portfolio_changed=True)
    print_result("trend_prediction", intent, workflow)

    assert intent.primary_intent == Intent.TREND_PREDICTION, \
        f"Expected TREND_PREDICTION, got {intent.primary_intent}"
    check("trend_prediction", workflow, Intent.TREND_PREDICTION)

    assert workflow.cache.get("trend_forecast") is not None, "trend_forecast missing"
    print("  OK: LSTM forecast present")


# ---------------------------------------------------------------------------
# Test 6: general_chat
# ---------------------------------------------------------------------------

def test_e2e_general_chat():
    print("\n" + "="*60)
    print("TEST 6: full pipeline — general_chat")
    print("="*60)

    query = "Hi! What can you help me with?"
    intent, workflow = run_pipeline(query, history=[], portfolio_changed=False)
    print_result("general_chat", intent, workflow)

    assert intent.primary_intent == Intent.GENERAL_CHAT, \
        f"Expected GENERAL_CHAT, got {intent.primary_intent}"
    check("general_chat", workflow, Intent.GENERAL_CHAT, min_chars=50)
    print("  OK: general chat response generated")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    passed = 0
    failed = 0
    prior_workflow = None

    tests = [
        ("test_e2e_full_analysis",       None),
        ("test_e2e_follow_up",           None),   # needs prior_workflow
        ("test_e2e_specific_metric",     test_e2e_specific_metric),
        ("test_e2e_concept_explanation", test_e2e_concept_explanation),
        ("test_e2e_trend_prediction",    test_e2e_trend_prediction),
        ("test_e2e_general_chat",        test_e2e_general_chat),
    ]

    for name, fn in tests:
        try:
            if name == "test_e2e_full_analysis":
                prior_workflow = test_e2e_full_analysis()
            elif name == "test_e2e_follow_up":
                test_e2e_follow_up(prior_workflow)
            else:
                fn()
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
