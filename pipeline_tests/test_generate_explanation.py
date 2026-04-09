"""
Tests for generate_explanation.

Verifies that generate_explanation returns a coherent non-empty string for
each intent type, given realistic mock ExplanationContext inputs.
No live data fetching — only the Gemini API call is made.

Run: python test_generate_explanation.py
"""

import sys
from dotenv import load_dotenv
load_dotenv()

from agent_tools.workflow_tools.agent_llm import KeyRotator, generate_explanation, Intent

_key_rotator = KeyRotator()

# ---------------------------------------------------------------------------
# Shared mock data
# ---------------------------------------------------------------------------

MOCK_PORTFOLIO = {
    "tickers": ["AAPL", "GOOGL", "MSFT"],
    "weights": [0.4, 0.3, 0.3],
    "investment_amount": 10000,
    "currency": "USD",
}

MOCK_METRICS = {
    "portfolio_volatility": 0.24,
    "var_95":               -0.022,
    "cvar_95":              -0.031,
    "max_drawdown":         -0.346,
    "sharpe_ratio":         0.60,
    "sortino_ratio":        0.84,
    "beta":                 1.18,
    "hhi_concentration":    0.34,
    "avg_pairwise_correlation": 0.71,
    "vol_of_vol":           1.42,
    "skewness":             -0.21,
    "excess_kurtosis":      0.95,
}

MOCK_RISK_CONTRIBUTIONS = {
    "AAPL":  0.52,
    "GOOGL": 0.26,
    "MSFT":  0.22,
}

MOCK_METRIC_BENCHMARKS = {
    "portfolio_volatility": {"value": 0.24, "label": "high",   "comment": "Above 20% — high volatility"},
    "sharpe_ratio":         {"value": 0.60, "label": "poor",   "comment": "Below 0.8 — poor risk-adjusted return"},
    "max_drawdown":         {"value": -0.346, "label": "high", "comment": "More than 25% — severe historical loss"},
    "beta":                 {"value": 1.18, "label": "high",   "comment": "Above 1.1 — amplifies market moves"},
}

MOCK_RISK_LEVEL = {"label": "Medium", "confidence": 0.53}

MOCK_TREND_FORECAST = {
    "predicted_direction":  "Up",
    "predicted_volatility": 0.2587,
    "prob_up":              0.699,
    "confidence":           0.398,
}

MOCK_PREV_METRICS = {
    "portfolio_volatility": 0.31,
    "var_95":               -0.029,
    "sharpe_ratio":         0.42,
    "max_drawdown":         -0.41,
    "beta":                 1.35,
}

MOCK_PREV_CONTRIBUTIONS = {"AAPL": 0.61, "GOOGL": 0.22, "MSFT": 0.17}
MOCK_PREV_RISK_LEVEL    = {"label": "High", "confidence": 0.78}

MOCK_RAG_CONTEXT = (
    "**Reference material**\n"
    "1. (risk_kb) The Sharpe ratio measures excess return per unit of total risk. "
    "Values above 1.0 are considered good; above 2.0 are excellent.\n"
    "2. (risk_kb) Maximum drawdown captures the worst peak-to-trough decline. "
    "It is a key measure of downside risk and investor pain."
)

MOCK_CHAT_HISTORY = [
    {"role": "user",      "content": "Give me a full analysis of my portfolio"},
    {"role": "assistant", "content": "Your portfolio has Medium risk with annualised volatility of 24%. "
                                     "The Sharpe ratio is 0.60, indicating poor risk-adjusted returns. "
                                     "AAPL drives 52% of portfolio risk."},
    {"role": "user",      "content": "Can you say more about what you just told me?"},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_explanation(ctx: dict) -> str:
    return _key_rotator.call_with_retry(
        lambda client: generate_explanation(ctx, client=client)
    )


def check_explanation(text: str, label: str, keywords: list[str] = None):
    assert isinstance(text, str), f"[{label}] response is not a string"
    assert text.strip(),          f"[{label}] response is empty"
    print(f"\n  [{label}] response ({len(text)} chars):")
    print(f"  {text[:300].strip()}...")
    if keywords:
        for kw in keywords:
            if kw.lower() not in text.lower():
                print(f"  WARN: expected keyword '{kw}' not found in response")
    print(f"  OK: non-empty string returned")


# ---------------------------------------------------------------------------
# Test 1: full_analysis — first portfolio, no comparison
# ---------------------------------------------------------------------------

def test_full_analysis():
    print("\n" + "="*60)
    print("TEST 1: generate_explanation — full_analysis (first portfolio)")
    print("="*60)

    ctx = {
        "intent":               Intent.FULL_ANALYSIS,
        "user_query":           "Give me a full analysis of my portfolio",
        "portfolio":            MOCK_PORTFOLIO,
        "metrics":              MOCK_METRICS,
        "risk_contributions":   MOCK_RISK_CONTRIBUTIONS,
        "metric_benchmarks":    MOCK_METRIC_BENCHMARKS,
        "risk_level":           MOCK_RISK_LEVEL,
        "trend_forecast":       MOCK_TREND_FORECAST,
        "portfolio_changed":    False,
        "educational_context":  MOCK_RAG_CONTEXT,
        "chat_history":         MOCK_CHAT_HISTORY[:1],
    }
    text = run_explanation(ctx)
    check_explanation(text, "full_analysis", keywords=["risk", "sharpe", "volatility"])


# ---------------------------------------------------------------------------
# Test 2: full_analysis — portfolio changed, comparison mode
# ---------------------------------------------------------------------------

def test_full_analysis_comparison():
    print("\n" + "="*60)
    print("TEST 2: generate_explanation — full_analysis (portfolio changed, comparison)")
    print("="*60)

    ctx = {
        "intent":                 Intent.FULL_ANALYSIS,
        "user_query":             "I removed Tesla — how does my risk look now?",
        "portfolio":              MOCK_PORTFOLIO,
        "metrics":                MOCK_METRICS,
        "risk_contributions":     MOCK_RISK_CONTRIBUTIONS,
        "metric_benchmarks":      MOCK_METRIC_BENCHMARKS,
        "risk_level":             MOCK_RISK_LEVEL,
        "trend_forecast":         MOCK_TREND_FORECAST,
        "portfolio_changed":      True,
        "previous_metrics":       MOCK_PREV_METRICS,
        "previous_contributions": MOCK_PREV_CONTRIBUTIONS,
        "previous_risk_level":    MOCK_PREV_RISK_LEVEL,
        "educational_context":    MOCK_RAG_CONTEXT,
    }
    text = run_explanation(ctx)
    check_explanation(text, "full_analysis_comparison", keywords=["improved", "risk"])


# ---------------------------------------------------------------------------
# Test 3: specific_metric — sharpe + max_drawdown
# ---------------------------------------------------------------------------

def test_specific_metric():
    print("\n" + "="*60)
    print("TEST 3: generate_explanation — specific_metric (sharpe + max_drawdown)")
    print("="*60)

    ctx = {
        "intent":              Intent.SPECIFIC_METRIC,
        "user_query":          "What is my Sharpe ratio and max drawdown?",
        "portfolio":           MOCK_PORTFOLIO,
        "metrics":             MOCK_METRICS,
        "metric_benchmarks":   MOCK_METRIC_BENCHMARKS,
        "requested_metrics":   ["sharpe_ratio", "max_drawdown"],
        "educational_context": MOCK_RAG_CONTEXT,
    }
    text = run_explanation(ctx)
    check_explanation(text, "specific_metric", keywords=["sharpe", "drawdown"])


# ---------------------------------------------------------------------------
# Test 4: concept_explanation — Value at Risk
# ---------------------------------------------------------------------------

def test_concept_explanation():
    print("\n" + "="*60)
    print("TEST 4: generate_explanation — concept_explanation (Value at Risk)")
    print("="*60)

    ctx = {
        "intent":              Intent.CONCEPT_EXPLANATION,
        "user_query":          "What is Value at Risk?",
        "portfolio":           MOCK_PORTFOLIO,
        "concept_name":        "Value at Risk",
        "educational_context": MOCK_RAG_CONTEXT,
    }
    text = run_explanation(ctx)
    check_explanation(text, "concept_explanation", keywords=["value at risk", "var"])


# ---------------------------------------------------------------------------
# Test 5: trend_prediction
# ---------------------------------------------------------------------------

def test_trend_prediction():
    print("\n" + "="*60)
    print("TEST 5: generate_explanation — trend_prediction")
    print("="*60)

    ctx = {
        "intent":         Intent.TREND_PREDICTION,
        "user_query":     "What will my portfolio volatility look like in the next 60 days?",
        "portfolio":      MOCK_PORTFOLIO,
        "metrics":        MOCK_METRICS,
        "trend_forecast": MOCK_TREND_FORECAST,
    }
    text = run_explanation(ctx)
    check_explanation(text, "trend_prediction", keywords=["volatility", "forecast", "confidence"])


# ---------------------------------------------------------------------------
# Test 6: follow_up — with prior chat history
# ---------------------------------------------------------------------------

def test_follow_up():
    print("\n" + "="*60)
    print("TEST 6: generate_explanation — follow_up (with chat history)")
    print("="*60)

    ctx = {
        "intent":       Intent.FOLLOW_UP,
        "user_query":   "Can you say more about what you just told me?",
        "portfolio":    MOCK_PORTFOLIO,
        "metrics":      MOCK_METRICS,
        "risk_level":   MOCK_RISK_LEVEL,
        "chat_history": MOCK_CHAT_HISTORY,
    }
    text = run_explanation(ctx)
    check_explanation(text, "follow_up", keywords=["sharpe", "aapl", "risk"])


# ---------------------------------------------------------------------------
# Test 7: general_chat
# ---------------------------------------------------------------------------

def test_general_chat():
    print("\n" + "="*60)
    print("TEST 7: generate_explanation — general_chat")
    print("="*60)

    ctx = {
        "intent":     Intent.GENERAL_CHAT,
        "user_query": "Hello! What can you help me with?",
        "portfolio":  MOCK_PORTFOLIO,
    }
    text = run_explanation(ctx)
    check_explanation(text, "general_chat", keywords=["portfolio", "analysis"])


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    passed = 0
    failed = 0

    tests = [
        ("test_full_analysis",            test_full_analysis),
        ("test_full_analysis_comparison", test_full_analysis_comparison),
        ("test_specific_metric",          test_specific_metric),
        ("test_concept_explanation",      test_concept_explanation),
        ("test_trend_prediction",         test_trend_prediction),
        ("test_follow_up",                test_follow_up),
        ("test_general_chat",             test_general_chat),
    ]

    for name, fn in tests:
        try:
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
