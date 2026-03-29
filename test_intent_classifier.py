"""
Test suite for the intent classifier.
Calls the ACTUAL Gemini API — requires GEMINI_API_KEY env var.

Run: GEMINI_API_KEY=your_key python test_intent_classifier.py

Tests are grouped by category:
  1. Clear-cut intents (should always pass)
  2. Disambiguation edge cases
  3. Multi-intent messages
  4. Disguised intents
  5. Portfolio-state edge cases
  6. Override cases
  7. Ambiguous / stress tests
"""

import os
import sys
import time
from google import genai

from agent_tools import classify_intent, Intent

from dotenv import load_dotenv
load_dotenv()


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    print("ERROR: Set GEMINI_API_KEY environment variable")
    sys.exit(1)

CLIENT = genai.Client(api_key=API_KEY)

# Shared test fixtures
NO_PORTFOLIO = None
NO_HISTORY = []

# SAMPLE_PORTFOLIO = {
# "tickers": ["AAPL", "TSLA", "BND"],
# "weights": [0.5, 0.3, 0.2],
# "total_investment": 10000,
# }

SAMPLE_HISTORY = [
    {"role": "user", "content": "Analyze my portfolio"},
    {"role": "assistant", "content": "Your portfolio has a Sharpe ratio of 1.2, "
     "volatility of 18%, and a max drawdown of -12%. The NN classifies this as "
     "Medium risk with 78% confidence. Your biggest risk contributor is TSLA at 55%."},
]

# Rate limiting: Gemini Flash-Lite allows 15 RPM
# We add a small delay between tests to be safe
DELAY = 0.5  # seconds between API calls

# Counters
passed = 0
failed = 0
errors = []


# ---------------------------------------------------------------------------
# Test helper
# ---------------------------------------------------------------------------
def test(
    name: str,
    message: str,
    expected_primary: Intent,
    expected_secondary: Intent = None,
    history=NO_HISTORY,
    portfolio_changed=False,
    is_first_portfolio=False,
    check_entities: dict = None,
):
    """Run one test case against the live Gemini API."""
    global passed, failed, errors

    time.sleep(DELAY)

    try:
        result = classify_intent(
            message=message,
            recent_history=history,
            portfolio_changed=portfolio_changed,
            is_first_portfolio=is_first_portfolio,
            client=CLIENT,
        )
    except Exception as e:
        failed += 1
        errors.append(f"  CRASH  {name}: {e}")
        print(f"  [CRASH] {name}: CRASHED -- {e}")
        return

    # Check primary intent
    primary_ok = result.primary_intent == expected_primary

    # Check secondary intent (if expected)
    secondary_ok = True
    if expected_secondary is not None:
        secondary_ok = result.secondary_intent == expected_secondary

    # Check entities (if specified)
    entity_ok = True
    entity_notes = []
    if check_entities:
        if "metrics" in check_entities and check_entities["metrics"]:
            if not result.extracted_metrics:
                entity_ok = False
                entity_notes.append(f"expected metrics {check_entities['metrics']}, got None")
            else:
                for m in check_entities["metrics"]:
                    if m not in result.extracted_metrics:
                        entity_ok = False
                        entity_notes.append(f"missing metric '{m}' in {result.extracted_metrics}")

        if "tickers" in check_entities and check_entities["tickers"]:
            if not result.extracted_tickers:
                entity_ok = False
                entity_notes.append(f"expected tickers {check_entities['tickers']}, got None")
            else:
                for t in check_entities["tickers"]:
                    if t not in result.extracted_tickers:
                        entity_ok = False
                        entity_notes.append(f"missing ticker '{t}' in {result.extracted_tickers}")

        if "concept" in check_entities and check_entities["concept"]:
            if not result.extracted_concept:
                entity_ok = False
                entity_notes.append(f"expected concept '{check_entities['concept']}', got None")

    all_ok = primary_ok and secondary_ok and entity_ok

    if all_ok:
        passed += 1
        symbol = "[PASS]"
    else:
        failed += 1
        symbol = "[FAIL]"

    # Build output
    primary_str = result.primary_intent.value
    if not primary_ok:
        primary_str += f" (expected {expected_primary.value})"

    secondary_str = ""
    if result.secondary_intent:
        secondary_str = f" + {result.secondary_intent.value}"
    if expected_secondary and not secondary_ok:
        secondary_str += f" (expected secondary: {expected_secondary.value})"

    print(f"  {symbol} {name}")
    print(f"    -> {primary_str}{secondary_str} (conf={result.confidence:.2f})")
    print(f"    -> {result.reasoning}")

    if result.extracted_metrics:
        print(f"    -> metrics: {result.extracted_metrics}")
    if result.extracted_tickers:
        print(f"    -> tickers: {result.extracted_tickers}")
    if result.extracted_concept:
        print(f"    -> concept: {result.extracted_concept}")
    if entity_notes:
        print(f"    -> ENTITY ISSUES: {'; '.join(entity_notes)}")

    if not all_ok:
        errors.append(f"  {symbol} {name}: got {primary_str}{secondary_str}")

    print()


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def run_tests():
    print("=" * 70)
    print("INTENT CLASSIFIER TESTS (live Gemini API)")
    print("=" * 70)

    # =====================================================================
    # 1. CLEAR-CUT INTENTS
    # =====================================================================
    print("\n--- 1. Clear-cut intents ---\n")

    test("General: simple greeting",
         "Hi", Intent.GENERAL_CHAT)

    test("General: thanks",
         "Thanks!", Intent.GENERAL_CHAT)

    test("General: capability question",
         "What can you do?", Intent.GENERAL_CHAT)

    test("Full analysis: explicit request",
         "Analyze my portfolio", Intent.FULL_ANALYSIS)

    test("Full analysis: first form submission",
         "", Intent.FULL_ANALYSIS,
         portfolio_changed=True, is_first_portfolio=True)

    test("Full analysis: risk question",
         "Is my portfolio risky?", Intent.FULL_ANALYSIS)

    test("Rebalance: what-if",
         "What if I sell Tesla?", Intent.REBALANCE,
         check_entities={"tickers": ["TSLA"]})

    test("Rebalance: add asset",
         "Should I add bonds to my portfolio?", Intent.REBALANCE)

    test("Rebalance: form update",
         "", Intent.REBALANCE,
         portfolio_changed=True, is_first_portfolio=False)

    test("Specific metric: Sharpe",
         "What's my Sharpe ratio?", Intent.SPECIFIC_METRIC,
         check_entities={"metrics": ["sharpe_ratio"]})

    test("Specific metric: VaR",
         "Show me the VaR", Intent.SPECIFIC_METRIC,
         check_entities={"metrics": ["var_95"]})

    test("Concept: what is VaR",
         "What is Value at Risk?", Intent.CONCEPT_EXPLANATION,
         check_entities={"concept": "value at risk"})

    test("Concept: explain beta",
         "How does beta work?", Intent.CONCEPT_EXPLANATION)

    test("Trend: future volatility",
         "Predict future volatility", Intent.TREND_PREDICTION)

    test("Trend: risk outlook",
         "Will my risk increase?", Intent.TREND_PREDICTION)

    test("Follow-up: elaborate",
         "Can you elaborate on that?", Intent.FOLLOW_UP,
         history=SAMPLE_HISTORY)

    test("Follow-up: why",
         "Why did you say that?", Intent.FOLLOW_UP,
         history=SAMPLE_HISTORY)

    # =====================================================================
    # 2. DISAMBIGUATION EDGE CASES
    # =====================================================================
    print("\n--- 2. Disambiguation edge cases ---\n")

    test("specific_metric vs concept: 'what is MY sharpe' -> specific",
         "What is my Sharpe ratio?", Intent.SPECIFIC_METRIC)

    test("specific_metric vs concept: 'what IS sharpe ratio' -> concept",
         "What is the Sharpe ratio?", Intent.CONCEPT_EXPLANATION)

    test("specific_metric vs concept: 'tell me more about skewness' -> concept",
         "Tell me more about skewness", Intent.CONCEPT_EXPLANATION)

    # =====================================================================
    # 3. MULTI-INTENT MESSAGES
    # =====================================================================
    print("\n--- 3. Multi-intent messages ---\n")

    test("Multi: analyze + explain concept",
         "Analyze my portfolio and explain what VaR means",
         Intent.FULL_ANALYSIS,
         expected_secondary=Intent.CONCEPT_EXPLANATION)

    test("Multi: metric + is it good",
         "What's my Sharpe ratio and is that good?",
         Intent.SPECIFIC_METRIC,
         expected_secondary=Intent.CONCEPT_EXPLANATION)

    test("Multi: rebalance + explain concept",
         "Compare adding bonds and tell me about diversification",
         Intent.REBALANCE,
         expected_secondary=Intent.CONCEPT_EXPLANATION)

    # =====================================================================
    # 4. DISGUISED INTENTS
    # =====================================================================
    print("\n--- 4. Disguised intents ---\n")

    test("Disguised: looks like follow-up -> concept",
         "What did you mean by skewness?", Intent.CONCEPT_EXPLANATION,
         history=SAMPLE_HISTORY)

    test("Disguised: thanks + metric -> specific_metric",
         "Thanks! What's my beta though?", Intent.SPECIFIC_METRIC,
         history=SAMPLE_HISTORY)

    test("Disguised: greeting + rebalance",
         "Hi can you tell me if I should sell Tesla?", Intent.REBALANCE)

    test("Disguised: greeting + analysis",
         "Hey! Analyze my portfolio please", Intent.FULL_ANALYSIS)

    # =====================================================================
    # 5. PORTFOLIO-STATE EDGE CASES
    # =====================================================================
    print("\n--- 5. Portfolio-state edge cases ---\n")

    test("No portfolio: metric question -> concept",
         "What's my Sharpe ratio?", Intent.CONCEPT_EXPLANATION)

    test("No portfolio: rebalance question -> full_analysis",
         "What if I add bonds?", Intent.FULL_ANALYSIS)

    test("No portfolio: trend question -> general_chat",
         "Will my risk increase?", Intent.GENERAL_CHAT)

    # =====================================================================
    # 6. OVERRIDE CASES
    # =====================================================================
    print("\n--- 6. Override cases ---\n")

    test("Override: form update + 'analyze fresh' -> full_analysis",
         "Analyze this new portfolio from scratch",
         Intent.FULL_ANALYSIS,
         portfolio_changed=True, is_first_portfolio=False)

    test("Override: form update + no message -> rebalance (default)",
         "", Intent.REBALANCE,
         portfolio_changed=True, is_first_portfolio=False)

    # =====================================================================
    # 7. AMBIGUOUS / STRESS TESTS
    # =====================================================================
    print("\n--- 7. Ambiguous / stress tests ---\n")

    test("Ambiguous: 'How's my risk?'",
         "How's my risk?", Intent.FULL_ANALYSIS)

    test("Ambiguous: 'Tell me about my volatility'",
         "Tell me about my volatility", Intent.SPECIFIC_METRIC)

    test("Ambiguous: 'What about bonds?'",
         "What about bonds?", Intent.REBALANCE,
         history=SAMPLE_HISTORY)

    test("Ambiguous: 'Is this good?'",
         "Is this good?", Intent.FOLLOW_UP,
         history=SAMPLE_HISTORY)

    test("Ambiguous: single word 'Risk'",
         "Risk", Intent.FULL_ANALYSIS)

    test("Ambiguous: 'Help me understand my portfolio better'",
         "Help me understand my portfolio better", Intent.FULL_ANALYSIS)

    # =====================================================================
    # 8. ENTITY EXTRACTION STRESS TESTS
    # =====================================================================
    print("\n--- 8. Entity extraction stress tests ---\n")

    test("Entity: vague downside -> should infer var/cvar/sortino",
         "How's my downside looking?", Intent.SPECIFIC_METRIC,
         check_entities={"metrics": ["var_95"]})

    test("Entity: vague performance -> should infer sharpe/sortino",
         "How's my portfolio performing?", Intent.SPECIFIC_METRIC,
         check_entities={"metrics": ["sharpe_ratio"]})

    test("Entity: explicit metric extraction",
         "What's my max drawdown and beta?", Intent.SPECIFIC_METRIC,
         check_entities={"metrics": ["max_drawdown", "beta"]})

    test("Entity: concept should extract concept name",
         "Explain what conditional value at risk means", Intent.CONCEPT_EXPLANATION,
         check_entities={"concept": "conditional value at risk"})

    test("Entity: rebalance should extract ticker",
         "What if I replace Tesla with Google?", Intent.REBALANCE,
         check_entities={"tickers": ["TSLA"]})

    test("Entity: vague concentration -> should infer hhi/correlation",
         "Is my portfolio too concentrated?", Intent.SPECIFIC_METRIC,
         check_entities={"metrics": ["hhi_concentration"]})

    # =====================================================================
    # SUMMARY
    # =====================================================================
    print("=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 70)

    if errors:
        print("\nFailed tests:")
        for e in errors:
            print(e)

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
