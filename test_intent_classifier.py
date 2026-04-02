"""
Test suite for the intent classifier.
Calls the ACTUAL Gemini API — requires GEMINI_API_KEY1 to GEMINI_API_KEY6 in .env

Run: python test_intent_classifier.py
     (loads keys from .env file automatically)

Assumptions:
  - Portfolio is always present (UI enforces this)
  - Rebalance is merged into full_analysis
  - Ticker extraction removed (UI handles portfolio changes)
  - 6 intents: full_analysis, specific_metric, concept_explanation,
    trend_prediction, follow_up, general_chat
"""

import os
import sys
import time
from google import genai

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # .env vars must be set manually

from agent_tools import classify_intent, Intent, IntentResult


# ---------------------------------------------------------------------------
# Key rotation setup
# ---------------------------------------------------------------------------
class KeyRotator:
    """Rotates across multiple Gemini API keys when rate limits are hit.
    
    Sticks with the SAME key until a 429 error occurs, then switches
    to the next key and retries the failed request.
    
    Gemini Flash-Lite free tier: 15 RPM per key.
    With 6 keys: effectively 90 RPM.
    """

    def __init__(self):
        # Load keys from env: GEMINI_API_KEY1 through GEMINI_API_KEY6
        self.keys = []
        for i in range(1, 7):
            key = os.environ.get(f"GEMINI_API_KEY{i}")
            if key:
                self.keys.append(key)

        # Fallback: try single GEMINI_API_KEY or GOOGLE_API_KEY
        if not self.keys:
            single = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if single:
                self.keys.append(single)

        if not self.keys:
            print("ERROR: No API keys found.")
            print("Set GEMINI_API_KEY1 through GEMINI_API_KEY6 in .env file,")
            print("or set a single GEMINI_API_KEY environment variable.")
            sys.exit(1)

        self.current_index = 0
        self.clients = [genai.Client(api_key=k) for k in self.keys]
        print(f"Loaded {len(self.keys)} API key(s). Using key 1 until rate limited.")

    @property
    def current_client(self) -> genai.Client:
        return self.clients[self.current_index]

    def rotate(self):
        """Move to the next key. Only called when current key hits 429."""
        old = self.current_index
        self.current_index = (self.current_index + 1) % len(self.clients)
        print(f"    ↻ Key {old + 1} rate limited → switched to key {self.current_index + 1}")

    def call_with_retry(self, test_fn, max_retries=None):
        """
        Execute test_fn(client), retrying with the next key on 429 errors.
        
        Stays on the current key for all subsequent tests until another
        429 is hit. Only rotates when necessary.
        """
        if max_retries is None:
            max_retries = len(self.keys) * 2  # two full rotations

        for attempt in range(max_retries):
            try:
                return test_fn(self.current_client)
            except Exception as e:
                # Detect rate limit errors from google-genai SDK
                # The SDK raises google.genai.errors.ClientError with code 429
                # or google.api_core.exceptions.ResourceExhausted
                error_str = str(e).lower()
                error_code = getattr(e, 'code', None)
                is_rate_limit = (
                    error_code == 429
                    or "429" in error_str
                    or "resource_exhausted" in error_str
                    or "rate_limit_exceeded" in error_str
                    or "quota" in error_str
                )

                if is_rate_limit and attempt < max_retries - 1:
                    print(f"    ⚠ Rate limit hit on key {self.current_index + 1}: {type(e).__name__}")
                    self.rotate()
                    time.sleep(1)  # brief pause before retrying with new key
                    # If we've cycled through all keys, wait longer
                    if (attempt + 1) % len(self.keys) == 0:
                        print(f"    ⏳ All {len(self.keys)} keys exhausted, waiting 30s...")
                        time.sleep(30)
                    continue
                else:
                    raise  # non-rate-limit error, or retries exhausted


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
rotator = KeyRotator()

# Shared fixtures
NO_HISTORY = []

SAMPLE_HISTORY = [
    {"role": "user", "content": "Analyze my portfolio"},
    {"role": "assistant", "content": "Your portfolio has a Sharpe ratio of 1.2, "
     "volatility of 18%, and a max drawdown of -12%. The NN classifies this as "
     "Medium risk with 78% confidence. Your biggest risk contributor is TSLA at 55%."},
]

DELAY = 20  # seconds between API calls (shorter now with key rotation)

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
    check_entities: dict = None,
):
    """Run one test case against the live Gemini API with key rotation."""
    global passed, failed, errors

    time.sleep(DELAY)

    try:
        def call(client):
            return classify_intent(
                message=message,
                recent_history=history,
                portfolio_changed=portfolio_changed,
                client=client,
            )

        result = rotator.call_with_retry(call)
    except Exception as e:
        failed += 1
        errors.append(f"  CRASH  {name}: {e}")
        print(f"  💥 {name}: CRASHED — {e}")
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

        if "concept" in check_entities and check_entities["concept"]:
            if not result.extracted_concept:
                entity_ok = False
                entity_notes.append(f"expected concept '{check_entities['concept']}', got None")

    all_ok = primary_ok and secondary_ok and entity_ok

    if all_ok:
        passed += 1
        symbol = "✓"
    else:
        failed += 1
        symbol = "✗"

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
    print(f"    → {primary_str}{secondary_str} (conf={result.confidence:.2f})")
    print(f"    → {result.reasoning}")

    if result.extracted_metrics:
        print(f"    → metrics: {result.extracted_metrics}")
    if result.extracted_concept:
        print(f"    → concept: {result.extracted_concept}")
    if entity_notes:
        print(f"    → ENTITY ISSUES: {'; '.join(entity_notes)}")

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

#     test("General: simple greeting",
#          "Hi", Intent.GENERAL_CHAT)

#     test("General: thanks",
#          "Thanks!", Intent.GENERAL_CHAT)

#     test("General: capability question",
#          "What can you do?", Intent.GENERAL_CHAT)

#     test("Full analysis: explicit request",
#          "Analyze my portfolio", Intent.FULL_ANALYSIS)

#     test("Full analysis: risk question",
#          "Is my portfolio risky?", Intent.FULL_ANALYSIS)

#     test("Full analysis: risk breakdown",
#          "Give me a risk breakdown", Intent.FULL_ANALYSIS)

#     test("Full analysis: what-if (formerly rebalance)",
#          "What if I sell Tesla?", Intent.FULL_ANALYSIS)

#     test("Full analysis: add asset (formerly rebalance)",
#          "Should I add bonds to my portfolio?", Intent.FULL_ANALYSIS)

#     test("Full analysis: form update",
#          "", Intent.FULL_ANALYSIS,
#          portfolio_changed=True)

#     test("Specific metric: Sharpe",
#          "What's my Sharpe ratio?", Intent.SPECIFIC_METRIC,
#          check_entities={"metrics": ["sharpe_ratio"]})

#     test("Specific metric: VaR",
#          "Show me the VaR", Intent.SPECIFIC_METRIC,
#          check_entities={"metrics": ["var_95"]})

#     test("Specific metric: beta",
#          "What's my beta?", Intent.SPECIFIC_METRIC,
#          check_entities={"metrics": ["beta"]})

#     test("Concept: what is VaR",
#          "What is Value at Risk?", Intent.CONCEPT_EXPLANATION,
#          check_entities={"concept": "value at risk"})

#     test("Concept: explain beta",
#          "How does beta work?", Intent.CONCEPT_EXPLANATION)

#     test("Trend: future volatility",
#          "Predict future volatility", Intent.TREND_PREDICTION)

#     test("Trend: risk outlook",
#          "Will my risk increase?", Intent.TREND_PREDICTION)

#     test("Follow-up: elaborate",
#          "Can you elaborate on that?", Intent.FOLLOW_UP,
#          history=SAMPLE_HISTORY)

#     test("Follow-up: why",
#          "Why did you say that?", Intent.FOLLOW_UP,
#          history=SAMPLE_HISTORY)

    # =====================================================================
    # 2. DISAMBIGUATION EDGE CASES
    # =====================================================================
    print("\n--- 2. Disambiguation edge cases ---\n")

#     test("specific vs concept: 'what is MY sharpe' → specific",
#          "What is my Sharpe ratio?", Intent.SPECIFIC_METRIC)

#     test("specific vs concept: 'what IS sharpe ratio' → concept",
#          "What is the Sharpe ratio?", Intent.CONCEPT_EXPLANATION)

#     test("specific vs concept: 'tell me more about skewness' → concept",
#          "Tell me more about skewness", Intent.CONCEPT_EXPLANATION)

#     test("Empty + no form change → general_chat",
#          "", Intent.GENERAL_CHAT,
#          portfolio_changed=False)

    # =====================================================================
    # 3. MULTI-INTENT MESSAGES
    # =====================================================================
    print("\n--- 3. Multi-intent messages ---\n")

#     test("Multi: analyze + explain concept",
#          "Analyze my portfolio and explain what VaR means",
#          Intent.FULL_ANALYSIS,
#          expected_secondary=Intent.CONCEPT_EXPLANATION)

#     test("Multi: metric + is it good",
#          "What's my Sharpe ratio and is that good?",
#          Intent.SPECIFIC_METRIC,
#          expected_secondary=Intent.CONCEPT_EXPLANATION)

#     test("Multi: what-if + explain concept",
#          "Compare adding bonds and tell me about diversification",
#          Intent.FULL_ANALYSIS,
#          expected_secondary=Intent.CONCEPT_EXPLANATION)

    # =====================================================================
    # 4. DISGUISED INTENTS
    # =====================================================================
    print("\n--- 4. Disguised intents ---\n")

#     test("Disguised: looks like follow-up → concept",
#          "What did you mean by skewness?", Intent.CONCEPT_EXPLANATION,
#          history=SAMPLE_HISTORY)

#     test("Disguised: thanks + metric → specific_metric",
#          "Thanks! What's my beta though?", Intent.SPECIFIC_METRIC,
#          history=SAMPLE_HISTORY)

#     test("Disguised: greeting + analysis",
#          "Hey! Analyze my portfolio please", Intent.FULL_ANALYSIS)

#     test("Disguised: greeting + what-if",
#          "Hi can you tell me if I should sell Tesla?", Intent.FULL_ANALYSIS)

    # =====================================================================
    # 5. AMBIGUOUS / STRESS TESTS
    # =====================================================================
    print("\n--- 5. Ambiguous / stress tests ---\n")

#     test("Ambiguous: 'How's my risk?'",
#          "How's my risk?", Intent.FULL_ANALYSIS)

#     test("Ambiguous: 'Tell me about my volatility'",
#          "Tell me about my volatility", Intent.SPECIFIC_METRIC)

#     test("Ambiguous: 'What about bonds?'",
#          "What about bonds?", Intent.FULL_ANALYSIS,
#          history=SAMPLE_HISTORY)

#     test("Ambiguous: 'Is this good?'",
#          "Is this good?", Intent.FOLLOW_UP,
#          history=SAMPLE_HISTORY)

#     test("Ambiguous: single word 'Risk'",
#          "Risk", Intent.FULL_ANALYSIS)

#     test("Ambiguous: 'Help me understand my portfolio better'",
#          "Help me understand my portfolio better", Intent.FULL_ANALYSIS)

    # =====================================================================
    # 6. ENTITY EXTRACTION STRESS TESTS
    # =====================================================================
    print("\n--- 6. Entity extraction stress tests ---\n")

#     test("Entity: vague downside → should infer var/cvar/sortino",
#          "How's my downside looking?", Intent.SPECIFIC_METRIC,
#          check_entities={"metrics": ["var_95"]})

#     test("Entity: vague performance → should infer sharpe/sortino",
#          "How's my portfolio performing?", Intent.FULL_ANALYSIS) # honestly i'll give it to the LLM for this one

#     test("Entity: explicit metric extraction",
#          "What's my max drawdown and beta?", Intent.SPECIFIC_METRIC,
#          check_entities={"metrics": ["max_drawdown", "beta"]})

#     test("Entity: concept should extract concept name",
#          "Explain what conditional value at risk means", Intent.CONCEPT_EXPLANATION,
#          check_entities={"concept": "conditional value at risk"})

#     test("Entity: vague concentration → should infer hhi/correlation",
#          "Is my portfolio too concentrated?", Intent.SPECIFIC_METRIC,
#          check_entities={"metrics": ["hhi_concentration"]})

    # =====================================================================
    # 7. HARD MULTI-INTENT EDGE CASES
    # =====================================================================
    print("\n--- 7. Hard multi-intent edge cases ---\n")

#     test("Multi: sell + analyze (both map to full_analysis)",
#          "Sell Tesla and analyze what's left",
#          Intent.FULL_ANALYSIS)

#     test("Multi: fresh analysis override",
#          "Forget my current portfolio. Analyze a new one with MSFT and GOOG at 50/50",
#          Intent.FULL_ANALYSIS)

#     test("Multi: metric + trend",
#          "What's my current volatility and will it get worse?",
#          Intent.SPECIFIC_METRIC,
#          expected_secondary=Intent.TREND_PREDICTION,
#          check_entities={"metrics": ["portfolio_volatility"]})

#     test("Multi: what-if + concept",
#          "Should I add gold to hedge? And what exactly is hedging?",
#          Intent.FULL_ANALYSIS,
#          expected_secondary=Intent.CONCEPT_EXPLANATION)

#     test("Multi: follow-up + metric",
#          "You mentioned my Sharpe was low — what's my Sortino?",
#          Intent.SPECIFIC_METRIC,
#          history=SAMPLE_HISTORY,
#          check_entities={"metrics": ["sortino_ratio"]})

    # =====================================================================
    # 8. ADVERSARIAL / TRICKY PHRASING
    # =====================================================================
    print("\n--- 8. Adversarial / tricky phrasing ---\n")

#     test("Tricky: negation — 'don't analyze'",
#          "Don't analyze my portfolio, just tell me what beta means",
#          Intent.CONCEPT_EXPLANATION,
#          check_entities={"concept": "beta"})

    test("Tricky: hypothetical",
         "If I had a portfolio of all tech stocks, would it be risky?",
         Intent.CONCEPT_EXPLANATION) # mistake made

#     test("Tricky: question about the agent",
#          "What metrics can you calculate?",
#          Intent.GENERAL_CHAT)

    test("Tricky: comparison language but not what-if",
         "How does my portfolio compare to the S&P 500?",
         Intent.SPECIFIC_METRIC,
         check_entities={"metrics": ["beta"]}) # mistake made but this is bc 503 error

#     test("Tricky: past tense",
#          "Why did my portfolio drop last week?",
#          Intent.FULL_ANALYSIS)

#     test("Tricky: very long message with buried intent",
#          "So I've been reading about portfolio theory and modern asset allocation "
#          "frameworks and I was wondering about my specific case — I know I have AAPL "
#          "and TSLA but honestly I just want to know my Sharpe ratio right now",
#          Intent.SPECIFIC_METRIC,
#          check_entities={"metrics": ["sharpe_ratio"]})

#     test("Tricky: typo/informal",
#          "whats my sharp ratio lol",
#          Intent.SPECIFIC_METRIC,
#          check_entities={"metrics": ["sharpe_ratio"]})

#     test("Tricky: multiple metrics in one ask",
#          "Give me the VaR, CVaR, and max drawdown",
#          Intent.SPECIFIC_METRIC,
#          check_entities={"metrics": ["var_95", "cvar_95", "max_drawdown"]})

#     test("Tricky: sounds like what-if but is concept",
#          "What happens to risk when you add bonds to a portfolio?",
#          Intent.CONCEPT_EXPLANATION)

# #     test("Tricky: empty message no form change",
# #          "", Intent.GENERAL_CHAT,
# #          portfolio_changed=False)

    test("Trend vs full_analysis: 'Will my volatility get worse?' → trend",
          "Will my volatility get worse?", Intent.TREND_PREDICTION,
          check_entities={"metrics": ["portfolio_volatility"]})

    test("Hypothetical follow-up: 'What if Sharpe ratios were negative?' → concept",
          "What if Sharpe ratios were negative across the board?",
          Intent.CONCEPT_EXPLANATION,
          history=SAMPLE_HISTORY)

    # =====================================================================
    # 9. BOUNDARY: full_analysis vs specific_metric
    # =====================================================================
    print("\n--- 9. Boundary: full_analysis vs specific_metric ---\n")

#     test("Boundary: 'Is this risky?' → full_analysis",
#          "Is this risky?", Intent.FULL_ANALYSIS)

#     test("Boundary: 'How risky is my portfolio?' → full_analysis",
#          "How risky is my portfolio?", Intent.FULL_ANALYSIS)

#     test("Boundary: 'What's my risk level?' → full_analysis",
#          "What's my risk level?", Intent.FULL_ANALYSIS)

#     test("Boundary: 'Show me my volatility' → specific_metric",
#          "Show me my volatility", Intent.SPECIFIC_METRIC,
#          check_entities={"metrics": ["portfolio_volatility"]})

#     test("Boundary: 'Risk breakdown' → full_analysis",
#          "Give me a risk breakdown", Intent.FULL_ANALYSIS)

    test("Boundary: 'How diversified am I?' → specific_metric",
         "How diversified am I?", Intent.SPECIFIC_METRIC,
         check_entities={"metrics": ["avg_pairwise_correlation"]}) # mistake made

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
