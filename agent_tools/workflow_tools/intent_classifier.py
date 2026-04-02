"""
Intent Classification Module for Portfolio Risk Analyst Agent
==============================================================
Classifies user messages into 6 intents using Gemini 2.5 Flash-Lite.

Design:
  - LLM-only classification via Google GenAI SDK (no heuristic layer)
  - Constrained output via Pydantic structured JSON schema
  - Multi-intent support (primary + optional secondary, max 2)
  - All disambiguation rules baked into the system prompt
  - Fuzzy matching as safety net (not a classification system)
  - Minimal pre-check for empty messages only

Assumptions:
  - A portfolio is ALWAYS present (UI prevents queries without one)
  - Ticker validation is handled by the UI form
  - Rebalance is merged into full_analysis — if portfolio_changed is true,
    the orchestrator compares current vs previous automatically

Models:
  - Intent classification: gemini-2.5-flash-lite (15 RPM, 1000 RPD free)
  - Explanation generation: gemini-2.5-flash (separate module)

SDK: google-genai (the unified SDK, NOT the deprecated google-generativeai)
  Install: pip install google-genai
"""

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASSIFIER_MODEL = "gemini-2.5-flash-lite"


# ---------------------------------------------------------------------------
# Intent enum (6 intents — rebalance merged into full_analysis)
# ---------------------------------------------------------------------------
class Intent(str, Enum):
    FULL_ANALYSIS = "full_analysis"
    SPECIFIC_METRIC = "specific_metric"
    CONCEPT_EXPLANATION = "concept_explanation"
    TREND_PREDICTION = "trend_prediction"
    FOLLOW_UP = "follow_up"
    GENERAL_CHAT = "general_chat"


# ---------------------------------------------------------------------------
# Pydantic schema for constrained output
# ---------------------------------------------------------------------------
class IntentClassification(BaseModel):
    """Schema enforced by Gemini's structured output. The model MUST return
    a JSON object matching this shape."""

    primary_intent: str = Field(
        description=(
            "The main intent. Must be one of: full_analysis, "
            "specific_metric, concept_explanation, trend_prediction, "
            "follow_up, general_chat"
        )
    )
    confidence: float = Field(
        description="Confidence in the classification, 0.0 to 1.0",
    )
    reasoning: str = Field(
        description="1-2 sentence explanation of why this intent was chosen"
    )
    secondary_intent: Optional[str] = Field(
        default=None,
        description=(
            "Optional second intent if the message contains two distinct "
            "requests. One of the 6 intent names, or null."
        ),
    )
    extracted_metrics: Optional[list[str]] = Field(
        default=None,
        description=(
            "Metric names mentioned, normalized to: portfolio_volatility, "
            "vol_of_vol, var_95, cvar_95, max_drawdown, sharpe_ratio, "
            "sortino_ratio, skewness, excess_kurtosis, beta, "
            "hhi_concentration, avg_pairwise_correlation, risk_contribution"
        ),
    )
    extracted_concept: Optional[str] = Field(
        default=None,
        description="The concept name if the user is asking about a concept",
    )


# ---------------------------------------------------------------------------
# Result dataclass (what route_and_execute() consumes)
# ---------------------------------------------------------------------------
@dataclass
class IntentResult:
    """Structured output from the intent classifier."""
    primary_intent: Intent
    confidence: float
    reasoning: str
    secondary_intent: Optional[Intent] = None
    extracted_metrics: Optional[list[str]] = None
    extracted_concept: Optional[str] = None


# ---------------------------------------------------------------------------
# System prompt — all disambiguation rules baked in
# ---------------------------------------------------------------------------
CLASSIFY_SYSTEM_PROMPT = """\
You are the intent classifier for a portfolio risk analysis chatbot.
Given a user message and context, classify it into the correct intent(s).

IMPORTANT CONTEXT: A portfolio is ALWAYS loaded — the UI prevents users from
chatting without one. You never need to handle "no portfolio" cases.
Portfolio changes (adding/removing assets, changing weights) are handled through
the UI form, not through chat. The orchestrator will check portfolio_changed
to decide whether to compare against a previous portfolio.

## The 6 intents

1. **full_analysis** — Complete, holistic risk assessment of a portfolio.
   Examples: "Analyze my portfolio", "Is my portfolio risky?", "Give me a risk breakdown",
   "How risky is this?", "Assess my portfolio", "How's my risk?",
   "What if I sell Tesla?", "Should I add bonds?", "Compare this allocation"
   KEY: full_analysis is for BROAD, HOLISTIC questions about overall risk. If the user
   asks something vague like "is this risky?" or "how's my risk?", that is full_analysis
   because they want a comprehensive answer, not a single number.
   Also covers rebalancing questions — any question about changing portfolio composition
   is full_analysis. The orchestrator handles comparison logic separately.

2. **specific_metric** — User asks about a SPECIFIC, NAMED metric for THEIR portfolio.
   Examples: "What's my Sharpe ratio?", "Show me the VaR", "How skewed are my returns?",
   "What's my max drawdown?", "Show me the beta"
   KEY: the user must reference a SPECIFIC metric by name (Sharpe, VaR, beta, drawdown,
   volatility, etc.). Vague questions like "is my portfolio risky?" or "how's my risk?"
   are NOT specific_metric — those are full_analysis.
   RULE: If you cannot identify a specific metric the user is asking about, it is
   probably full_analysis, not specific_metric.

3. **concept_explanation** — User asks what a concept/metric MEANS in general.
   Examples: "What is Value at Risk?", "Explain CVaR", "How does beta work?",
   "What happens to risk when you add bonds to a portfolio?",
   "If I had a portfolio of all tech stocks, would it be risky?"
   KEY: educational — about definitions and theory, not about their portfolio's values.
   Also covers HYPOTHETICAL questions — if the user asks "what if I had X" or
   "would Y be risky in general", they are asking about concepts, not requesting
   analysis of their actual portfolio.
   
   The agent can ONLY compute these 13 metrics — there are no others:
    - portfolio_volatility: annualized standard deviation
    - vol_of_vol: volatility instability (21-day rolling)
    - var_95: Value at Risk at 95% confidence
    - cvar_95: Conditional VaR / Expected Shortfall
    - max_drawdown: largest peak-to-trough decline
    - sharpe_ratio: return per unit of total volatility
    - sortino_ratio: return per unit of downside volatility
    - skewness: crash vs rally asymmetry
    - excess_kurtosis: tail fatness / black swan frequency
    - beta: portfolio sensitivity to S&P 500
    - hhi_concentration: weight concentration index
    - avg_pairwise_correlation: diversification level
    - risk_contribution: per-asset share of portfolio volatility

    When a user's question maps to one or more of these metrics, classify as
    specific_metric. If their question cannot be answered by any of these
    metrics, classify as full_analysis or concept_explanation as appropriate.

4. **trend_prediction** — User asks about FUTURE risk outlook or volatility forecast.
   Examples: "Will my risk increase?", "What's the outlook?", "Predict future volatility"

5. **follow_up** — User asks for clarification about something from the PREVIOUS response.
   Examples: "Why did you say that?", "Can you elaborate on that last point?"
   CAREFUL: many apparent follow-ups are actually other intents. See rules below.

6. **general_chat** — Greetings, thanks, capability questions, off-topic.
   Examples: "Hi", "Thanks!", "What can you do?"

## Critical disambiguation rules

### full_analysis vs specific_metric (IMPORTANT — read carefully)
- If the user names a SPECIFIC metric (Sharpe, VaR, beta, drawdown, sortino, skewness,
  kurtosis, volatility, correlation, HHI, risk contribution) → specific_metric
- If the user asks about a specific DIMENSION of risk, even without naming the exact
  metric (e.g. "how diversified am I?", "what's my downside?", "how concentrated is my
  portfolio?", "how sensitive am I to the market?") → specific_metric
  These map to known metrics and the user is clearly asking about one aspect, not a
  holistic assessment.
- If the user asks a BROAD or VAGUE question about OVERALL risk/portfolio quality
  ("is this risky?", "how's my risk?", "assess my portfolio", "risk breakdown",
  single word "risk") → full_analysis
- When in doubt between full_analysis and specific_metric → choose full_analysis.

### Empty messages and form signals (IMPORTANT)
- When the user message is EMPTY, the form signals tell you the intent:
    - Empty message + portfolio_changed=true → full_analysis
    - Empty message + portfolio_changed=false → general_chat
- Do NOT ignore form signals. They are the primary classification input when the
  message is empty.

### specific_metric vs concept_explanation
- "What's MY Sharpe ratio?" (possessive + metric) → specific_metric
- "What IS the Sharpe ratio?" (definitional) → concept_explanation
- "Tell me more about skewness" (educational) → concept_explanation

### follow_up vs other intents (disguised intents)
- "What do you mean by skewness?" → concept_explanation (NOT follow_up)
- "What's my actual Sharpe number?" after analysis → specific_metric (NOT follow_up)
- Only classify as follow_up when the user is genuinely asking about something
  ALREADY SAID and not introducing a new topic.

### Greetings with embedded queries
- "Hi, analyze my portfolio" → primary: full_analysis. The greeting is incidental.
- "Thanks! What's my Sharpe?" → primary: specific_metric. The thanks is incidental.
- ONLY classify as general_chat when the ENTIRE message is a greeting/thanks with no query.

### Multi-intent messages
- "Analyze my portfolio and explain what VaR means" →
    primary: full_analysis, secondary: concept_explanation
- "What's my Sharpe and is that good?" →
    primary: specific_metric, secondary: concept_explanation
- Only set secondary_intent when there are clearly TWO distinct requests.
  Most messages are single-intent.

## Entity extraction rules

Entity extraction is CRITICAL — downstream workflows depend on it.

### Mandatory extraction by intent

- **specific_metric** → you MUST populate extracted_metrics. If the user says something
  vague like "how's my downside?" or "tell me about my risk numbers", infer the most
  likely metrics. Map common phrases:
    - "downside" / "downside risk" → var_95, cvar_95, sortino_ratio
    - "risk" / "how risky" (when specific_metric, not full_analysis) → portfolio_volatility
    - "performance" / "returns" → sharpe_ratio, sortino_ratio
    - "concentration" / "diversification" → hhi_concentration, avg_pairwise_correlation
    - "tail risk" / "extreme events" → excess_kurtosis, cvar_95
    - "market sensitivity" → beta
  If you truly cannot determine any metric, still classify correctly — the orchestrator
  will ask the user to clarify.

- **concept_explanation** → you MUST populate extracted_concept with the concept name.
  If the user says "explain that ratio", infer from conversation history which concept.
  If ambiguous, use the broadest reasonable interpretation.

- **full_analysis**, **trend_prediction**, **follow_up**, **general_chat** →
  entities are optional.

### General rules
- Normalize metric names to: portfolio_volatility, vol_of_vol, var_95, cvar_95,
  max_drawdown, sharpe_ratio, sortino_ratio, skewness, excess_kurtosis, beta,
  hhi_concentration, avg_pairwise_correlation, risk_contribution
- Do NOT hallucinate entities that have no basis in the message
- When inferring, prefer the most commonly intended meaning
"""


# ---------------------------------------------------------------------------
# Fuzzy match safety net
# ---------------------------------------------------------------------------
INTENT_ALIASES: dict[str, str] = {
    "full": "full_analysis",
    "analysis": "full_analysis",
    "rebalance": "full_analysis",      # merged into full_analysis
    "compare": "full_analysis",         # merged into full_analysis
    "comparison": "full_analysis",      # merged into full_analysis
    "metric": "specific_metric",
    "specific": "specific_metric",
    "concept": "concept_explanation",
    "explain": "concept_explanation",
    "explanation": "concept_explanation",
    "trend": "trend_prediction",
    "predict": "trend_prediction",
    "prediction": "trend_prediction",
    "follow": "follow_up",
    "followup": "follow_up",
    "clarify": "follow_up",
    "chat": "general_chat",
    "general": "general_chat",
    "greeting": "general_chat",
}


def _resolve_intent(raw: str) -> Intent:
    """Map LLM output to Intent enum, with fuzzy fallback."""
    raw = raw.strip().lower()
    try:
        return Intent(raw)
    except ValueError:
        pass
    for alias, intent_name in INTENT_ALIASES.items():
        if alias in raw:
            logger.warning(
                "Fuzzy matched intent '%s' -> '%s'", raw, intent_name,
            )
            return Intent(intent_name)
    logger.error("Could not resolve intent '%s', defaulting to general_chat", raw)
    return Intent.GENERAL_CHAT


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------
def _build_context(
    message: str,
    recent_history: list[dict],
    portfolio_changed: bool,
) -> str:
    """Build the user-turn context string for the LLM.
    Portfolio is always present (UI guarantees this)."""
    # Chat history (already trimmed by caller)
    if recent_history:
        lines = []
        for msg in recent_history:
            role = msg["role"].upper()
            content = msg["content"]
            if len(content) > 200:
                content = content[:200] + "..."
            lines.append(f"[{role}]: {content}")
        history_text = "\n".join(lines)
    else:
        history_text = "(no prior messages — start of conversation)"

    return f"""## Portfolio state
- portfolio_changed_this_turn: {portfolio_changed}

## Recent conversation history
{history_text}

## Current user message
{message}"""


# ---------------------------------------------------------------------------
# Minimal pre-check (guard clause only)
# ---------------------------------------------------------------------------
def _pre_check(message: str) -> Optional[IntentResult]:
    """Return IntentResult for trivially invalid inputs, or None to proceed."""
    if not message or not message.strip():
        return IntentResult(
            primary_intent=Intent.GENERAL_CHAT,
            confidence=1.0,
            reasoning="Empty message — nothing to classify",
        )
    return None


# ---------------------------------------------------------------------------
# Main classify function
# ---------------------------------------------------------------------------
def classify_intent(
    message: str,
    recent_history: list[dict],
    portfolio_changed: bool = False,
    client: Optional[genai.Client] = None,
) -> IntentResult:
    """
    Classify a user message into one or more of 6 intents.

    Parameters
    ----------
    message : str
        The user's current message.
    recent_history : list[dict]
        Recent chat history, already trimmed by caller (last 3-4 exchanges).
        Each dict: {"role": "user"|"assistant", "content": "..."}.
    portfolio_changed : bool
        True if the portfolio form was just updated this turn.
    client : genai.Client | None
        Gemini client. If None, creates one from env var GEMINI_API_KEY.

    Returns
    -------
    IntentResult
    """
    # Step 0: Guard clause
    quick = _pre_check(message)
    if quick is not None:
        return quick

    # Step 1: Initialize client
    if client is None:
        client = genai.Client()

    # Step 2: Build context
    context = _build_context(message, recent_history, portfolio_changed)

    # Step 3: Call Gemini with structured output
    try:
        response = client.models.generate_content(
            model=CLASSIFIER_MODEL,
            contents=context,
            config=types.GenerateContentConfig(
                system_instruction=CLASSIFY_SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_json_schema=IntentClassification.model_json_schema(),
                temperature=0.0,
            ),
        )
    except Exception as e:
        # Let rate limit errors bubble up so callers (e.g. key rotator) can handle them
        error_str = str(e).lower()
        error_code = getattr(e, 'code', None)
        is_rate_limit = (
            error_code == 429
            or "429" in error_str
            or "resource_exhausted" in error_str
            or "rate_limit_exceeded" in error_str
            or "quota" in error_str
        )
        if is_rate_limit:
            raise  # let caller handle rotation

        logger.error("Gemini API call failed: %s", e)
        return IntentResult(
            primary_intent=Intent.GENERAL_CHAT,
            confidence=0.0,
            reasoning=f"LLM call failed: {e}",
        )

    # Step 4: Parse structured response
    try:
        data = json.loads(response.text)
    except (json.JSONDecodeError, TypeError) as e:
        logger.error("Failed to parse response: %s | raw: %s", e, response.text)
        return IntentResult(
            primary_intent=Intent.GENERAL_CHAT,
            confidence=0.0,
            reasoning=f"Response parsing failed: {e}",
        )

    # Step 5: Build IntentResult with fuzzy match safety net
    primary = _resolve_intent(data.get("primary_intent", "general_chat"))

    secondary = None
    raw_secondary = data.get("secondary_intent")
    if raw_secondary:
        secondary = _resolve_intent(raw_secondary)

    result = IntentResult(
        primary_intent=primary,
        confidence=data.get("confidence", 0.5),
        reasoning=data.get("reasoning", ""),
        secondary_intent=secondary,
        extracted_metrics=data.get("extracted_metrics"),
        extracted_concept=data.get("extracted_concept"),
    )

    logger.info(
        "Classified: %s (%.2f)%s | %s",
        result.primary_intent.value,
        result.confidence,
        f" + {result.secondary_intent.value}" if result.secondary_intent else "",
        result.reasoning,
    )

    return result
