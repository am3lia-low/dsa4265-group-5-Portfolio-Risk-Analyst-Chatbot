"""
Intent Classification Module for Portfolio Risk Analyst Agent
==============================================================
Classifies user messages into 7 intents using Gemini 2.5 Flash-Lite.
 
Design/Workflow:
  - model
  - creating Pydantic schema for constrained output
  - system prompt to classify user intents
  - building context (taking previous messages)
 
Models:
  - Intent classification: gemini-2.5-flash-lite (15 RPM, 1000 RPD free)
  - Explanation generation: gemini-2.5-flash (separate module)
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
# 0. Model
# ---------------------------------------------------------------------------
CLASSIFIER_MODEL = "gemini-2.5-flash" # boy, i hope this works



# ---------------------------------------------------------------------------
# 1. creating Pydantic schema for constrained output
# gemini has a nice way to configure these outputs
# https://ai.google.dev/gemini-api/docs/structured-output?example=recipe
# ---------------------------------------------------------------------------
class Intent(str, Enum):
    FULL_ANALYSIS = "full_analysis"
    REBALANCE = "rebalance"
    SPECIFIC_METRIC = "specific_metric"
    CONCEPT_EXPLANATION = "concept_explanation"
    TREND_PREDICTION = "trend_prediction"
    FOLLOW_UP = "follow_up"
    GENERAL_CHAT = "general_chat"
 

# Intents that require an active portfolio
PORTFOLIO_REQUIRED = {
    Intent.FULL_ANALYSIS,
    Intent.REBALANCE,
    Intent.SPECIFIC_METRIC,
    Intent.TREND_PREDICTION,
}


class IntentClassification(BaseModel):
    """Schema enforced by Gemini's structured output. The model MUST return a JSON object matching this shape."""
 
    primary_intent: str = Field(
        description="The main intent. Must be one of: full_analysis, rebalance, specific_metric, concept_explanation, trend_prediction, follow_up, general_chat"
    )
    confidence: float = Field(
        description="Confidence in the classification, 0.0 to 1.0",
    )
    reasoning: str = Field(
        description="1-2 sentence explanation of why this intent was chosen"
    )
    secondary_intent: Optional[str] = Field(
        default=None,
        description="Optional second intent if the message contains two distinct requests. One of the 7 intent names, or null."
    )
    extracted_metrics: Optional[list[str]] = Field(
        default=None,
        description="Metric names mentioned, normalized to: portfolio_volatility, vol_of_vol, var_95, cvar_95, max_drawdown, sharpe_ratio, sortino_ratio, skewness, excess_kurtosis, beta, hhi_concentration, avg_pairwise_correlation, risk_contribution"
    )
    extracted_tickers: Optional[list[str]] = Field(
        default=None,
        description="Stock ticker symbols mentioned, uppercase (e.g. AAPL, TSLA)",
    )
    extracted_concept: Optional[str] = Field(
        default=None,
        description="The concept name if the user is asking about a concept",
    )


# this will pose as the output from the intent classifier. inherits from dataclass
@dataclass
class IntentResult:
    """Structured output from the intent classifier."""
    primary_intent: Intent
    confidence: float
    reasoning: str
    secondary_intent: Optional[Intent] = None
    extracted_metrics: Optional[list[str]] = None
    extracted_tickers: Optional[list[str]] = None
    extracted_concept: Optional[str] = None
    requires_portfolio: bool = False



# ---------------------------------------------------------------------------
# 2. system prompt — all disambiguation rules baked in
# might put this in a separate config later
# ---------------------------------------------------------------------------
CLASSIFY_SYSTEM_PROMPT = """\
You are the intent classifier for a portfolio risk analysis chatbot.
Given a user message and context, classify it into the correct intent(s).
 
## The 7 intents
 
1. **full_analysis** — Complete risk assessment of a portfolio.
   Examples: "Analyze my portfolio", "Is my portfolio risky?", "Give me a risk breakdown"
   Also: first portfolio submission via form, or user explicitly asking for a fresh/new
   analysis ("analyze this new portfolio", "start fresh").
 
2. **rebalance** — Compare a PROPOSED change against the CURRENT portfolio.
   Examples: "What if I sell Tesla?", "Should I add bonds?", "Compare 50/50 split"
   Also: form update when a portfolio already exists (DEFAULT for form edits).
   KEY: rebalance is about COMPARISON (before vs after). Requires an existing portfolio.
 
3. **specific_metric** — User asks about specific metric VALUE(S) for THEIR portfolio.
   Examples: "What's my Sharpe ratio?", "Show me the VaR", "How skewed are my returns?"
   KEY: the user wants their portfolio's number, not a general explanation.
 
4. **concept_explanation** — User asks what a concept/metric MEANS in general.
   Examples: "What is Value at Risk?", "Explain CVaR", "How does beta work?"
   KEY: educational — about definitions and theory, not about their portfolio's values.
 
5. **trend_prediction** — User asks about FUTURE risk outlook or volatility forecast.
   Examples: "Will my risk increase?", "What's the outlook?", "Predict future volatility"
 
6. **follow_up** — User asks for clarification about something from the PREVIOUS response.
   Examples: "Why did you say that?", "Can you elaborate on that last point?"
   CAREFUL: many apparent follow-ups are actually other intents. See rules below.
 
7. **general_chat** — Greetings, thanks, capability questions, off-topic.
   Examples: "Hi", "Thanks!", "What can you do?"
 
## Critical disambiguation rules
 
### full_analysis vs rebalance
- First portfolio submission (is_first_portfolio=true) → always full_analysis
- Form update + existing portfolio + no override language → rebalance
- Form update + user says "analyze fresh" / "start over" / "new analysis" → full_analysis
- Chat message with entirely new tickers + "analyze this" → full_analysis
- Chat message proposing changes to current holdings → rebalance
 
### specific_metric vs concept_explanation
- "What's MY Sharpe ratio?" (possessive + metric) → specific_metric
- "What IS the Sharpe ratio?" (definitional) → concept_explanation
- "Tell me more about skewness" (educational) → concept_explanation
- Metric question but NO portfolio loaded → concept_explanation
 
### follow_up vs other intents (disguised intents)
- "What do you mean by skewness?" → concept_explanation (NOT follow_up)
- "What's my actual Sharpe number?" after analysis → specific_metric (NOT follow_up)
- Only classify as follow_up when the user is genuinely asking about something
  ALREADY SAID and not introducing a new topic.
 
### Greetings with embedded queries
- "Hi, analyze my portfolio" → primary: full_analysis. The greeting is incidental.
- "Thanks! What's my Sharpe?" → primary: specific_metric. The thanks is incidental.
- ONLY classify as general_chat when the ENTIRE message is a greeting/thanks with no query.
 
### Portfolio-dependent intents without a portfolio
- specific_metric + no portfolio → concept_explanation instead
- trend_prediction + no portfolio → general_chat instead
- rebalance + no portfolio → full_analysis instead
 
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
 
- **rebalance** → you SHOULD populate extracted_tickers if specific assets are mentioned.
  "What if I sell Tesla?" → extracted_tickers: ["TSLA"]
  "Should I add bonds?" → extracted_tickers: ["BND"] (infer common ticker)
  Vague rebalance ("rebalance my portfolio") → extracted_tickers can be null.
 
- **full_analysis** → extracted_tickers only if the user mentions specific new tickers.
 
- **trend_prediction**, **follow_up**, **general_chat** → entities are optional.
 
### General rules
- Normalize metric names to: portfolio_volatility, vol_of_vol, var_95, cvar_95,
  max_drawdown, sharpe_ratio, sortino_ratio, skewness, excess_kurtosis, beta,
  hhi_concentration, avg_pairwise_correlation, risk_contribution
- Ticker symbols must be uppercase (e.g. AAPL, TSLA, BND)
- Do NOT hallucinate entities that have no basis in the message
- When inferring, prefer the most commonly intended meaning
"""
 
 
# ---------------------------------------------------------------------------
# 3. context builder
# taking in chat history (last 200 characters)
# and checks if portfolio exists (to differentiate rebalance and full_analysis intent)
# ---------------------------------------------------------------------------
def _build_context(
    message: str,
    recent_history: list[dict],
    portfolio_changed: bool,
    is_first_portfolio: bool,
) -> str:
    """Build the user-turn context string for the LLM."""
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
- is_first_portfolio: {is_first_portfolio}
 
## Recent conversation history
{history_text}
 
## Current user message
{message}"""



# ---------------------------------------------------------------------------
# 4. pre-check
# prelim checking of message if needed
# honestly this depends on UI
# right now it checks if message is empty
# let's get back to this if needed 
# ---------------------------------------------------------------------------
def _pre_check(message: str) -> Optional[IntentResult]:
    """Return IntentResult for trivially invalid inputs, or None to proceed."""
    if not message or not message.strip():
        return IntentResult(
            primary_intent=Intent.GENERAL_CHAT,
            confidence=1.0,
            reasoning="Empty message — nothing to classify",
            requires_portfolio=False,
        )
    return None



# ---------------------------------------------------------------------------
# 5. Fuzzy match safety net
# this exists in case gemini decided to give me the wrong words
# nothing like resorting to good ol' hardcoding
# ---------------------------------------------------------------------------
INTENT_ALIASES: dict[str, str] = {
    "full": "full_analysis",
    "analysis": "full_analysis",
    "rebalance": "rebalance",
    "compare": "rebalance",
    "comparison": "rebalance",
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
# 6. main classify function
# ---------------------------------------------------------------------------
def classify_intent(
    message: str,
    recent_history: list[dict],
    portfolio_changed: bool = False,
    is_first_portfolio: bool = False,
    client: Optional[genai.Client] = None,
) -> IntentResult:
    """
    Classify a user message into one or more of 7 intents.
 
    Parameters
    ----------
    message : str
        The user's current message.
    recent_history : list[dict]
        Recent chat history, already trimmed by caller (last 3-4 exchanges).
        Each dict: {"role": "user"|"assistant", "content": "..."}.
    portfolio_changed : bool
        True if the portfolio form was just updated this turn.
    is_first_portfolio : bool
        True if this is the very first portfolio submission in the session.
    client : genai.Client | None
        Gemini client. If None, creates one from env var GEMINI_API_KEY.
 
    Returns
    -------
    IntentResult
    """
    # Step 0: Guard clause
    # shelved for now
    # quick = _pre_check(message)
    # if quick is not None:
    #     return quick
 
    # Step 1: Initialize client
    if client is None:
        client = genai.Client()
 
    # Step 2: Build context
    context = _build_context(
        message, recent_history, portfolio_changed, is_first_portfolio
    )
 
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
        logger.error("Gemini API call failed: %s", e)
        return IntentResult(
            primary_intent=Intent.GENERAL_CHAT,
            confidence=0.0,
            reasoning=f"LLM call failed: {e}",
            requires_portfolio=False,
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
            requires_portfolio=False,
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
        extracted_tickers=data.get("extracted_tickers"),
        extracted_concept=data.get("extracted_concept"),
        requires_portfolio=primary in PORTFOLIO_REQUIRED,
    )
 
    logger.info(
        "Classified: %s (%.2f)%s | %s",
        result.primary_intent.value,
        result.confidence,
        f" + {result.secondary_intent.value}" if result.secondary_intent else "",
        result.reasoning,
    )
 
    return result