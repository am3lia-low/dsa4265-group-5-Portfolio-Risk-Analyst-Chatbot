"""
Agent LLM Module for Portfolio Risk Analyst Agent
===================================================
Unified module for all LLM interactions:
  1. Intent classification (Gemini 2.5 Flash-Lite)
  2. Explanation generation (Gemini 2.5 Flash)
  3. API key rotation with automatic retry on 429

Design:
  - Single shared Gemini client with key rotation
  - Intent classification: structured JSON output via Pydantic
  - Explanation generation: free-form text, prompt adapts to intent
  - Rate limit errors bubble up for key rotation handling
  - Non-Gemini API keys (e.g. RAG/embeddings) loaded separately

Assumptions:
  - Portfolio is ALWAYS present (UI enforces this)
  - Rebalance merged into full_analysis
  - Orchestrator assembles ExplanationContext before calling generate_explanation()

SDK: google-genai (unified SDK)
  Install: pip install google-genai
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================
CLASSIFIER_MODEL = "gemini-2.5-flash-lite"
EXPLANATION_MODEL = "gemini-2.5-flash"


# =============================================================================
# API Key Rotation
# =============================================================================
class KeyRotator:
    """Manages multiple Gemini API keys with automatic rotation on 429 errors.

    Sticks with the SAME key until a rate limit is hit, then switches
    to the next key and retries the failed request.

    Usage:
        rotator = KeyRotator()
        result = rotator.call_with_retry(lambda client: classify_intent(..., client=client))
    """

    def __init__(self, keys: Optional[list[str]] = None):
        """
        Parameters
        ----------
        keys : list[str] | None
            List of Gemini API keys. If None, loads from env vars
            GEMINI_API_KEY1 through GEMINI_API_KEY6, falling back
            to GEMINI_API_KEY or GOOGLE_API_KEY.
        """
        if keys is None:
            keys = self._load_keys_from_env()

        if not keys:
            raise ValueError(
                "No Gemini API keys found. Set GEMINI_API_KEY1 through "
                "GEMINI_API_KEY6 in your .env file, or set GEMINI_API_KEY."
            )

        self.keys = keys
        self.current_index = 0
        self.clients = [genai.Client(api_key=k) for k in self.keys]
        logger.info("KeyRotator initialized with %d API key(s).", len(self.keys))

    @staticmethod
    def _load_keys_from_env() -> list[str]:
        """Load API keys from environment variables."""
        keys = []
        for i in range(1, 7):
            key = os.environ.get(f"GEMINI_API_KEY{i}")
            if key:
                keys.append(key)

        # Fallback to single key
        if not keys:
            single = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if single:
                keys.append(single)

        return keys

    @property
    def current_client(self) -> genai.Client:
        """The currently active Gemini client."""
        return self.clients[self.current_index]

    def rotate(self):
        """Switch to the next API key. Only called on 429 errors."""
        old = self.current_index
        self.current_index = (self.current_index + 1) % len(self.clients)
        logger.warning(
            "Key %d rate limited → switched to key %d", old + 1, self.current_index + 1
        )

    def call_with_retry(self, fn, max_retries: Optional[int] = None):
        """
        Execute fn(client) with automatic key rotation on rate limit errors.

        Parameters
        ----------
        fn : callable
            Function that takes a genai.Client and returns a result.
        max_retries : int | None
            Max retry attempts. Defaults to 2x the number of keys.

        Returns
        -------
        The return value of fn(client).
        """
        if max_retries is None:
            max_retries = len(self.keys) * 2

        for attempt in range(max_retries):
            try:
                return fn(self.current_client)
            except Exception as e:
                error_str = str(e).lower()
                error_code = getattr(e, "code", None)
                is_rate_limit = (
                    error_code in (429, 503)
                    or "429" in error_str
                    or "503" in error_str
                    or "resource_exhausted" in error_str
                    or "rate_limit_exceeded" in error_str
                    or "quota" in error_str
                    or "unavailable" in error_str
                )

                if is_rate_limit and attempt < max_retries - 1:
                    self.rotate()
                    time.sleep(1)
                    # If all keys cycled, wait longer
                    if (attempt + 1) % len(self.keys) == 0:
                        logger.warning(
                            "All %d keys exhausted, waiting 30s...", len(self.keys)
                        )
                        time.sleep(30)
                    continue
                else:
                    raise


# =============================================================================
# Intent Classification
# =============================================================================

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
# Pydantic schema for constrained classifier output
# ---------------------------------------------------------------------------
class IntentClassification(BaseModel):
    """Schema enforced by Gemini's structured output."""

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
# Intent result dataclass
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
# Classifier system prompt
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
- If the user asks about changing/rebalancing their portfolio → full_analysis
- When in doubt between full_analysis and specific_metric → choose full_analysis.
  It is better to give a comprehensive answer than to guess which single metric they want.

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
    "rebalance": "full_analysis",
    "compare": "full_analysis",
    "comparison": "full_analysis",
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
            logger.warning("Fuzzy matched intent '%s' -> '%s'", raw, intent_name)
            return Intent(intent_name)
    logger.error("Could not resolve intent '%s', defaulting to general_chat", raw)
    return Intent.GENERAL_CHAT


# ---------------------------------------------------------------------------
# Classifier context builder
# ---------------------------------------------------------------------------
def _build_classifier_context(
    message: str,
    recent_history: list[dict],
    portfolio_changed: bool,
) -> str:
    """Build the context string for the classifier LLM."""
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
# Classify intent
# ---------------------------------------------------------------------------
def classify_intent(
    message: str,
    recent_history: list[dict],
    portfolio_changed: bool = False,
    client: Optional[genai.Client] = None,
) -> IntentResult:
    """
    Classify a user message into one or more of 6 intents.
    Uses Gemini 2.5 Flash-Lite with structured JSON output.

    Rate limit errors (429) are re-raised for the caller (e.g. KeyRotator)
    to handle via key rotation.

    Parameters
    ----------
    message : str
        The user's current message.
    recent_history : list[dict]
        Recent chat history, already trimmed by caller (last 3-4 exchanges).
    portfolio_changed : bool
        True if the portfolio form was just updated this turn.
    client : genai.Client | None
        Gemini client. If None, creates one from env var.

    Returns
    -------
    IntentResult
    """
    # Guard clause: empty messages
    if not message or not message.strip():
        return IntentResult(
            primary_intent=Intent.GENERAL_CHAT,
            confidence=1.0,
            reasoning="Empty message — nothing to classify",
        )

    if client is None:
        client = genai.Client()

    context = _build_classifier_context(message, recent_history, portfolio_changed)

    # Call Gemini with structured output
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
        # Let rate limit errors bubble up for key rotation
        error_str = str(e).lower()
        error_code = getattr(e, "code", None)
        is_rate_limit = (
            error_code == 429
            or "429" in error_str
            or "resource_exhausted" in error_str
            or "rate_limit_exceeded" in error_str
            or "quota" in error_str
        )
        if is_rate_limit:
            raise

        logger.error("Gemini API call failed: %s", e)
        return IntentResult(
            primary_intent=Intent.GENERAL_CHAT,
            confidence=0.0,
            reasoning=f"LLM call failed: {e}",
        )

    # Parse structured response
    try:
        data = json.loads(response.text)
    except (json.JSONDecodeError, TypeError) as e:
        logger.error("Failed to parse response: %s | raw: %s", e, response.text)
        return IntentResult(
            primary_intent=Intent.GENERAL_CHAT,
            confidence=0.0,
            reasoning=f"Response parsing failed: {e}",
        )

    # Build IntentResult with fuzzy match safety net
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


# =============================================================================
# Explanation Generation
# =============================================================================

# ---------------------------------------------------------------------------
# Explanation system prompt
# ---------------------------------------------------------------------------
EXPLAIN_SYSTEM_PROMPT = """\
You are a portfolio risk analyst assistant. Your job is to explain portfolio
risk analysis results to users in a clear, informative, and helpful way.

You will receive structured data about the user's portfolio, risk metrics,
ML predictions, and contextual information. Your response should be
conversational, insightful, and tailored to what the user asked.

## Guidelines

- Be concise but thorough. Prioritize the most important insights.
- Use plain language. Define technical terms when you first use them.
- Reference specific numbers from the data provided.
- When metrics have benchmarks, compare the user's values to them.
- When giving recommendations, explain the tradeoffs.
- Always be honest about uncertainty, especially for predictions.
- Do NOT make up data. Only reference values that were provided to you.
- Do NOT provide specific financial advice (e.g. "you should buy X").
  Instead, explain what the data shows and let the user decide.

## Response style by intent

### full_analysis (standalone — no portfolio change)
Give a comprehensive risk assessment covering:
1. Overall risk level (from NN classification) and what it means
2. Key risk drivers (which assets contribute most to risk)
3. Notable metrics — highlight anything concerning or strong
4. Trend outlook (if LSTM forecast is provided)
5. Brief recommendations for consideration

### full_analysis (comparison — portfolio changed)
Focus on what CHANGED:
1. What improved and by how much
2. What worsened and by how much
3. How the risk level shifted
4. Which assets are now the biggest risk drivers
5. Overall assessment: is this change beneficial?
Use clear before/after comparisons with specific numbers.

### specific_metric
Focus narrowly on the requested metric(s):
1. Current value and what it means
2. How it compares to benchmarks (good/bad/neutral)
3. Brief context on why it matters for this portfolio
Keep it focused — don't volunteer a full analysis unless asked.

### concept_explanation
Teach the concept clearly:
1. What it is in plain language
2. Why it matters for portfolio risk
3. How it's calculated (simplified)
4. If the user has portfolio data available, illustrate with their real numbers
Keep it educational. This is a teaching moment.

### trend_prediction
Present the forecast with appropriate caveats:
1. What the model predicts (direction + magnitude)
2. Current baseline for comparison
3. What could cause the prediction to be wrong
4. ALWAYS include uncertainty disclaimers — predictions are estimates, not guarantees

CRITICAL — interpret forecast fields correctly:
- predicted_direction / prob_up refer to the direction of PORTFOLIO RETURNS, NOT volatility.
  "Up" means the model predicts positive/increasing returns. Do NOT describe this as
  volatility going up or down.
- predicted_volatility is the forecast volatility level. Compare it to current volatility
  to describe whether risk is expected to rise or fall.
- Never conflate return direction with volatility direction — they are separate signals.

### follow_up
Be concise and direct:
1. Address exactly what the user asked about
2. Reference the previous response
3. Provide additional detail or clarification
Don't repeat the entire previous analysis.

### general_chat
Be friendly and helpful:
- Greetings: respond warmly, mention what you can do
- Capabilities: list the types of analysis available
- Off-topic: gently redirect to portfolio risk analysis

### dual-intent (secondary_intent present)
Address both requests in a single coherent response:
1. Lead with the primary intent's answer
2. Use a clear section break (e.g. "---") before the secondary answer
3. Do not repeat shared data (e.g. portfolio details, metrics already shown)
"""


# ---------------------------------------------------------------------------
# Build explanation prompt from context dict
# ---------------------------------------------------------------------------
def _build_explanation_prompt(ctx: dict) -> str:
    """Convert a context dict into a structured prompt for the explanation LLM."""
    sections = []

    # User query
    sections.append(f"## User's question\n{ctx['user_query']}")

    # Portfolio
    portfolio  = ctx["portfolio"]
    tickers    = portfolio.get("tickers", [])
    weights    = portfolio.get("weights", [])
    investment = portfolio.get("investment_amount", "N/A")
    currency   = portfolio.get("currency", "USD")
    holdings   = ", ".join(f"{t} ({w:.0%})" for t, w in zip(tickers, weights))
    sections.append(
        f"## Current portfolio\n{holdings}\n"
        f"Total investment: {currency} {investment}"
    )

    # Requested metrics (for specific_metric)
    if ctx.get("requested_metrics"):
        sections.append(
            f"## Requested metrics (focus on these)\n"
            f"{', '.join(ctx['requested_metrics'])}"
        )

    # Risk contributions
    if ctx.get("risk_contributions"):
        contrib_str = "\n".join(
            f"- {k}: {v:.1%}" if isinstance(v, float) else f"- {k}: {v}"
            for k, v in ctx["risk_contributions"].items()
        )
        sections.append(f"## Risk contributions (% of portfolio volatility)\n{contrib_str}")

    # Benchmarks
    if ctx.get("metric_benchmarks"):
        lines = []
        for k, v in ctx["metric_benchmarks"].items():
            if isinstance(v, dict):
                raw_val = v.get("value", "N/A")
                label   = v.get("label", "")
                comment = v.get("comment", "")
                val_str = f"{raw_val:.4f}" if isinstance(raw_val, float) else str(raw_val)
                line = f"- {k}: {val_str}"
                if label:
                    line += f" [{label}]"
                if comment:
                    line += f" — {comment}"
                lines.append(line)
            else:
                lines.append(f"- {k}: {v}")
        sections.append("## Metric benchmarks\n" + "\n".join(lines))

    # ML risk level
    if ctx.get("risk_level"):
        label      = ctx["risk_level"].get("label", "Unknown")
        confidence = ctx["risk_level"].get("confidence", 0)
        sections.append(
            f"## NN risk classification\n"
            f"Label: {label} (confidence: {confidence:.0%})"
        )

    # Trend forecast
    if ctx.get("trend_forecast"):
        forecast_str = "\n".join(f"- {k}: {v}" for k, v in ctx["trend_forecast"].items())
        sections.append(
            "## LSTM volatility forecast\n"
            "IMPORTANT — field definitions:\n"
            "- predicted_direction: direction of PORTFOLIO RETURNS (not volatility)."
            " 'Up' means returns are predicted to be positive/increasing.\n"
            "- prob_up: probability that portfolio returns move upward (positive).\n"
            "- predicted_volatility: the forecast volatility level (annualised).\n"
            "- confidence: model confidence in the directional prediction.\n"
            + forecast_str
        )

    # Comparison data (portfolio changed)
    if ctx.get("portfolio_changed") and ctx.get("previous_metrics"):
        prev_str = "\n".join(f"- {k}: {v}" for k, v in ctx["previous_metrics"].items())
        sections.append(f"## Previous portfolio metrics (BEFORE change)\n{prev_str}")

        if ctx.get("previous_contributions"):
            prev_contrib = "\n".join(
                f"- {k}: {v:.1%}" if isinstance(v, float) else f"- {k}: {v}"
                for k, v in ctx["previous_contributions"].items()
            )
            sections.append(f"## Previous risk contributions\n{prev_contrib}")

        if ctx.get("previous_risk_level"):
            prev_label = ctx["previous_risk_level"].get("label", "Unknown")
            prev_conf  = ctx["previous_risk_level"].get("confidence", 0)
            sections.append(
                f"## Previous NN classification\n"
                f"Label: {prev_label} (confidence: {prev_conf:.0%})"
            )

        sections.append(
            "## MODE: COMPARISON\n"
            "The user changed their portfolio. Compare BEFORE vs AFTER.\n"
            "Highlight what improved, what worsened, and the tradeoffs."
        )

    # RAG context
    if ctx.get("company_context"):
        sections.append(f"## Company context (from SEC filings)\n{ctx['company_context']}")

    if ctx.get("educational_context"):
        sections.append(f"## Educational context (from knowledge base)\n{ctx['educational_context']}")

    # Concept name
    if ctx.get("concept_name"):
        sections.append(f"## Concept to explain\n{ctx['concept_name']}")

    # Secondary intent
    secondary_intent = ctx.get("secondary_intent")
    if secondary_intent:
        parts = [f"## Secondary request\nIntent: {secondary_intent.value}"]
        if ctx.get("secondary_concept"):
            parts.append(f"Concept: {ctx['secondary_concept']}")
        if ctx.get("secondary_requested_metrics"):
            parts.append(f"Metrics: {', '.join(ctx['secondary_requested_metrics'])}")
        sections.append("\n".join(parts))

    # Chat history
    if ctx.get("chat_history"):
        history_lines = []
        for msg in ctx["chat_history"][-6:]:
            role    = msg["role"].upper()
            content = msg["content"]
            if len(content) > 300:
                content = content[:300] + "..."
            history_lines.append(f"[{role}]: {content}")
        sections.append(f"## Recent conversation\n" + "\n".join(history_lines))

    # Canonical numbers block — single authorised reference for all figures
    authorised: list[str] = []
    _metrics = ctx.get("metrics") or {}
    _benchmarks = ctx.get("metric_benchmarks") or {}
    seen_keys: set[str] = set()
    for k, v in _benchmarks.items():
        seen_keys.add(k)
        if isinstance(v, dict):
            raw = v.get("value", "N/A")
            val_str = f"{raw:.4f}" if isinstance(raw, float) else str(raw)
            authorised.append(f"{k} = {val_str}")
        elif isinstance(v, (int, float)):
            authorised.append(f"{k} = {v:.4f}")
    for k, v in _metrics.items():
        if k not in seen_keys and isinstance(v, (int, float)):
            authorised.append(f"{k} = {v:.4f}")
    _contrib = ctx.get("risk_contributions") or {}
    for k, v in _contrib.items():
        if isinstance(v, float):
            authorised.append(f"risk_contribution[{k}] = {v:.4f}")
    if ctx.get("risk_level"):
        conf = ctx["risk_level"].get("confidence", 0)
        authorised.append(f"risk_score = {float(conf):.4f}")
    if ctx.get("trend_forecast"):
        tf = ctx["trend_forecast"]
        for k, v in tf.items():
            if isinstance(v, (int, float)):
                authorised.append(f"forecast[{k}] = {v:.4f}")
    if authorised:
        sections.append(
            "## AUTHORISED NUMBERS (single source of truth)\n"
            "Every numerical value you cite MUST appear in this list. "
            "Do not round, adjust, or invent any figure.\n"
            + "\n".join(authorised)
        )

    # Grounding reminder — placed last so it is closest to generation
    sections.append(
        "## GROUNDING REQUIREMENT\n"
        "Only cite numbers that appear verbatim in the AUTHORISED NUMBERS section above. "
        "Do not use values from your training data, round to 'nicer' numbers, "
        "or fill gaps with plausible-sounding figures. "
        "If a value is not in the data provided, say so explicitly."
    )

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Post-generation hallucination check
# ---------------------------------------------------------------------------
import re as _re

def _check_numbers(response_text: str, ctx: dict) -> list[str]:
    """
    Extract all decimal numbers from the response and check each one exists
    in the context data. Returns a list of suspect values not found in context.

    Only flags numbers with at least 1 decimal place (ignores round integers
    like "5 stocks" or "4 dimensions") to reduce false positives.
    """
    # Collect all authorised floats from context
    authorised_vals: set[str] = set()

    def _add(v):
        if isinstance(v, float):
            # Store multiple representations the LLM might use
            authorised_vals.add(f"{v:.4f}")
            authorised_vals.add(f"{v:.2f}")
            authorised_vals.add(f"{v:.1f}")
            authorised_vals.add(f"{v*100:.2f}")   # percentage form
            authorised_vals.add(f"{v*100:.1f}")
            authorised_vals.add(f"{v*100:.0f}")

    for v in (ctx.get("metrics") or {}).values():
        _add(v)
    for v in (ctx.get("risk_contributions") or {}).values():
        _add(v)
    for v in (ctx.get("metric_benchmarks") or {}).values():
        if isinstance(v, dict):
            _add(v.get("value"))
        else:
            _add(v)
    if ctx.get("risk_level"):
        _add(float(ctx["risk_level"].get("confidence", 0)))
    for v in (ctx.get("trend_forecast") or {}).values():
        if isinstance(v, float):
            _add(v)
    # Also allow investment amount and portfolio weights
    p = ctx.get("portfolio") or {}
    if isinstance(p.get("investment_amount"), (int, float)):
        authorised_vals.add(str(int(p["investment_amount"])))
    for w in p.get("weights", []):
        _add(float(w))

    # Extract numbers with at least one decimal place from the response
    found = _re.findall(r"\b\d+\.\d+\b", response_text)
    suspects = [n for n in found if n not in authorised_vals]
    return suspects


# ---------------------------------------------------------------------------
# Generate explanation
# ---------------------------------------------------------------------------
def generate_explanation(
    ctx: dict,
    client: Optional[genai.Client] = None,
) -> str:
    """
    Generate a natural language explanation from a context dict.
    Uses Gemini 2.5 Flash for richer reasoning than Flash-Lite.

    Rate limit errors (429) are re-raised for key rotation handling.

    Parameters
    ----------
    ctx : dict
        Context assembled by _build_explanation_context in orchestrator.py.
        Required keys: "intent", "user_query", "portfolio".
        All other keys are optional and intent-dependent.
    client : genai.Client | None
        Gemini client. If None, creates one from env var.

    Returns
    -------
    str
        The generated explanation text.
    """
    if client is None:
        client = genai.Client()

    prompt = _build_explanation_prompt(ctx)
    print(prompt)

    try:
        response = client.models.generate_content(
            model=EXPLANATION_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=EXPLAIN_SYSTEM_PROMPT,
                temperature=0.0,  # lower = less creative drift, fewer hallucinated numbers
                max_output_tokens=8192,
                thinking_config=types.ThinkingConfig(thinking_budget=1024),
            ),
        )
    except Exception as e:
        # Re-raise rate limit and transient server errors for KeyRotator retry
        error_str = str(e).lower()
        error_code = getattr(e, "code", None)
        is_retryable = (
            error_code in (429, 503)
            or "429" in error_str
            or "503" in error_str
            or "resource_exhausted" in error_str
            or "rate_limit_exceeded" in error_str
            or "quota" in error_str
            or "unavailable" in error_str
        )
        if is_retryable:
            raise

        logger.error("Explanation generation failed: %s", e)
        return (
            "I'm sorry, I encountered an error generating the explanation. "
            "Please try again."
        )

    text = response.text

    suspects = _check_numbers(text, ctx)
    if suspects:
        logger.warning(
            "Possible hallucinated numbers in explanation (not found in context): %s",
            suspects,
        )

    return text
