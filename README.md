# dsa4265-group-5-Portfolio-Risk-Analyst-Chatbot
Portfolio Risk Analyst Chatbot using AI agent and various tools!

## Portfolio Risk Analyst Chatbot — Architecture

```mermaid
flowchart TD

    %% ── User Interface ──────────────────────────────────────────────
    subgraph UI["Streamlit UI  (chatbot.py)"]
        A([User Query + Portfolio])
        B[classify_intent\nvia KeyRotator]
        C[route_and_execute]
        D[update_cache]
        E([Response to User])
    end

    %% ── Intent Classification ────────────────────────────────────────
    subgraph IC["Intent Classification  (agent_llm.py)"]
        F[classify_intent\nGemini 2.5 Flash-Lite]
        G["IntentResult\nprimary · secondary\nextracted_metrics · extracted_concept"]
        H[KeyRotator\n429 / 503 retry]
    end

    %% ── Orchestrator ────────────────────────────────────────────────
    subgraph OR["Orchestrator  (orchestrator.py)"]
        I[route_and_execute]

        subgraph S1["Step 1 — Market Data + Metrics\n(full_analysis · specific_metric · trend_prediction)"]
            J[fetch_price_data\nyfinance]
            K[calculate_returns]
            L[calculate_all_metrics]
            M[metric_benchmarks]
        end

        subgraph S2["Step 2 — Risk Classification\n(full_analysis only)"]
            N[current_portfolio_risk_tool\nNeural Network Classifier]
        end

        subgraph S3["Step 3 — Volatility Forecast\n(full_analysis · trend_prediction)"]
            O[future_portfolio_risk\nLSTM Model]
        end

        subgraph S4["Step 4 — RAG Retrieval\n(intent-conditional)"]
            P[_body_for_intent\n_rag_block]
            Q[retrieve_context\nVector Search]
        end

        subgraph S5["Step 5 — Explanation"]
            R[_build_explanation_context]
            S[generate_explanation\nGemini 2.5 Flash]
        end

        T[WorkflowResult\ncontent · cache]
    end

    %% ── Cache / Session State ────────────────────────────────────────
    subgraph SS["Session State  (ui/state.py)"]
        U["st.session_state.cache\nreturns_df · metrics · risk_level\ntrend_forecast · rag_context\nportfolio_hash · computed_at"]
        V["st.session_state\nchat_history · portfolio\nall_portfolios · portfolio_updated"]
    end

    %% ── Data Flow ────────────────────────────────────────────────────

    A -->|query + portfolio| B
    B -->|uses| H
    H -->|rotates keys| F
    F --> G
    G -->|IntentResult| C
    C --> I

    I -->|fetch prices| J
    J --> K
    K -->|returns_df| L
    L --> M

    M -->|metrics + benchmarks| N
    M -->|metrics| O

    N -->|risk_level + score| R
    O -->|direction · volatility · confidence| R

    I -->|intent-mapped| P
    P --> Q
    Q -->|educational_context| R

    L -->|metrics| R
    G -->|intent · concept · query| R
    V -->|portfolio · chat_history| R

    R -->|context dict| S
    S -->|LLM explanation| T

    T -->|content| E
    T -->|cache payload| D
    D --> U

    U -->|stale-check\navoids recompute| I
```

### Component Index

| Component | File | Role |
|---|---|---|
| `chatbot.py` | `chatbot.py` | Streamlit entry point, session orchestration |
| `classify_intent` | `agent_tools/workflow_tools/agent_llm.py` | Intent classification via Gemini 2.5 Flash-Lite |
| `KeyRotator` | `agent_tools/workflow_tools/agent_llm.py` | API key rotation on 429/503 errors |
| `route_and_execute` | `agent_tools/workflow_tools/orchestrator.py` | Main orchestration pipeline |
| `fetch_price_data` | `agent_tools/data_tools/` | Price history via yfinance |
| `calculate_returns` | `agent_tools/data_tools/` | Returns computation |
| `calculate_all_metrics` | `agent_tools/quant_tools/` | Vol, VaR, CVaR, Sharpe, Sortino, beta, HHI, etc. |
| `metric_benchmarks` | `agent_tools/quant_tools/` | good / neutral / poor labels per metric |
| `current_portfolio_risk_tool` | `agent_tools/ml_risk_tools/` | Neural network risk level + score |
| `future_portfolio_risk` | `agent_tools/ml_risk_tools/` | LSTM volatility direction forecast |
| `retrieve_context` | `agent_tools/rag_tools/` | Vector search over knowledge base |
| `generate_explanation` | `agent_tools/workflow_tools/agent_llm.py` | Final LLM response via Gemini 2.5 Flash |
| `update_cache` | `ui/state.py` | Persists computed data to session state |

### Intent Routing

| Intent | Step 1 | Step 2 | Step 3 | Step 4 (RAG) |
|---|:---:|:---:|:---:|:---:|
| `full_analysis` | yes | yes | yes | yes |
| `specific_metric` | yes | — | — | — |
| `trend_prediction` | yes | — | yes | yes |
| `concept_explanation` | — | — | — | yes |
| `follow_up` | — | — | — | conditional |
| `general_chat` | — | — | — | — |
