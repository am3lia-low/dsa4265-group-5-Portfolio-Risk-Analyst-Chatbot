import streamlit as st

# =========================
# Imports from modules
# =========================
from ui.state import initialize_session_state, update_cache
from ui.sidebar import render_sidebar
from ui.chat import render_chat_history, render_status
from agent_tools.workflow_tools import classify_intent, route_and_execute


# === Page setup ===
st.markdown("""
<h2 style='margin-top:-30px; font-weight:700;'>📈 Portfolio Analysis Chatbot</h2>
""", unsafe_allow_html=True)

st.caption("Conversational portfolio analysis with validation, metrics, and explanations.")

# === Initialise session state ===
initialize_session_state()

# === Render sidebar & chat messages ===
render_sidebar()
render_status(st.session_state.portfolio_messages)
render_chat_history(st.session_state.chat_history)

# === Chatbox ===
user_query = st.chat_input(
    "Ask something about your portfolio...",
    disabled=not st.session_state.portfolio_ready
)

if user_query:
    # show user message
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_query
    })

    intent = classify_intent(
        message=user_query,
        recent_history=st.session_state.chat_history,
        portfolio_changed=st.session_state.portfolio_updated,
        client=None,
    )
    if intent.secondary_intent:
        print(f"intent detected {intent.primary_intent} and {intent.secondary_intent}")

    workflow = route_and_execute(
        intent,
        portfolio=st.session_state.portfolio,
        is_first_portfolio=len(st.session_state.all_portfolios) == 1,
        portfolio_changed=st.session_state.portfolio_updated,
        recent_history=st.session_state.chat_history,
        cache = st.session_state.cache,
    )
    response = workflow.content

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response,
        "metadata": {
            "intent": "intent",
            "tools_used": ["tool1"],        # to be updated
            "models_called": ["model1"]     # to be updated
        }
    })

    update_cache(
        returns_df=workflow.cache.get("returns_df"),
        metrics=workflow.cache.get("metrics"),
        risk_level=workflow.cache.get("risk_level"),
        trend_forecast=workflow.cache.get("trend_forecast"),
        rag_context=workflow.cache.get("rag_context"),
    )
    st.session_state.portfolio_updated = False

    st.rerun()