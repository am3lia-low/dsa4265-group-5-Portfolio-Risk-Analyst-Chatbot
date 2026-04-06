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

    st.session_state.full_chat_history.append({"role": "user", "content": user_query})

    intent = classify_intent(
        message=user_query,
        recent_history=st.session_state.chat_history,
        portfolio_changed=st.session_state.portfolio_updated,
        client=None,
    )
    print(intent)

    workflow = route_and_execute(
        intent,
        portfolio=st.session_state.current_portfolio,
        portfolio_changed=st.session_state.portfolio_updated,
        recent_history=st.session_state.chat_history,
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

    st.session_state.full_chat_history.append({
        "role": "assistant",
        "content": response,
        "metadata": {
            "intent": "intent",
            "tools_used": ["tool1"],        # to be updated
            "models_called": ["model1"]     # to be updated
        }
    })

    update_cache()  # to be updated
    st.session_state.portfolio_updated = False

    st.rerun()