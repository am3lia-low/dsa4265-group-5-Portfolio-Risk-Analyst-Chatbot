import streamlit as st

# =========================
# Imports from modules
# =========================
from ui.state import initialize_session_state
from ui.sidebar import render_sidebar
from ui.chat import render_chat_history
from agent_tools.workflow_tools.intent_classification import classify_intent


# === Page setup ===
st.markdown("""
<h2 style='margin-top:-30px; font-weight:700;'>📈 Portfolio Analysis Chatbot</h2>
""", unsafe_allow_html=True)

st.caption("Conversational portfolio analysis with validation, metrics, and explanations.")

# === Initialise session state ===
initialize_session_state()

# === Render sidebar ===
render_sidebar()
render_chat_history(st.session_state.messages)

# === Chatbox ===
user_query = st.chat_input(
    "Ask something about your portfolio...",
    disabled=not st.session_state.portfolio_ready
)

if user_query:
    # show user message
    st.session_state.messages.append({
        "role": "user",
        "type": "question",
        "portfolio_id": st.session_state.current_portfolio["id"],
        "content": user_query
    })

    st.session_state.all_user_inputs.append(user_query)
    print("INPUTS:")
    print(st.session_state.all_user_inputs)
    print(st.session_state.current_portfolio)
    print(st.session_state.messages)

    intent = classify_intent(
        message=user_query,
        recent_history=st.session_state.messages,
        portfolio_changed=st.session_state.portfolio_updated,
        is_first_portfolio=(len(st.session_state.all_portfolios) == 1),
        client=None,
    )
    print(intent)

    st.session_state.messages.append({
        "role": "assistant",
        "type": "answer",
        "portfolio_id": st.session_state.current_portfolio["id"],
        "content": "insert response here" #response
    })

    # st.session_state.all_assistant_outputs.append(response)
    st.session_state.portfolio_updated = False

    print(st.session_state.messages)
    print(st.session_state.all_portfolios)

    st.rerun()