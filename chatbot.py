import streamlit as st

# =========================
# Imports from modules
# =========================
from ui.state import initialize_session_state
from ui.sidebar import render_sidebar
from ui.chat import render_chat_history
from agent_tools.workflow_tools import classify_intent, route_and_execute


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
        client=None,
    )
    print(intent)

    workflow = route_and_execute(
        intent,
        portfolio=st.session_state.current_portfolio,
        portfolio_changed=st.session_state.portfolio_updated,
        recent_history=st.session_state.messages,
    )
    response = workflow.content

    st.session_state.messages.append({
        "role": "assistant",
        "type": "answer",
        "portfolio_id": st.session_state.current_portfolio["id"],
        "content": response,
    })

    # st.session_state.all_assistant_outputs.append(response)
    st.session_state.portfolio_updated = False

    print(st.session_state.messages)
    print(st.session_state.all_portfolios)

    st.rerun()