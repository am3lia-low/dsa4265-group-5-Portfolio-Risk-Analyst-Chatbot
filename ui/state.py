import streamlit as st

def initialize_session_state():
    defaults = {
        "portfolio": {
            "tickers": [],
            "weights": [],
            "investment_amount": None,
            "currency": "USD",
            "id": 1
        },
        "all_portfolios": [],
        "current_portfolio": None,
        "portfolio_version": 1,
        "num_stocks": 1,
        "portfolio_ready": False,
        "messages": [],
        "all_user_inputs": [],
        "all_assistant_outputs": [],
        "last_validation": None,
        "all_intent": [],
        "all_result": []
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    if "current_portfolio" not in st.session_state:
        st.session_state.current_portfolio = []

    if "plots" not in st.session_state:
        st.session_state.plots = []

    if not st.session_state.messages:
        st.session_state.messages.append({
            "role": "assistant",
            "type": "status",
            "content": "Please fill in your portfolio before asking me questions!"
        })



def update_status_message():
    if st.session_state.portfolio_ready:
        msg = "You have inputted a valid portfolio. Feel free to ask me questions!"
    else:
        msg = "Please fill in your portfolio before asking me questions!"

    for message in st.session_state.messages:
        if message.get("type") == "status":
            message["content"] = msg
            return

    st.session_state.messages.insert(0, {
        "role": "assistant",
        "type": "status",
        "content": msg
    })