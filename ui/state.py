import streamlit as st
from datetime import datetime
import hashlib
import json

def initialize_session_state():
    defaults = {
        "portfolio": {
            "tickers": [],
            "weights": [],
            "investment_amount": None
        },
        "all_portfolios": [],
        "num_stocks": 1,
        "portfolio_ready": False,
        "chat_history": [],
        "cache": empty_cache(),
        "portfolio_messages": [],
        "portfolio_history": [],
        "portfolio_updated": False
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if "plots" not in st.session_state:
        st.session_state.plots = []

    if not st.session_state.portfolio_messages:
        st.session_state.portfolio_messages.append({
            "role": "assistant",
            "content": "Please fill in your portfolio before asking me questions!"
        })
    
    if not st.session_state.chat_history:
        st.session_state.chat_history = []


def update_status_message():
    if st.session_state.portfolio_ready:
        msg = "You have inputted a valid portfolio. Feel free to ask me questions!"
    else:
        msg = "Please fill in your portfolio before asking me questions!"

    for message in st.session_state.portfolio_messages:
        message["content"] = msg
        return

    st.session_state.portfolio_messages.insert(0, {
        "role": "assistant",
        "content": msg
    })


def compute_portfolio_hash(portfolio=None):
    if portfolio is None:
        portfolio = st.session_state.portfolio

    tickers = portfolio.get("tickers", [])
    weights = portfolio.get("weights", [])

    pairs = sorted(zip(tickers, weights), key=lambda x: x[0])
    payload = json.dumps(pairs, separators=(",", ":"), sort_keys=True)

    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def update_cache(
    price_data=None,
    returns_df=None,
    cov_matrix=None,
    metrics=None,
    risk_contributions=None,
    risk_level=None,
    trend_forecast=None,
    rag_context=None
):
    st.session_state.cache = {
        "portfolio_hash": compute_portfolio_hash(),
        "returns_df": returns_df, # i removed price data cos i think returns should suffice!
        "cov_matrix": cov_matrix,
        "metrics": metrics,
        "risk_contributions": risk_contributions,
        "risk_level": risk_level,
        "trend_forecast": trend_forecast,
        "rag_context": rag_context,
        "computed_at": datetime.now().isoformat(timespec="seconds")
    }


def empty_cache():
    return {
        "portfolio_hash": None,
        "price_data": None,
        "returns_df": None,
        "cov_matrix": None,
        "metrics": None,
        "risk_contributions": None,
        "risk_level": None,
        "trend_forecast": None,
        "rag_context": None,
        "computed_at": None
    }


def clear_cache():
    st.session_state.cache = empty_cache()


def snapshot_portfolio():
    snapshot = {
        "portfolio": st.session_state.portfolio,
        "metrics": st.session_state.cache.get("metrics"),
        "risk_contributions": st.session_state.cache.get("risk_contributions"),
        "risk_level": st.session_state.cache.get("risk_level"),
        "trend_forecast": st.session_state.cache.get("trend_forecast"),
        "timestamp": datetime.now().isoformat(timespec="seconds")
    }

    history = st.session_state.portfolio_history
    history.insert(0, snapshot)

    # keep only newest 5
    st.session_state.portfolio_history = history[:5]