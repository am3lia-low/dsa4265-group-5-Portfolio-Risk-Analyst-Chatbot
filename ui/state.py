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
        "full_chat_history": [],
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
    returns_df=None,
    metrics=None,
    risk_level=None,
    trend_forecast=None,
    rag_context=None
):
    if "cache" not in st.session_state:
        st.session_state.cache = {}

    st.session_state.cache["portfolio_hash"] = compute_portfolio_hash()
    st.session_state.cache["computed_at"] = datetime.now().isoformat(timespec="seconds")

    if returns_df is not None:
        st.session_state.cache["returns_df"] = returns_df
    if metrics is not None:
        st.session_state.cache["metrics"] = metrics
    if risk_level is not None:
        st.session_state.cache["risk_level"] = risk_level
    if trend_forecast is not None:
        st.session_state.cache["trend_forecast"] = trend_forecast
    if rag_context is not None:
        st.session_state.cache["rag_context"] = rag_context


def empty_cache():
    return {
        "portfolio_hash": None,
        "returns_df": None,
        "metrics": None,
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


def add_portfolio_summary_message():
    st.session_state.full_chat_history.append({
        "role": "assistant",
        "content": summary_message()
    })


def summary_message():
    cache = st.session_state.get("cache", {})
    history = st.session_state.get("portfolio_history", [])
    portfolio = st.session_state.get("portfolio", {})

    metrics = cache.get("metrics")
    if metrics is None:
        return "No results stored for current portfolio. Please ask a question first."

    tickers = portfolio.get("tickers", [])
    weights = portfolio.get("weights", [])
    investment = portfolio.get("investment_amount", "N/A")

    all_metrics = metrics.get("all_metrics", {})
    risk_contribution = all_metrics.get("risk_contribution", {})
    risk_level = cache.get("risk_level")

    lines = []
    lines.append("## Portfolio Summary")
    lines.append("")
    lines.append("### Current Portfolio")
    lines.append(f"- **Tickers:** {', '.join(tickers) if tickers else 'N/A'}")
    lines.append(f"- **Weights:** {', '.join(str(w) for w in weights) if weights else 'N/A'}")
    lines.append(f"- **Investment:** ${investment:,.2f}" if isinstance(investment, (int, float)) else f"- **Investment:** {investment}")
    lines.append("")
    lines.append("### Current Results")
    lines.append(f"- **Portfolio Volatility:** {all_metrics.get('portfolio_volatility', 0):.2%}")
    lines.append(f"- **Sharpe Ratio:** {all_metrics.get('sharpe_ratio', 0):.3f}")
    lines.append(f"- **VaR (95%):** {all_metrics.get('var_95', 0):.2%}")
    lines.append(f"- **Max Drawdown:** {all_metrics.get('max_drawdown', 0):.2%}")
    lines.append(f"- **Beta:** {all_metrics.get('beta', 0):.3f}")
    lines.append(f"- **Risk Level:** {risk_level if risk_level is not None else 'Not available'}")

    if risk_contribution:
        top_rc = max(risk_contribution.items(), key=lambda x: x[1])
        lines.append(f"- **Largest Risk Contributor:** {top_rc[0]} ({top_rc[1]:.2%})")

    if history:
        lines.append("")
        lines.append("### Previous Portfolio Snapshots")

        for i, snap in enumerate(history, 1):
            old_portfolio = snap.get("portfolio", {})
            old_metrics = snap.get("metrics") or {}
            old_all_metrics = old_metrics.get("all_metrics", {})
            old_risk_level = snap.get("risk_level")
            old_timestamp = snap.get("timestamp", "Unknown time")

            old_tickers = old_portfolio.get("tickers", [])
            old_weights = old_portfolio.get("weights", [])
            old_investment = old_portfolio.get("investment_amount", "N/A")

            lines.append(f"**Portfolio {i}** ({old_timestamp})")
            lines.append(f"- Tickers: {', '.join(old_tickers) if old_tickers else 'N/A'}")
            lines.append(f"- Weights: {', '.join(str(w) for w in old_weights) if old_weights else 'N/A'}")
            lines.append(f"- Investment: ${old_investment:,.2f}" if isinstance(old_investment, (int, float)) else f"- Investment: {old_investment}")
            lines.append(f"- Volatility: {old_all_metrics.get('portfolio_volatility', 0):.2%}")
            lines.append(f"- Sharpe Ratio: {old_all_metrics.get('sharpe_ratio', 0):.3f}")
            lines.append(f"- Risk Level: {old_risk_level if old_risk_level is not None else 'Not available'}")
            lines.append("")
    
    return "\n".join(lines)