import streamlit as st
from collections import defaultdict
from ui.state import update_status_message, clear_cache, snapshot_portfolio
from agent_tools.data_tools.valid_tickers import valid_tickers
from agent_tools.data_tools.valid_weights import valid_weights

# === CSS (display) ===
st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            min-width: 350px;
        }

        section[data-testid="stSidebar"] > div {
            min-width: 350px;
        }

    </style>
    """,
    unsafe_allow_html=True
)
# ===

def render_sidebar():
    st.sidebar.header("Your Portfolio")

    investment_input = st.sidebar.text_input(
        "Investment Amount",
        value=str(st.session_state.portfolio["investment_amount"] or 0.0),
        disabled=st.session_state.portfolio_ready,
    )

    # Convert to float safely
    try:
        investment_amount = float(investment_input)
        if investment_amount < 0:
            st.sidebar.error("Value must be ≥ 0")
            investment_amount = 0.0
    except ValueError:
        st.sidebar.error("Please enter a valid number")
        investment_amount = 0.0

    st.sidebar.subheader("Holdings")

    portfolio = []

    for i in range(st.session_state.num_stocks):
        col1, col2 = st.sidebar.columns(2)

        default_ticker = ""
        default_weight = 0.0

        if i < len(st.session_state.portfolio["tickers"]):
            default_ticker = st.session_state.portfolio["tickers"][i]

        if i < len(st.session_state.portfolio["weights"]):
            default_weight = st.session_state.portfolio["weights"][i]

        with col1:
            ticker = st.text_input(
                f"Ticker {i+1}",
                value=default_ticker,
                key=f"ticker_{i}",
                disabled=st.session_state.portfolio_ready,
            )

        with col2:
            weight = st.number_input(
                f"Weight {i+1} (%)",
                min_value=0.0,
                max_value=100.0,
                step=1.0,
                value=float(default_weight),
                key=f"weight_{i}",
                disabled=st.session_state.portfolio_ready,
            )
        
        portfolio.append({
            "ticker": ticker.strip().upper(),
            "weight": weight
        })

    combined = defaultdict(float)

    for holding in portfolio:
        ticker = holding["ticker"]
        weight = holding["weight"]

        if ticker:
            combined[ticker] += weight

    portfolio = [
        {"ticker": ticker, "weight": weight}
        for ticker, weight in combined.items()
    ]
    portfolio = sorted(portfolio, key=lambda x: x["weight"], reverse=True)

    colA, colB = st.sidebar.columns(2)

    with colA:
        if st.button("Add more", key="add_stock", disabled=st.session_state.portfolio_ready):
            st.session_state.num_stocks += 1
            st.rerun()

    with colB:
        if st.button("Remove last", key="remove_stock", disabled=st.session_state.portfolio_ready) and st.session_state.num_stocks > 1:
            st.session_state.num_stocks -= 1
            st.rerun()

    if st.session_state.portfolio_ready:
        if st.sidebar.button("Modify Portfolio"):
            st.session_state.portfolio_ready = False
            update_status_message()
            st.rerun()

    analyze_clicked = st.sidebar.button("Analyze Portfolio")


    proposed_portfolio = {
        "tickers": [holding["ticker"] for holding in portfolio],
        "weights": [(holding["weight"] / 100) for holding in portfolio],
        "investment_amount": investment_amount
    }

    if analyze_clicked:
        is_valid, msg = portfolio_checker(portfolio, investment_amount)

        if is_valid:
            st.session_state.portfolio_ready = True

            if portfolios_are_equal(proposed_portfolio, st.session_state.portfolio):
                st.session_state.portfolio_updated = False
            else:
                if st.session_state.portfolio["tickers"] and st.session_state.cache["metrics"] is not None:
                    snapshot_portfolio()
                    
                clear_cache()
                st.session_state.portfolio = proposed_portfolio
                st.session_state.all_portfolios.append(proposed_portfolio.copy())
                st.session_state.portfolio_updated = True

            update_status_message()
            st.rerun()

        else:
            st.sidebar.error(msg)
            st.session_state.portfolio_ready = False
            st.session_state.portfolio_updated = False
            update_status_message()

        return portfolio, investment_amount



def portfolio_checker(portfolio, investment_amount):
    tickers = [holding["ticker"] for holding in portfolio]
    weights = [holding["weight"] for holding in portfolio]

    tickers_ok, bad_tickers = valid_tickers(tickers)
    if not tickers_ok:
        if bad_tickers == ["<empty>"]:
            return False, "⚠️ Please fill in a valid ticker before proceeding."
        return False, f"⚠️ Invalid ticker(s): {', '.join(bad_tickers)}."

    # check weights (using helper)
    weights_ok, weight_msg = valid_weights(weights)
    if not weights_ok:
        return False, f"⚠️ Weights do not add up to 100%. Please input the correct weights."

    # enforce percentages only (sum ≈ 100)
    total = sum(weights)
    if not (99.99 <= total <= 100.01):   # small tolerance
        return False, "⚠️ Weights do not add up to 100%. Please input the correct weights."

    if investment_amount <= 0:
        return False, "⚠️ Please ensure that Investment Amount is more than 0."

    return True, "You have inputted a valid portfolio. Feel free to ask me questions!"


def portfolios_are_equal(p1, p2):
    if p1 is None or p2 is None:
        return False

    return (
        p1["tickers"] == p2["tickers"]
        and p1["weights"] == p2["weights"]
        and p1["investment_amount"] == p2["investment_amount"]
    )