"""
kb1_generate_tickers.py 
===============================
Generates HTML + TXT knowledge-base files for any yfinance-resolvable ticker.
- Generate_tickers() accepts an optional `tickers` dict so callers can pass in dynamically-resolved tickers.
- A helper 'build_ticker_meta()' turns a raw symbol into a meta dict by querying yfinance for sector/type info.
- Convert_tickers_into_txt() accepts an optional html_folder so you can point it at a subset of files.
"""

import os
import re
import json
import datetime

import yfinance as yf
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore", message="Converting to PeriodArray/Index representation")

# ── Default hardcoded 20 tickers (unchanged) ─────────────────────────────────
OUTPUT_DIR_HTML = "knowledge_base/kb1_tickers/html"
OUTPUT_DIR_JSON = "knowledge_base/kb1_tickers/json"

DEFAULT_TICKERS = {
    "AAPL":  {"name": "Apple Inc.",                         "type": "equity"},
    "MSFT":  {"name": "Microsoft Corp.",                    "type": "equity"},
    "GOOGL": {"name": "Alphabet Inc. Class A",              "type": "equity"},
    "AMZN":  {"name": "Amazon.com Inc.",                    "type": "equity"},
    "NVDA":  {"name": "NVIDIA Corp.",                       "type": "equity"},
    "META":  {"name": "Meta Platforms Inc.",                "type": "equity"},
    "JPM":   {"name": "JPMorgan Chase & Co.",               "type": "equity"},
    "BAC":   {"name": "Bank of America Corp.",              "type": "equity"},
    "GS":    {"name": "Goldman Sachs Group Inc.",           "type": "equity"},
    "V":     {"name": "Visa Inc.",                          "type": "equity"},
    "TSLA":  {"name": "Tesla Inc.",                         "type": "equity"},
    "WMT":   {"name": "Walmart Inc.",                       "type": "equity"},
    "COST":  {"name": "Costco Wholesale Corp.",             "type": "equity"},
    "XOM":   {"name": "Exxon Mobil Corp.",                  "type": "equity"},
    "CVX":   {"name": "Chevron Corp.",                      "type": "equity"},
    "CAT":   {"name": "Caterpillar Inc.",                   "type": "equity"},
    "BA":    {"name": "Boeing Co.",                         "type": "equity"},
    "SPY":   {"name": "SPDR S&P 500 ETF",                  "type": "etf"},
    "GLD":   {"name": "SPDR Gold Shares",                  "type": "commodity_etf"},
    "TLT":   {"name": "iShares 20+ Yr Treasury Bond ETF",  "type": "bond_etf"},
}

SECTIONS = [
    "Description",
    "Fundamentals",
    "Price & Momentum",
    "Historical — last 60 trading days",
    "Weekly breakdown",
    "Earnings & Analyst Coverage",
]

OUTPUT_TXT = "output/kb1_tickers_processed.txt"

# ── Type detection ────────────────────────────────────────────────────────────

# Map yfinance quoteType → our internal type string
_QUOTE_TYPE_MAP = {
    "EQUITY":       "equity",
    "ETF":          "etf",
    "MUTUALFUND":   "etf",
    "INDEX":        "etf",
    "FUTURE":       "etf",
    "CRYPTOCURRENCY": "equity",  # treat crypto like equity for HTML sections
}

# Map yfinance category keywords → more specific etf sub-type
_ETF_CATEGORY_MAP = {
    "gold":     "commodity_etf",
    "commodit": "commodity_etf",
    "bond":     "bond_etf",
    "treasury": "bond_etf",
    "fixed":    "bond_etf",
}


def build_ticker_meta(ticker: str) -> dict | None:
    """
    Query yfinance to determine the asset name and type for any ticker.
    Returns a meta dict compatible with DEFAULT_TICKERS, or None on failure.

    Parameters
    ----------
    ticker : str
        Upper-case ticker symbol, e.g. "LULU"

    Returns
    -------
    dict | None
        {"name": "Lululemon Athletica Inc.", "type": "equity"}
    """
    try:
        t    = yf.Ticker(ticker)
        info = t.info

        # Bail early if yfinance returned an empty / error response
        if not info or info.get("regularMarketPrice") is None and \
                       info.get("currentPrice") is None and \
                       info.get("navPrice") is None and \
                       info.get("previousClose") is None:
            print(f"  {ticker}: no market data returned by yfinance — skipping")
            return None

        name = (
            info.get("shortName")
            or info.get("longName")
            or info.get("displayName")
            or ticker
        )

        quote_type = info.get("quoteType", "EQUITY").upper()
        asset_type = _QUOTE_TYPE_MAP.get(quote_type, "equity")

        # Refine ETF sub-type by category name
        if asset_type == "etf":
            category = (info.get("category") or "").lower()
            for kw, sub in _ETF_CATEGORY_MAP.items():
                if kw in category:
                    asset_type = sub
                    break

        return {"name": name, "type": asset_type}

    except Exception as e:
        print(f"  ⚠  {ticker}: meta lookup failed — {e}")
        return None


# ── Helpers ────────────────────────────────────────

def safe(val, fmt=None, na="N/A"):
    try:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return na
        if fmt == "pct":
            return f"{val * 100:.2f}%"
        if fmt == "pct_direct":
            return f"{val:.2f}%"
        if fmt == "B":
            return f"${val / 1e9:.2f}B"
        if fmt == "M":
            return f"${val / 1e6:.2f}M"
        if fmt == "price":
            return f"${val:,.2f}"
        if fmt == "int":
            return f"{int(val):,}"
        if fmt == "2f":
            return f"{val:.2f}"
        return str(val)
    except Exception:
        return na


def pct_change(start, end):
    try:
        if start and end and start != 0:
            chg = (end - start) / abs(start) * 100
            sign = "+" if chg >= 0 else ""
            return f"{sign}{chg:.2f}%"
    except Exception:
        pass
    return "N/A"


def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss
    return (100 - (100 / (1 + rs))).iloc[-1]


def sma_signal(price_raw, sma_raw, label):
    try:
        pct = (price_raw - sma_raw) / sma_raw * 100
        direction = "above" if price_raw >= sma_raw else "below"
        return f"{pct:+.1f}% {direction} its {label} of {safe(sma_raw, 'price')}"
    except Exception:
        return f"compared to its {label} of {safe(sma_raw, 'price')}"


def rsi_signal(rsi_raw):
    try:
        if rsi_raw >= 70:
            return f"an RSI (14-day) of {rsi_raw:.2f}, which is in overbought territory (above 70)"
        elif rsi_raw <= 30:
            return f"an RSI (14-day) of {rsi_raw:.2f}, which is in oversold territory (below 30)"
        else:
            return f"an RSI (14-day) of {rsi_raw:.2f}, which is in neutral territory"
    except Exception:
        return f"an RSI (14-day) of {safe(rsi_raw, '2f')}"


def de_signal(de_raw):
    try:
        if de_raw > 200:   return " — very high leverage"
        elif de_raw > 100: return " — elevated but common for large-caps with buyback programmes"
        elif de_raw > 50:  return " — moderate leverage"
        else:              return " — conservative leverage"
    except Exception:
        return ""


def cr_signal(cr_raw):
    try:
        if cr_raw >= 2:  return " (strong short-term liquidity)"
        elif cr_raw >= 1: return " (adequate short-term liquidity)"
        else:            return " (current liabilities exceed current assets — watch liquidity)"
    except Exception:
        return ""


def beta_signal(beta_raw):
    try:
        if beta_raw > 1.5:   return f" — high volatility, moves roughly {beta_raw:.1f}x the market on average"
        elif beta_raw > 1.0: return " — moderately more volatile than the broader market"
        elif beta_raw > 0:   return " — less volatile than the broader market"
        else:                return " — moves inversely to the broader market"
    except Exception:
        return ""


def html_wrapper(ticker, asset_name, asset_type, body_html, updated):
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{ticker} — {asset_name} | RAG Knowledge Base (Tickers) </title>
<style>
  body    {{ font-family: system-ui, sans-serif; font-size: 14px;
            line-height: 1.7; color: #1a1a1a; max-width: 820px;
            margin: 2rem auto; padding: 0 1rem; }}
  h1      {{ font-size: 22px; margin: 0 0 4px; }}
  h2      {{ font-size: 16px; font-weight: 600; margin: 2rem 0 6px;
            border-bottom: 1px solid #e0e0e0; padding-bottom: 4px; }}
  h3      {{ font-size: 14px; font-weight: 600; margin: 1.2rem 0 4px; color: #333; }}
  .meta-bar {{ font-size: 12px; color: #666; margin-bottom: 2rem; }}
  ol, ul  {{ padding-left: 1.4rem; margin: 6px 0; }}
  li      {{ margin: 6px 0; }}
  .meta   {{ color: #888; font-size: 12px; }}
  section {{ margin-bottom: 2rem; }}
  p       {{ margin: 0.6rem 0; }}
</style>
</head>
<body>
<h1>{ticker} — {asset_name}</h1>
<div class="meta-bar">
  Type: {asset_type} &nbsp;|&nbsp; Generated: {updated}
</div>
{body_html}
</body>
</html>
"""


# ── Section builders ────────────────────────────────

def section_description(info):
    desc = (
        info.get("longBusinessSummary")
        or info.get("description")
        or "No description available."
    )
    if len(desc) > 520:
        cutoff = desc[:500].rfind(".")
        desc = desc[:cutoff + 1] if cutoff > 0 else desc[:500]
    return f'<section id="description"><h2>Description</h2><p>{desc}</p></section>'


def section_fundamentals_equity(info, ticker):
    mc       = safe(info.get("marketCap"), "B")
    pe       = safe(info.get("trailingPE"), "2f")
    fpe      = safe(info.get("forwardPE"), "2f")
    eps      = safe(info.get("trailingEps"), "price")
    rev      = safe(info.get("totalRevenue"), "B")
    ni       = safe(info.get("netIncomeToCommon"), "B")
    gm       = safe(info.get("grossMargins"), "pct")
    om       = safe(info.get("operatingMargins"), "pct")
    pm       = safe(info.get("profitMargins"), "pct")
    de_fmt   = safe(info.get("debtToEquity"), "2f")
    cr_fmt   = safe(info.get("currentRatio"), "2f")
    dy       = safe(info.get("dividendYield"), "pct_direct")
    pr       = safe(info.get("payoutRatio"), "pct")
    sector   = info.get("sector", "N/A")
    industry = info.get("industry", "N/A")
    name     = info.get("shortName", ticker)

    de_note = de_signal(info.get("debtToEquity"))
    cr_note = cr_signal(info.get("currentRatio"))

    return f"""<section id="fundamentals"><h2>Fundamentals</h2>
<p>{ticker} ({name}) operates in the {sector} sector, {industry} industry,
with a market capitalisation of {mc}.</p>

<p>{ticker} has a trailing twelve-month P/E ratio of {pe} and a forward P/E of {fpe}.
Earnings per share (EPS) for the trailing twelve months is {eps}.
Revenue for the trailing twelve months is {rev}, with net income of {ni}.</p>

<p>{ticker}'s gross margin is {gm}, operating margin is {om},
and net profit margin is {pm}.</p>

<p>{ticker} has a debt-to-equity ratio of {de_fmt}{de_note},
and a current ratio of {cr_fmt}{cr_note}.
The dividend yield is {dy} with a payout ratio of {pr}.</p>
</section>"""


def section_fundamentals_etf(info, ticker):
    category = info.get("category", "N/A")
    family   = info.get("fundFamily", "N/A")
    er       = safe(info.get("annualReportExpenseRatio"), "pct")
    aum      = safe(info.get("totalAssets"), "B")
    vol      = safe(info.get("averageVolume"), "int")
    nav      = safe(info.get("navPrice"), "price")
    ytd      = safe(info.get("ytdReturn"), "pct")
    ret_3yr  = safe(info.get("threeYearAverageReturn"), "pct")
    ret_5yr  = safe(info.get("fiveYearAverageReturn"), "pct")
    name     = info.get("shortName", ticker)

    return f"""<section id="profile"><h2>Fund Profile</h2>
<p>{ticker} ({name}) is a {category} fund managed by {family},
with total assets under management of {aum}.</p>

<p>{ticker} charges an annual expense ratio of {er}.
The fund's NAV is {nav} and its average daily trading volume is {vol} shares.</p>

<p>{ticker}'s year-to-date return is {ytd}.
The 3-year average annual return is {ret_3yr}
and the 5-year average annual return is {ret_5yr}.</p>
</section>"""


def section_price_momentum(info, hist, ticker):
    prices = hist["Close"]
    curr   = prices.iloc[-1]

    sma50_raw  = prices.rolling(50).mean().iloc[-1]  if len(prices) >= 50  else None
    sma200_raw = prices.rolling(200).mean().iloc[-1] if len(prices) >= 200 else None
    rsi_raw    = compute_rsi(prices) if len(prices) > 14 else None
    beta_raw   = info.get("beta")

    high52  = safe(info.get("fiftyTwoWeekHigh"), "price")
    low52   = safe(info.get("fiftyTwoWeekLow"), "price")
    avg_vol = safe(info.get("averageVolume"), "int")
    beta_fmt = safe(beta_raw, "2f")

    def ret(days):
        if len(prices) >= days:
            return pct_change(prices.iloc[-days], curr)
        return "N/A"

    sma50_desc  = sma_signal(curr, sma50_raw,  "50-day SMA")  if sma50_raw  else "50-day SMA unavailable"
    sma200_desc = sma_signal(curr, sma200_raw, "200-day SMA") if sma200_raw else "200-day SMA unavailable"
    rsi_desc    = rsi_signal(rsi_raw) if rsi_raw is not None else "RSI unavailable"
    beta_note   = beta_signal(beta_raw) if beta_raw is not None else ""

    return f"""<section id="price-momentum"><h2>Price &amp; Momentum</h2>
<p>{ticker}'s current price is {safe(curr, 'price')},
within a 52-week range of {low52} to {high52}.</p>

<p>{ticker}'s price returns are {ret(30)} over the past 30 days,
{ret(90)} over the past 90 days,
and {ret(252)} over the past 12 months.</p>

<p>{ticker} is currently {sma50_desc}, and {sma200_desc}.</p>

<p>{ticker} has {rsi_desc}.
Beta relative to the S&P 500 is {beta_fmt}{beta_note}.
Average daily trading volume is {avg_vol} shares.</p>
</section>"""


def section_earnings(info, ticker):
    consensus  = info.get("recommendationKey", "N/A").upper()
    target     = safe(info.get("targetMeanPrice"), "price")
    n_analysts = safe(info.get("numberOfAnalystOpinions"), "int")
    curr_price = info.get("currentPrice") or info.get("navPrice")

    ed = info.get("earningsDate")
    if isinstance(ed, list) and ed:
        next_earnings = str(ed[0])[:10]
    elif ed:
        next_earnings = str(ed)[:10]
    else:
        next_earnings = "not yet announced"

    upside_note = ""
    try:
        t_val  = float(str(target).replace("$", "").replace(",", ""))
        c_val  = float(curr_price)
        upside = (t_val - c_val) / c_val * 100
        word   = "upside" if upside >= 0 else "downside"
        upside_note = f", implying {abs(upside):.1f}% {word} from the current price"
    except Exception:
        pass

    consensus_map = {
        "STRONG_BUY":  "a strong buy — analysts are broadly bullish",
        "BUY":         "a buy — analysts lean bullish overall",
        "HOLD":        "a hold — analysts are broadly neutral",
        "SELL":        "a sell — analysts lean bearish overall",
        "STRONG_SELL": "a strong sell — analysts are broadly bearish",
    }
    consensus_desc = consensus_map.get(consensus, f"a {consensus} rating")

    return f"""<section id="analyst"><h2>Earnings &amp; Analyst Coverage</h2>
<p>{ticker}'s next earnings release date is {next_earnings}.</p>

<p>{ticker} carries {consensus_desc} among {n_analysts} analysts.
The mean analyst price target is {target}{upside_note}.</p>
</section>"""


def section_holdings_etf(info, ticker):
    holdings = info.get("holdings", [])
    if not holdings:
        return ""
    sentences = []
    for i, h in enumerate(holdings[:10], 1):
        symbol    = h.get("symbol", "?")
        hold_name = h.get("holdingName", "")
        pct       = safe(h.get("holdingPercent"), "pct")
        name_part = f" ({hold_name})" if hold_name else ""
        sentences.append(
            f"{ticker}'s #{i} holding is {symbol}{name_part}, representing {pct} of the fund."
        )
    return (
        f'<section id="holdings"><h2>Top Holdings</h2>'
        f'<p>{"  ".join(sentences)}</p></section>'
    )


def section_historical_60d(hist, ticker):
    if hist.empty or len(hist) < 10:
        return (
            f'<section id="historical-60d">'
            f'<h2>Historical — last 60 trading days</h2>'
            f'<p>Insufficient data for {ticker}.</p></section>'
        )

    df = hist.tail(60).copy()
    df.index = pd.to_datetime(df.index)

    close  = df["Close"]
    volume = df["Volume"]
    high   = df["High"]
    low    = df["Low"]
    opens  = df["Open"]

    period_open   = opens.iloc[0]
    period_close  = close.iloc[-1]
    period_high   = high.max()
    period_low    = low.min()
    period_return = (period_close - period_open) / period_open * 100
    period_start  = df.index[0].strftime("%b %d, %Y")
    period_end    = df.index[-1].strftime("%b %d, %Y")

    roll_max     = close.cummax()
    drawdown     = (close - roll_max) / roll_max * 100
    max_drawdown = drawdown.min()

    avg_vol      = volume.mean()
    peak_vol_day = volume.idxmax().strftime("%b %d")
    peak_vol     = volume.max()
    vol_ratio    = peak_vol / avg_vol if avg_vol > 0 else 0

    daily_returns  = close.pct_change().dropna()
    realised_vol   = daily_returns.std() * np.sqrt(252) * 100
    big_up_days    = int((daily_returns > 0.02).sum())
    big_down_days  = int((daily_returns < -0.02).sum())
    best_day_ret   = daily_returns.max() * 100
    worst_day_ret  = daily_returns.min() * 100
    best_day_date  = daily_returns.idxmax().strftime("%b %d")
    worst_day_date = daily_returns.idxmin().strftime("%b %d")

    tr  = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr     = tr.rolling(14).mean().iloc[-1]
    atr_pct = atr / period_close * 100

    direction  = "gained" if period_return >= 0 else "declined"
    trend_str  = "strongly" if abs(period_return) > 10 else ("moderately" if abs(period_return) > 4 else "slightly")

    two_week_ret  = close.pct_change(10).dropna() * 100
    swing_idx     = two_week_ret.abs().idxmax()
    biggest_swing = two_week_ret.abs().max()
    swing_dir     = "rally" if two_week_ret.loc[swing_idx] > 0 else "sell-off"
    swing_date    = swing_idx.strftime("%b %d")

    if vol_ratio >= 2.5:
        vol_narrative = (
            f"Volume spiked to {vol_ratio:.1f}× the 60-day average on {peak_vol_day} "
            f"({int(peak_vol):,} shares), suggesting a significant catalyst on that date."
        )
    elif vol_ratio >= 1.5:
        vol_narrative = (
            f"A volume increase of {vol_ratio:.1f}× the 60-day average occurred on {peak_vol_day}, "
            f"indicating elevated but not exceptional interest."
        )
    else:
        vol_narrative = (
            f"Volume was relatively stable throughout the period, with no single session "
            f"exceeding 1.5× the 60-day average of {int(avg_vol):,} shares per day."
        )

    df["week"] = df.index.to_period("W")
    weekly = df.groupby("week").agg(
        w_open  = ("Open",   "first"),
        w_close = ("Close",  "last"),
        w_high  = ("High",   "max"),
        w_low   = ("Low",    "min"),
        w_vol   = ("Volume", "mean"),
    )
    weekly["w_ret"] = (weekly["w_close"] - weekly["w_open"]) / weekly["w_open"] * 100

    week_rows = []
    for wk, row in weekly.iterrows():
        sign     = "+" if row["w_ret"] >= 0 else ""
        vs_avg   = row["w_vol"] / avg_vol if avg_vol > 0 else 1
        vol_note = (
            f" Volume for the week averaged {vs_avg:.1f}× the period average."
            if abs(vs_avg - 1) > 0.4 else ""
        )
        week_rows.append(
            f"<li>{ticker} in the week of {wk.start_time.strftime('%b %d, %Y')} "
            f"opened at ${row['w_open']:.2f} and closed at ${row['w_close']:.2f} "
            f"({sign}{row['w_ret']:.1f}%), trading in a range of "
            f"${row['w_low']:.2f} to ${row['w_high']:.2f}.{vol_note}</li>"
        )

    return f"""<section id="historical-60d">
<h2>Historical — last 60 trading days</h2>

<h3>Trend summary</h3>
<p>{ticker} {trend_str} {direction} {abs(period_return):.1f}% from ${period_open:.2f}
to ${period_close:.2f} between {period_start} and {period_end}.
The period high was ${period_high:.2f} and the period low was ${period_low:.2f}.
The maximum drawdown over the period was {max_drawdown:.2f}%, measured peak to trough.
The sharpest two-week {swing_dir} of {biggest_swing:.1f}% centred around {swing_date}.
Average daily trading volume over the period was {int(avg_vol):,} shares.</p>

<h3>Volatility</h3>
<p>{ticker}'s annualised realised volatility over the past 60 days is {realised_vol:.1f}%.
The 14-day ATR for {ticker} is ${atr:.2f}, representing {atr_pct:.2f}% of the current price.
{ticker} recorded {big_up_days} trading days with gains exceeding 2%
and {big_down_days} days with losses exceeding 2% during this period.
The best single day was +{best_day_ret:.2f}% on {best_day_date},
and the worst was {worst_day_ret:.2f}% on {worst_day_date}.</p>

<h3>Volume</h3>
<p>{vol_narrative}</p>

<h3>Weekly breakdown</h3>
<ol>{"".join(week_rows)}</ol>

</section>"""


# ── Per-ticker generators ─────────────────────────────────────────────────────

def _generate_one(ticker: str, meta: dict) -> tuple[str, dict]:
    """Fetch data and build HTML + metadata for a single ticker."""
    t    = yf.Ticker(ticker)
    info = t.info
    hist = t.history(period="2y")
    now  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    asset_type = meta["type"]

    if asset_type == "equity":
        body  = section_description(info)
        body += section_fundamentals_equity(info, ticker)
        body += section_price_momentum(info, hist, ticker)
        body += section_historical_60d(hist, ticker)
        body += section_earnings(info, ticker)
        label = "Equity"
    else:
        body  = section_description(info)
        body += section_fundamentals_etf(info, ticker)
        body += section_price_momentum(info, hist, ticker)
        body += section_historical_60d(hist, ticker)
        body += section_holdings_etf(info, ticker)
        label = asset_type.replace("_", " ").title()

    html = html_wrapper(ticker, meta["name"], label, body, now)
    md   = {
        "ticker":     ticker,
        "name":       meta["name"],
        "type":       asset_type,
        "updated_at": now,
        "sections":   ["description", "fundamentals/profile", "price-momentum",
                       "historical-60d", "analyst/holdings"],
        "source":     "yfinance",
    }
    return html, md


# ── Generate Tickers ────────────────────────────────────────────────────────────────

def generate_tickers(tickers: dict | None = None) -> list[str]:
    """
    Generate HTML + JSON knowledge-base files for the given tickers dict.

    Parameters
    ----------
    tickers : dict | None
        Mapping of  { "TICKER": {"name": "...", "type": "equity|etf|..."} }
        If None, falls back to DEFAULT_TICKERS (the original 20).
        Unknown tickers with missing meta are resolved automatically via
        build_ticker_meta().

    Returns
    -------
    list[str]
        Successfully generated ticker symbols.
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS

    os.makedirs(OUTPUT_DIR_HTML, exist_ok=True)
    os.makedirs(OUTPUT_DIR_JSON, exist_ok=True)

    generated = []
    print(f"\n Generating KB files for: {list(tickers.keys())}\n")

    for ticker, meta in tickers.items():
        # Auto-fill missing meta via yfinance lookup
        if not meta.get("name") or not meta.get("type"):
            print(f"  Auto-resolving meta for {ticker}…")
            resolved = build_ticker_meta(ticker)
            if resolved is None:
                continue
            meta = {**meta, **resolved}

        try:
            print(f"  Fetching {ticker}…")
            html, md = _generate_one(ticker, meta)

            html_path = os.path.join(OUTPUT_DIR_HTML, f"{ticker.lower()}.html")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html)

            meta_path = os.path.join(OUTPUT_DIR_JSON, f"{ticker.lower()}_meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(md, f, indent=2, default=str)

            generated.append(ticker)
            print(f"  ✓ {ticker} → {html_path}")

        except Exception as e:
            print(f"  ✗ {ticker} failed: {e}")

    print(f"\n✅ Done — {len(generated)}/{len(tickers)} tickers processed.")
    return generated


# ── TXT converter ─────────────────

def _strip_html_tags(text: str) -> str:
    text = re.sub(r"<h\d>(.*?)</h\d>", r"\1: ", text, flags=re.IGNORECASE)
    text = re.sub(r"<.*?>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _clean_text(text: str) -> str:
    text = text.replace("\n", " ").replace("\r", " ")
    return re.sub(r"\s+", " ", text).strip()


def _convert_numbers(text: str) -> str:
    def repl(match):
        s = match.group()
        s_clean = s.replace(",", "")
        if s.startswith("$"):
            num = s_clean[1:]
            if num.endswith("B"):   return f"{float(num[:-1])} billion dollars"
            if num.endswith("M"):   return f"{float(num[:-1])} million dollars"
            if num.endswith("K"):   return f"{float(num[:-1])} thousand dollars"
            return f"{float(num)} dollars"
        elif s.endswith("%"):
            return str(float(s_clean[:-1]) / 100)
        elif "shares" in s.lower():
            number = re.findall(r"\d[\d,]*", s)[0]
            return number.replace(",", "") + " shares "
        return s_clean

    text = re.sub(
        r"\$\d[\d,]*\.?\d*[BMK]?|\d[\d,]*\.?\d*%|\d[\d,]*\s*shares",
        repl, text, flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", text).strip()


def _split_sections(html_text: str) -> dict[str, str]:
    section_texts = {}
    for i, sec in enumerate(SECTIONS):
        start = html_text.find(sec)
        if start == -1:
            continue
        end = len(html_text)
        for next_sec in SECTIONS[i + 1:]:
            next_start = html_text.find(next_sec)
            if next_start != -1:
                end = next_start
                break
        key  = sec.lower().replace(" — ", "_").replace(" ", "_")
        text = _strip_html_tags(html_text[start:end])
        text = _clean_text(text)
        text = _convert_numbers(text)
        section_texts[key] = text
    return section_texts


def convert_tickers_into_txt(
    html_folder: str = OUTPUT_DIR_HTML,
    output_txt:  str = OUTPUT_TXT,
) -> list[dict]:
    """
    Convert all ticker HTML files in a single combined TXT audit file and return a list of chunk dicts for direct Chroma ingestion. TXT is for human read oly.

    Returns
    ----------
    list of {"ticker": str, "section": str, "text": str}
      — ready for direct upsert into Chroma collection "tickers"
    """
    chunks = []

    with open(output_txt, "w", encoding="utf-8") as txt_f:
        for filename in sorted(os.listdir(html_folder)):
            if not filename.endswith(".html"):
                continue
            ticker = filename.replace(".html", "").upper()
            path = os.path.join(html_folder, filename)
            with open(path, "r", encoding="utf-8") as f:
                html_text = f.read()
            sections = _split_sections(html_text)

            # TXT audit output
            txt_f.write(f"--- {ticker} ---\n")
            for sec_key, sec_text in sections.items():
                txt_f.write(f"[{sec_key.upper()}]\n{sec_text}\n\n")
            txt_f.write("\n")


            # Chunk list for Chroma
            for sec_key, sec_text in sections.items():
                if sec_text.strip():
                    chunks.append({
                        "ticker":      ticker,
                        "section":     sec_key,
                        "text":        sec_text,
                        "kb_source":   "tickers",
                        "source_url":  f"https://finance.yahoo.com/quote/{ticker}/",
                        "source_type": "yfinance",
                        "citation_id": f"ticker-{ticker}-{sec_key}",
                    })
 
    print(f"✅ kb1: {len(chunks)} chunks written to {output_txt}")
    return chunks

# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    generate_tickers()
    convert_tickers_into_txt()