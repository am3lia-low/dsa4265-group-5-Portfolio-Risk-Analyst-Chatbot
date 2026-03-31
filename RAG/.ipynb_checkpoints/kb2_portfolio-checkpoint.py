"""
kb2_portfolio.py 
===============================
Portfolio store: holdings management, portfolio-level risk metrics, data cleaning. TXT eport, and chunk generation for Chroma ingestion.

Outut
------
  knowledge_base/portfolio/portfolio.json   — persisted holdings + target weights
  knowledge_base/portfolio/metrics.json     — last computed metrics snapshot
  kb2_portfolio.txt                         — human-readable audit of all chunks
 
Folder
------
  knowledge_base/portfolio/

"""

from __future__ import annotations

import os
import re
import json
import datetime
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

PORTFOLIO_DIR  = "knowledge_base/portfolio"
PORTFOLIO_FILE = os.path.join(PORTFOLIO_DIR, "portfolio.json")
METRICS_FILE = os.path.join(PORTFOLIO_DIR, "metrics.json")
OUTPUT_TXT = "output/kb2_portfolio.txt"

LOOKBACK_DAYS  = 252          # 1 trading year for metric computation
RISK_FREE_RATE = 0.05         # annualised, update to match current T-bill rate
TRADING_DAYS   = 252

os.makedirs(PORTFOLIO_DIR, exist_ok = True)

# =============================================================================
# DATA CLEANING
# =============================================================================
 
def _clean_text(text: str) -> str:
    """Collapse whitespace and strip control characters."""
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return re.sub(r"\s+", " ", text).strip()
 
 
def _convert_numbers(text: str) -> str:
    """
    Portfolio-specific number normalisation for RAG readability.
 
    Rules:
      • "$NNN.NN"           → "NNN.NN dollars"
      • "X%"                → "X percent"          (keeps numeric value)
      • "Xpp"               → "X percentage points"
      • "X.Xx" (multiplier) → "X.X times"
      • "0.XXXX" after ratio labels → kept as-is   (Sharpe, beta, HHI etc.)
    """
    # $NNN.NN → "NNN.NN dollars"
    def _dollar(m):
        s = m.group().replace(",", "")[1:]
        try:    return f"{float(s):.2f} dollars"
        except: return m.group()
 
    text = re.sub(r"\$\d[\d,]*\.?\d*", _dollar, text)
 
    # "X%" → "X percent"
    def _pct(m):
        s = m.group().replace(",", "").rstrip("%")
        try:    return f"{float(s):.4g} percent"
        except: return m.group()
 
    text = re.sub(r"-?\d[\d,]*\.?\d*%", _pct, text)
 
    # "Xpp" → "X percentage points"
    text = re.sub(
        r"(\d+\.?\d*)\s*pp\b",
        lambda m: f"{m.group(1)} percentage points",
        text,
    )
 
    # "X.Xx" multiplier → "X.X times"
    text = re.sub(
        r"(\d+\.?\d*)x\b",
        lambda m: f"{m.group(1)} times",
        text,
    )
 
    return re.sub(r"\s+", " ", text).strip()
 
 
def _clean_chunk(text: str) -> str:
    return _convert_numbers(_clean_text(text))

# =============================================================================
# METRIC HELPERS
# =============================================================================

def _fetch_prices(tickers: list[str], days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    period = f"{max(days // 21 + 2, 14)}mo"
    raw    = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    prices = prices[[t for t in tickers if t in prices.columns]]
    return prices.dropna(how="all").tail(days)

def _ann_return(returns: pd.Series) -> float: #Annualised returns
    return float((1 + returns.mean()) ** TRADING_DAYS - 1)


def _ann_vol(returns: pd.Series) -> float: #Annualised volatility
    return float(returns.std() * np.sqrt(TRADING_DAYS))


def _sharpe(returns: pd.Series, rf: float = RISK_FREE_RATE) -> float: #Sharpe ratio
    exc   = _ann_return(returns) - rf
    vol   = _ann_vol(returns)
    return round(exc / vol, 4) if vol > 0 else 0.0


def _sortino(returns: pd.Series, rf: float = RISK_FREE_RATE) -> float: #Sortino ratio
    exc         = _ann_return(returns) - rf
    downside    = returns[returns < 0].std() * np.sqrt(TRADING_DAYS)
    return round(exc / downside, 4) if downside > 0 else 0.0


def _max_drawdown(prices: pd.Series) -> float: #Maximum drawdown
    roll_max = prices.cummax()
    dd       = (prices - roll_max) / roll_max
    return round(float(dd.min()), 4)


def _var(returns: pd.Series, confidence: float = 0.95) -> float: #VaR
    """Daily VaR at given confidence level (as positive loss fraction)."""
    return round(float(-np.percentile(returns.dropna(), (1 - confidence) * 100)), 4)


def _herfindahl(weights: dict[str, float]) -> float: #HHI for concentration
    """
    Herfindahl-Hirschman Index for concentration.
    Range [1/n, 1]: 1/n = perfectly diversified, 1 = single position.
    """
    return round(sum(w ** 2 for w in weights.values()), 4)

def _skewness(returns: pd.Series) -> float: #Skewness
    return round(float(returns.skew()), 4)

def _excess_kurtosis(returns: pd.Series) -> float: #pandas .kurt() returns excess kurtosis
    return round(float(returns.kurt()), 4) 

def _rolling_vol_21d(returns: pd.Series) -> float: #Rolling volatility for 21 days
    """Annualised volatility over the most recent 21 trading days."""
    recent = returns.tail(21)
    return round(float(recent.std() * np.sqrt(TRADING_DAYS)), 4) if len(recent) >= 5 else float("nan")


def _risk_label(vol: float) -> str: #Risk Label
    if vol < 0.10:   return "low"
    elif vol < 0.18: return "moderate"
    elif vol < 0.28: return "high"
    else:            return "very high"


def _concentration_label(hhi: float, n: int) -> str: #Concentration Label
    min_hhi = 1 / n if n > 0 else 1
    ratio   = (hhi - min_hhi) / (1 - min_hhi) if (1 - min_hhi) > 0 else 0
    if ratio < 0.25:   return "well diversified"
    elif ratio < 0.50: return "moderately concentrated"
    elif ratio < 0.75: return "concentrated"
    else:              return "highly concentrated"


# =============================================================================
# PORTFOLIO STORE
# =============================================================================

class PortfolioStore:
    """
    Central object for all portfolio data and metric computation.

    Holdings format
    ---------------
    {
      "TICKER": {"weight": float, "cost_basis": float},
      ...
    }
    Weights should sum to 1.0 (or close to it).
    cost_basis is the average purchase price per share in USD.
    """

    def __init__(self, portfolio_path: str = PORTFOLIO_FILE):
        self.portfolio_path = portfolio_path
        self.holdings:      dict[str, dict] = {}
        self.target_weights: dict[str, float] = {}
        self._metrics:      dict | None = None
        self._prices:       pd.DataFrame | None = None
        self._chunks:       list[dict] = []

        if os.path.exists(portfolio_path):
            self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        with open(self.portfolio_path) as f:
            data = json.load(f)
        self.holdings       = data.get("holdings", {})
        self.target_weights = data.get("target_weights", {})
        print(f"  ✓ Portfolio loaded: {list(self.holdings.keys())}")

    def save(self) -> None:
        with open(self.portfolio_path, "w") as f:
            json.dump({
                "holdings":       self.holdings,
                "target_weights": self.target_weights,
                "updated_at":     datetime.datetime.now().isoformat(),
            }, f, indent=2)

    # ── Holdings management ───────────────────────────────────────────────────

    def set_holdings(
        self,
        holdings:        dict[str, dict],
        target_weights:  dict[str, float] | None = None,
    ) -> None:
        total = sum(h.get("weight", 0) for h in holdings.values())
        if total <= 0:
            raise ValueError("Weights must sum to a positive number.")
        self.holdings = {
            t: {"weight": round(h.get("weight", 0) / total, 6),
                "cost_basis": float(h.get("cost_basis", 0))}
            for t, h in holdings.items()
        }
        self.target_weights = target_weights or {
            t: h["weight"] for t, h in self.holdings.items()
        }
        self._metrics = None
        self._prices  = None
        self._chunks  = []
        self.save()
        print(f"  ✓ Holdings set: {list(self.holdings.keys())}")

    # ── Price fetching ────────────────────────────────────────────────────────

    def _get_prices(self) -> pd.DataFrame:
        if self._prices is not None:
            return self._prices
        tickers = list(self.holdings.keys()) + ["SPY"]
        self._prices = _fetch_prices(list(set(tickers)))
        return self._prices

    def get_metrics(self, force: bool = False) -> dict:
        if self._metrics is not None and not force:
            return self._metrics
 
        prices  = self._get_prices()
        returns = prices.pct_change().dropna()
        weights = {t: self.holdings[t]["weight"]
                   for t in self.holdings if t in returns.columns}
        tickers = list(weights.keys())
        if not tickers:
            return {}
 
        w_arr      = np.array([weights[t] for t in tickers])
        ret_mat    = returns[tickers].values
        port_ret   = pd.Series(ret_mat @ w_arr, index=returns.index[-len(ret_mat):])
        port_prices = (1 + port_ret).cumprod() * 100
 
        # Covariance and correlation matrices
        cov_df  = returns[tickers].cov() * TRADING_DAYS   # annualised covariance
        corr_df = returns[tickers].corr()
 
        # Portfolio volatility decomposition — risk contribution per asset
        port_vol = _ann_vol(port_ret)
        risk_contributions = {}
        for i, t in enumerate(tickers):
            # Marginal contribution = w_i * (Cov @ w)_i / port_vol
            mctr = float(w_arr[i] * (cov_df.values @ w_arr)[i] / (port_vol ** 2 + 1e-12))
            risk_contributions[t] = round(mctr, 4)
 
        # Beta vs SPY
        beta_vs_spy = None
        if "SPY" in returns.columns:
            cov_mat     = np.cov(port_ret.values, returns["SPY"].values)
            spy_var     = float(np.var(returns["SPY"].values))
            beta_vs_spy = round(cov_mat[0, 1] / spy_var, 4) if spy_var > 0 else None
 
        # Individual metrics
        individual = {}
        for t in tickers:
            r = returns[t].dropna()
            p = prices[t].dropna()
            # per-asset beta vs SPY
            asset_beta = None
            if "SPY" in returns.columns:
                cov2    = np.cov(r.values, returns["SPY"].dropna().values)
                spy_var2 = float(np.var(returns["SPY"].dropna().values))
                asset_beta = round(cov2[0, 1] / spy_var2, 4) if spy_var2 > 0 else None
 
            individual[t] = {
                "weight":            round(weights[t], 4),
                "annualised_return": round(_ann_return(r), 4),
                "annualised_vol":    round(_ann_vol(r), 4),
                "vol_21d":           _rolling_vol_21d(r),
                "sharpe":            _sharpe(r),
                "sortino":           _sortino(r),
                "max_drawdown":      _max_drawdown(p),
                "var_95":            _var(r, 0.95),
                "skewness":          _skewness(r),
                "excess_kurtosis":   _excess_kurtosis(r),
                "beta_vs_spy":       asset_beta,
                "risk_contribution": risk_contributions.get(t, 0.0),
                "risk_label":        _risk_label(_ann_vol(r)),
            }
 
        # Drift
        drift = {
            t: round(weights.get(t, 0) - self.target_weights.get(t, weights.get(t, 0)), 4)
            for t in tickers
        }
 
        hhi = _herfindahl(weights)
        self._metrics = {
            "computed_at":          datetime.datetime.now().isoformat(),
            "tickers":              tickers,
            "portfolio_return":     round(_ann_return(port_ret), 4),
            "portfolio_volatility": round(port_vol, 4),
            "vol_21d":              _rolling_vol_21d(port_ret),
            "sharpe":               _sharpe(port_ret),
            "sortino":              _sortino(port_ret),
            "max_drawdown":         _max_drawdown(port_prices),
            "var_95":               _var(port_ret, 0.95),
            "var_99":               _var(port_ret, 0.99),
            "skewness":             _skewness(port_ret),
            "excess_kurtosis":      _excess_kurtosis(port_ret),
            "beta_vs_spy":          beta_vs_spy,
            "herfindahl":           hhi,
            "concentration_label":  _concentration_label(hhi, len(tickers)),
            "risk_label":           _risk_label(port_vol),
            "covariance_matrix":    cov_df.round(6).to_dict(),
            "correlation_matrix":   corr_df.round(4).to_dict(),
            "risk_contributions":   risk_contributions,
            "individual_metrics":   individual,
            "drift_from_target":    drift,
        }
 
        # Persist metrics snapshot
        with open(METRICS_FILE, "w") as f:
            json.dump(self._metrics, f, indent=2, default=str)
 
        return self._metrics
 
    # ── Drift signals ─────────────────────────────────────────────────────────
 
    def get_rebalance_signals(self, drift_threshold: float = 0.05) -> list[dict]:
        metrics = self.get_metrics()
        signals = []
        for t, drift in metrics.get("drift_from_target", {}).items():
            if abs(drift) >= drift_threshold:
                direction = "overweight" if drift > 0 else "underweight"
                signals.append({
                    "ticker":         t,
                    "current_weight": metrics["individual_metrics"][t]["weight"],
                    "target_weight":  self.target_weights.get(t, 0),
                    "drift":          drift,
                    "direction":      direction,
                    "action_needed":  f"{'Trim' if drift > 0 else 'Add to'} {t}",
                })
        return sorted(signals, key=lambda x: abs(x["drift"]), reverse=True)
    
    # ── Chunk generation + cleaning ──────────────────────────────────────────────────────

    def generate_chunks(self, force: bool = False) -> list[dict]:
        """
        Build cleaned prose chunks from portfolio metrics.
        Each chunk is independently readable and tagged for intent routing.
        """
        if self._chunks and not force:
            return self._chunks
 
        m   = self.get_metrics()
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
 
        raw_chunks = []

        # ── 1. Overview ───────────────────────────────────────────────────────
        tickers_str = ", ".join(m["tickers"])
        raw_chunks.append(("portfolio-overview", ["full_analysis", "rebalance"],
            f"The portfolio holds {len(m['tickers'])} positions: {tickers_str}. "
            f"Annualised return is {m['portfolio_return']*100:.2f}%, "
            f"annualised volatility {m['portfolio_volatility']*100:.2f}% "
            f"({m['risk_label']} risk). "
            f"21-day rolling volatility is {m['vol_21d']*100:.2f}%. "
            f"Sharpe ratio {m['sharpe']:.2f}, Sortino ratio {m['sortino']:.2f} "
            f"(risk-free rate {RISK_FREE_RATE*100:.1f}%). "
            f"Maximum drawdown {m['max_drawdown']*100:.2f}%. "
            f"Beta vs S&P 500: {m['beta_vs_spy']:.2f}."
        ))

        # ── 2. Risk metrics ───────────────────────────────────────────────────
        skew_interp = (
            "positively skewed — occasional large gains" if m["skewness"] > 0.5
            else "negatively skewed — tail risk skews to large losses" if m["skewness"] < -0.5
            else "approximately symmetric"
        )
        kurt_interp = (
            "leptokurtic — fatter tails than normal, extreme moves more likely"
            if m["excess_kurtosis"] > 1
            else "platykurtic — thinner tails than normal"
            if m["excess_kurtosis"] < -1
            else "near-normal tail behaviour"
        )
        raw_chunks.append(("portfolio-risk", ["full_analysis"],
            f"95% daily VaR: {m['var_95']*100:.2f}% "
            f"(on 95 percent of days, loss will not exceed this level). "
            f"99% VaR: {m['var_99']*100:.2f}%. "
            f"Herfindahl-Hirschman Index (HHI): {m['herfindahl']:.4f} — "
            f"{m['concentration_label']} "
            f"(equal-weight benchmark: {1/len(m['tickers']):.4f}). "
            f"Return distribution is {skew_interp} (skewness {m['skewness']:.2f}), "
            f"with {kurt_interp} (excess kurtosis {m['excess_kurtosis']:.2f})."
        ))

        # ── 3. Correlation + covariance ───────────────────────────────────────
        corr  = m["correlation_matrix"]
        tickers = m["tickers"]
        pairs = []
        for i, t1 in enumerate(tickers):
            for t2 in tickers[i+1:]:
                val = corr.get(t1, {}).get(t2)
                if val is not None:
                    pairs.append((t1, t2, round(val, 3)))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
 
        high_corr = [p for p in pairs if abs(p[2]) >= 0.70]
        pair_str  = (
            "; ".join(f"{t1}-{t2} ({c:+.2f})" for t1, t2, c in high_corr)
            if high_corr else "none"
        )
        all_pairs_str = "; ".join(
            f"{t1}-{t2} ({c:+.2f})" for t1, t2, c in pairs
        )
        raw_chunks.append(("portfolio-correlation", ["full_analysis", "rebalance"],
            f"Pairwise correlations (all pairs): {all_pairs_str}. "
            f"Highly correlated pairs (correlation above 0.70): {pair_str}. "
            f"High positive correlation means assets move in lockstep, "
            f"reducing effective diversification."
        ))

        # ── 4. Risk contributions ─────────────────────────────────────────────
        rc = m["risk_contributions"]
        rc_lines = "; ".join(
            f"{t} contributes {v*100:.1f}% of total portfolio volatility"
            for t, v in sorted(rc.items(), key=lambda x: x[1], reverse=True)
        )
        raw_chunks.append(("portfolio-risk-contributions", ["full_analysis"],
            f"Risk contribution (share of total portfolio volatility): {rc_lines}. "
            f"A position with a small weight but high risk contribution "
            f"is disproportionately driving portfolio volatility."
        ))

        # ── 5. Individual positions ───────────────────────────────────────────
        ind = m["individual_metrics"]
        pos_lines = []
        for t, im in sorted(ind.items(), key=lambda x: x[1]["annualised_vol"], reverse=True):
            pos_lines.append(
                f"{t} (weight {im['weight']*100:.1f}%): "
                f"vol {im['annualised_vol']*100:.1f}% (21-day: {im['vol_21d']*100:.1f}%), "
                f"Sharpe {im['sharpe']:.2f}, Sortino {im['sortino']:.2f}, "
                f"MDD {im['max_drawdown']*100:.1f}%, "
                f"beta {im['beta_vs_spy']:.2f}, "
                f"skewness {im['skewness']:.2f}, "
                f"excess kurtosis {im['excess_kurtosis']:.2f}, "
                f"risk contribution {im['risk_contribution']*100:.1f}%, "
                f"risk level {im['risk_label']}"
            )
        raw_chunks.append(("portfolio-positions", ["full_analysis", "rebalance"],
            "Per-position metrics (sorted by volatility): "
            + ". ".join(pos_lines) + "."
        ))

        # ── 6. Drift / rebalance signals ──────────────────────────────────────
        signals = self.get_rebalance_signals()
        if signals:
            sig_lines = "; ".join(
                f"{s['ticker']} {s['direction']} by {abs(s['drift'])*100:.1f}pp "
                f"(current {s['current_weight']*100:.1f}% vs target "
                f"{s['target_weight']*100:.1f}%) — {s['action_needed']}"
                for s in signals
            )
            drift_text = (
                f"{len(signals)} rebalance signal(s) detected: {sig_lines}. "
                f"These positions have drifted beyond the 5 percentage point threshold."
            )
        else:
            drift_text = (
                "No rebalance signals. All positions are within "
                "5 percentage points of target allocations."
            )
        raw_chunks.append(("portfolio-drift", ["rebalance"], drift_text))
 
        # ── Build final chunk list with cleaning applied ───────────────────────
        self._chunks = [
            {
                "citation_id":  cid,
                "kb_source":    "portfolio",
                "intent_tags":  tags,
                "source_url":   "internal://portfolio",
                "source_type":  "portfolio_store",
                "updated_at":   now,
                "text":         _clean_chunk(raw_text),
            }
            for cid, tags, raw_text in raw_chunks
        ]
        return self._chunks

    # ── TXT export ────────────────────────────────────────────────────────────
 
    def export_txt(self, output_path: str = OUTPUT_TXT) -> str:
        """
        Write all portfolio chunks to a human-readable TXT file.
        Cleaning is applied before writing (the same text that goes into Chroma).
        """
        chunks = self.generate_chunks()
        now    = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
 
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"# kb2 PORTFOLIO — generated {now}\n\n")
            for c in chunks:
                f.write(f"--- {c['citation_id']} ---\n")
                f.write(f"Intent tags: {', '.join(c['intent_tags'])}\n")
                f.write(c["text"] + "\n\n")
 
        print(f"  ✓ kb2 TXT → {output_path}  ({len(chunks)} chunks)")
        return output_path

# =============================================================================
# CLI
# =============================================================================
# if __name__ == "__main__":
#     ps = PortfolioStore()
#     if not ps.holdings:
#         ps.set_holdings({
#             "AAPL": {"weight": 0.30, "cost_basis": 150.00},
#             "MSFT": {"weight": 0.25, "cost_basis": 280.00},
#             "NVDA": {"weight": 0.20, "cost_basis": 400.00},
#             "SPY":  {"weight": 0.15, "cost_basis": 420.00},
#             "TLT":  {"weight": 0.10, "cost_basis":  95.00},
#         })
#     chunks = ps.generate_chunks()
#     ps.export_txt()
#     print(f"\n✅ {len(chunks)} portfolio chunks ready")
#     for c in chunks:
#         print(f"  [{c['citation_id']}] {c['text'][:100]}…")