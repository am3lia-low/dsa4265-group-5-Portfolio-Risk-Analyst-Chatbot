"""
kb2_macro_regime.py 
===============================
Macro & market regime data: FRED + yfinance sector ETFs + VIX.

Input (run on terminal)
------
export FRED_API_KEY="API_KEY_ENTER_HERE"
export FRED_API_KEY="a8e1d8754aa94cb4fd2bb11dace89f4b"
python kb3_macro_regime.py

Output
------
  knowledge_base/macro/macro_cache.json   — cached raw data (refreshed every 6h)
  kb3_macro_regime.txt                    — human-readable audit of all chunks
 
Folder
------
  knowledge_base/macro/
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

MACRO_DIR       = "knowledge_base/macro"
MACRO_CACHE     = os.path.join(MACRO_DIR, "macro_cache.json")
OUTPUT_TXT      = "output/kb3_macro_regime.txt"
STALENESS_HOURS = 6           # macro data refreshes every 6 hours

os.makedirs(MACRO_DIR, exist_ok=True)

SECTOR_ETFS = {
    "XLK": "Technology", "XLF": "Financials", "XLE": "Energy",
    "XLV": "Healthcare", "XLI": "Industrials", "XLU": "Utilities",
    "XLY": "Consumer Discretionary", "XLP": "Consumer Staples",
    "XLB": "Materials", "XLRE": "Real Estate",
}
 
FRED_SERIES = {
    "FEDFUNDS":        "Fed Funds Rate (%)",
    "CPIAUCSL":        "CPI (YoY %)",
    "DGS10":           "10Y Treasury Yield (%)",
    "DGS2":            "2Y Treasury Yield (%)",
    "UNRATE":          "Unemployment Rate (%)",
    "A191RL1Q225SBEA": "GDP Growth QoQ (%)",
}

# =============================================================================
# DATA CLEANING
# =============================================================================
 
def _clean_text(text: str) -> str:
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return re.sub(r"\s+", " ", text).strip()
 
 
def _convert_numbers(text: str) -> str:
    """
    Macro-specific number normalisation.
 
    Rules:
      • Signed "±X.XX%"  → "positive/negative X.XX percent"
      • Plain "X.XX%"    → "X.XX percent"
      • "Xbps"           → "X basis points"
      • "X×" multiplier  → "X times"
    """
    # Signed percentages first
    def _signed_pct(m):
        s   = m.group()
        try:
            num  = float(s.replace(",", "").rstrip("%"))
            word = "positive" if num >= 0 else "negative"
            return f"{word} {abs(num):.4g} percent"
        except:
            return m.group()
 
    text = re.sub(r"[+\-]\d[\d,]*\.?\d*%", _signed_pct, text)
 
    # Plain percentages
    def _plain_pct(m):
        s = m.group().replace(",", "").rstrip("%")
        try:    return f"{float(s):.4g} percent"
        except: return m.group()
 
    text = re.sub(r"\d[\d,]*\.?\d*%", _plain_pct, text)
 
    # Basis points
    text = re.sub(
        r"(\d+\.?\d*)\s*bps?\b",
        lambda m: f"{m.group(1)} basis points",
        text, flags=re.IGNORECASE,
    )
 
    # "X×" multiplier
    text = re.sub(r"(\d+\.?\d*)×", lambda m: f"{m.group(1)} times", text)
 
    return re.sub(r"\s+", " ", text).strip()
 
 
def _clean_chunk(text: str) -> str:
    return _convert_numbers(_clean_text(text))
 
# =============================================================================
# REGIME CLASSIFIER
# =============================================================================
def _safe_fmt(val, fmt="{:.2f}"):
    return fmt.format(val) if val is not None else "N/A"

def _classify_regime(
    vix:          float | None,
    curve_spread: float | None,
    cpi_yoy:      float | None,
    fed_funds:    float | None,
) -> tuple[str, str]:
    vix_high      = vix is not None and vix > 25
    vix_extreme   = vix is not None and vix > 35
    inverted      = curve_spread is not None and curve_spread < 0
    high_cpi      = cpi_yoy is not None and cpi_yoy > 4.0
    very_high_cpi = cpi_yoy is not None and cpi_yoy > 6.0
    high_rates    = fed_funds is not None and fed_funds > 4.5
 
    if vix_extreme:
        return ("risk_off",
                f"Extreme fear: VIX at {_safe_fmt(vix)} (above 35 threshold). "
                "Defensive assets and cash historically outperform.")
    if very_high_cpi and vix_high:
        return ("stagflation",
                f"Stagflation signals: CPI at {_safe_fmt(cpi_yoy)} percent with VIX {vix:.1f}. "
                "Real assets (commodities, TIPS) historically outperform.")
    if high_rates and inverted:
        return ("rate_stress",
                f"Rate stress: Fed Funds at {_safe_fmt(fed_funds)} percent, "
                f"inverted yield curve ({_safe_fmt(curve_spread)}pp spread). "
                "Short-duration bonds and value stocks tend to outperform.")
    if inverted and high_cpi:
        return ("risk_off",
                f"Risk-off: inverted yield curve ({curve_spread:.2f}pp) "
                f"and elevated CPI ({_safe_fmt(cpi_yoy)} percent). "
                "Inversion historically precedes recession within 12 to 18 months.")
    if not vix_high and not inverted and not high_cpi:
        return ("risk_on",
                f"Risk-on: VIX at {vix:.1f}, positive yield curve "
                f"({curve_spread:.2f}pp spread), inflation within range. "
                "Conditions favour growth equities and risk assets.")
    if vix_high and not inverted:
        return ("recovery",
                f"Potential recovery: elevated VIX ({_safe_fmt(vix)}), "
                f"positive yield curve ({_safe_fmt(curve_spread)}pp). "
                "Watch for VIX breaking below 20 as confirmation.")
    return ("neutral", "Mixed macro signals — no dominant regime.")
 
 
def _sector_rotation_narrative(sector_perf: dict[str, float]) -> str:
    if not sector_perf:
        return "Sector performance data unavailable."
    sorted_s  = sorted(sector_perf.items(), key=lambda x: x[1], reverse=True)
    leaders   = [(t, SECTOR_ETFS.get(t, t), r) for t, r in sorted_s[:3]]
    laggards  = [(t, SECTOR_ETFS.get(t, t), r) for t, r in sorted_s[-3:]]
    def _fmt(r): return f"{'positive' if r >= 0 else 'negative'} {abs(r*100):.2f} percent"
    lead_str  = ", ".join(f"{n} ({_fmt(r)})" for _, n, r in leaders)
    lag_str   = ", ".join(f"{n} ({_fmt(r)})" for _, n, r in laggards)
    defensive = {"XLU", "XLP", "XLV"}
    growth    = {"XLK", "XLY", "XLF"}
    lead_t    = {t for t, _, _ in leaders}
    if lead_t & defensive and not (lead_t & growth):
        signal = "Defensive sector leadership — bearish signal for broader market."
    elif lead_t & growth and not (lead_t & defensive):
        signal = "Growth and cyclical sector leadership — constructive signal for equities."
    else:
        signal = "Mixed sector leadership — no clear rotation signal."
    return (f"30-day sector leaders: {lead_str}. "
            f"Laggards: {lag_str}. {signal}")

# =============================================================================
# MACRO STORE
# =============================================================================

class MacroStore:
    """
    Fetches, caches, and chunks macro + regime data for RAG.

    Parameters
    ----------
    fred_api_key : str | None
        FRED API key. If None, FRED data is skipped and only yfinance data is used.
        Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html
    """

    def __init__(self, fred_api_key: str | None = None):
        self.fred_api_key = fred_api_key or os.getenv("FRED_API_KEY")
        self._data:   dict       = {}
        self._chunks: list[dict] = []
        if os.path.exists(MACRO_CACHE):
            self._load_cache()
 
    def _load_cache(self) -> None:
        with open(MACRO_CACHE) as f:
            self._data = json.load(f)
 
    def _save_cache(self) -> None:
        with open(MACRO_CACHE, "w") as f:
            json.dump(self._data, f, indent=2, default=str)
 
    def _is_stale(self) -> bool:
        ts = self._data.get("fetched_at")
        if not ts:
            return True
        age = (datetime.datetime.now() - datetime.datetime.fromisoformat(ts)).total_seconds() / 3600
        return age > STALENESS_HOURS
 
    def _fetch_fred(self) -> dict[str, float | None]:
        if not self.fred_api_key:
            print("  ⚠  No FRED API key — skipping. Set FRED_API_KEY env var.")
            return {k: None for k in FRED_SERIES}
        try:
            import fredapi
        except ImportError:
            raise ImportError("Run: pip install fredapi")
        fred   = fredapi.Fred(api_key=self.fred_api_key)
        result = {}
        for sid in FRED_SERIES:
            try:
                s = fred.get_series(sid).dropna()
                val = float((s.iloc[-1] / s.iloc[-13] - 1) * 100) if sid == "CPIAUCSL" and len(s) >= 13 else float(s.iloc[-1])
                result[sid] = round(val, 4)
                print(f"  ✓ FRED {sid}: {val:.4f}")
            except Exception as e:
                print(f"  ⚠  FRED {sid}: {e}")
                result[sid] = None
        return result
 
    def _fetch_yfinance(self) -> dict:
        result = {}
        try:
            vix_hist = yf.Ticker("^VIX").history(period="3mo")["Close"]
            vix_now  = float(vix_hist.iloc[-1])
            vix_30d  = float(vix_hist.iloc[-22]) if len(vix_hist) >= 22 else vix_now
            result.update({
                "vix_current": round(vix_now, 2),
                "vix_30d_ago": round(vix_30d, 2),
                "vix_trend":   "rising" if vix_now > vix_30d else "falling",
            })
        except Exception as e:
            print(f"  ⚠  VIX: {e}")
            result["vix_current"] = None
 
        sector_perf = {}
        try:
            prices = yf.download(list(SECTOR_ETFS), period="2mo",
                                  auto_adjust=True, progress=False)["Close"]
            for t in SECTOR_ETFS:
                if t in prices.columns:
                    col = prices[t].dropna()
                    if len(col) >= 22:
                        sector_perf[t] = round(float((col.iloc[-1] / col.iloc[-22]) - 1), 4)
        except Exception as e:
            print(f"  ⚠  Sector ETFs: {e}")
        result["sector_30d_returns"] = sector_perf
 
        broad = {}
        try:
            for idx in ["SPY", "QQQ", "IWM"]:
                h = yf.Ticker(idx).history(period="2mo")["Close"].dropna()
                if len(h) >= 22:
                    broad[idx] = round(float((h.iloc[-1] / h.iloc[-22]) - 1), 4)
        except Exception as e:
            print(f"  ⚠  Broad market: {e}")
        result["broad_market_30d"] = broad
 
        return result
 
    def refresh(self, force: bool = False) -> None:
        if not force and not self._is_stale():
            print("  ✓ Macro cache fresh — skipping refresh")
            return
        print("\n📡 Refreshing macro data…")
        fred_data = self._fetch_fred()
        yf_data   = self._fetch_yfinance()
        y10, y2   = fred_data.get("DGS10"), fred_data.get("DGS2")
        spread    = round(y10 - y2, 4) if y10 is not None and y2 is not None else None
        regime, regime_explanation = _classify_regime(
            yf_data.get("vix_current"), spread,
            fred_data.get("CPIAUCSL"), fred_data.get("FEDFUNDS"),
        )
        self._data = {
            "fetched_at":         datetime.datetime.now().isoformat(),
            "fred":               fred_data,
            "vix_current":        yf_data.get("vix_current"),
            "vix_30d_ago":        yf_data.get("vix_30d_ago"),
            "vix_trend":          yf_data.get("vix_trend"),
            "sector_30d_returns": yf_data.get("sector_30d_returns", {}),
            "broad_market_30d":   yf_data.get("broad_market_30d", {}),
            "yield_curve_spread": spread,
            "regime":             regime,
            "regime_explanation": regime_explanation,
        }
        self._save_cache()
        self._chunks = []
        print(f"  📊 Regime: {regime.upper()} — {regime_explanation[:80]}…")
 
    # ── Chunk generation + cleaning ───────────────────────────────────────────
 
    def generate_chunks(self) -> list[dict]:
        if self._chunks:
            return self._chunks
        if not self._data:
            print("  ⚠  No macro data — call refresh() first")
            return []
 
        now   = self._data.get("fetched_at", datetime.datetime.now().isoformat())[:16]
        fred  = self._data.get("fred", {})
        raw   = []
 
        # Regime
        raw.append(("macro-regime", ["trend_prediction", "full_analysis", "rebalance"],
            "fred+yfinance",
            f"Current market regime: {self._data['regime'].upper()}. "
            + self._data.get("regime_explanation", "")
        ))
 
        # Rates
        fedfunds = fred.get("FEDFUNDS")
        y10      = fred.get("DGS10")
        y2       = fred.get("DGS2")
        spread   = self._data.get("yield_curve_spread")
        rate_parts = []
        if fedfunds is not None:
            rate_parts.append(f"Federal Funds rate is {fedfunds:.2f}%.")
        if y10 is not None:
            rate_parts.append(f"10-year Treasury yield is {y10:.2f}%.")
        if y2 is not None:
            rate_parts.append(f"2-year Treasury yield is {y2:.2f}%.")
        if spread is not None:
            if spread < 0:
                curve_desc = f"Yield curve is inverted by {abs(spread):.2f}pp (10Y minus 2Y = {spread:.2f}%), historically a reliable recession indicator."
            elif spread < 0.25:
                curve_desc = f"Yield curve is flat ({spread:.2f}pp spread), indicating market uncertainty."
            else:
                curve_desc = f"Yield curve is positive ({spread:.2f}pp spread), consistent with growth expectations."
            rate_parts.append(curve_desc)
        if rate_parts:
            raw.append(("macro-rates", ["trend_prediction", "full_analysis", "rebalance"],
                "fred", " ".join(rate_parts)
            ))
 
        # Economy
        cpi    = fred.get("CPIAUCSL")
        unrate = fred.get("UNRATE")
        gdp    = fred.get("A191RL1Q225SBEA")
        econ   = []
        if cpi is not None:
            interp = (f"CPI inflation is elevated at {cpi:.2f}% YoY, above the Fed's 2% target."
                      if cpi > 4 else
                      f"CPI inflation is {cpi:.2f}% YoY, near the Fed's target.")
            econ.append(interp)
        if unrate is not None:
            econ.append(f"Unemployment rate is {unrate:.1f}%.")
        if gdp is not None:
            econ.append(f"GDP growth last quarter was {gdp:.2f}% annualised ({'expansion' if gdp > 0 else 'contraction'}).")
        if econ:
            raw.append(("macro-economy", ["trend_prediction", "full_analysis"],
                "fred", " ".join(econ)
            ))
 
        # VIX
        vix = self._data.get("vix_current")
        if vix is not None:
            if vix < 15:
                vix_interp = f"VIX at {vix:.1f} — very low fear, complacency risk elevated."
            elif vix < 20:
                vix_interp = f"VIX at {vix:.1f} — calm conditions, normal risk appetite."
            elif vix < 30:
                vix_interp = f"VIX at {vix:.1f} — elevated uncertainty, options expensive."
            else:
                vix_interp = f"VIX at {vix:.1f} — extreme fear, historically coincides with market bottoms."
            vix_30d = self._data.get("vix_30d_ago")
            trend_note = ""
            if vix_30d:
                change     = vix - vix_30d
                trend_note = (f" VIX has {'risen' if change > 0 else 'fallen'} "
                              f"{abs(change):.1f} points over 30 days.")
            raw.append(("macro-vix", ["trend_prediction", "full_analysis", "rebalance"],
                "yfinance", vix_interp + trend_note
            ))
 
        # Sector rotation
        sp = self._data.get("sector_30d_returns", {})
        if sp:
            raw.append(("macro-sectors", ["trend_prediction", "rebalance"],
                "yfinance", _sector_rotation_narrative(sp)
            ))
 
        # Broad market
        broad = self._data.get("broad_market_30d", {})
        if broad:
            names = {"SPY": "S&P 500", "QQQ": "Nasdaq-100", "IWM": "Russell 2000"}
            def _fmt(r): return f"{'positive' if r >= 0 else 'negative'} {abs(r*100):.2f} percent"
            b_lines = "; ".join(f"{names.get(t, t)} {_fmt(r)} over 30 days"
                                for t, r in broad.items())
            spy_vs_iwm = ("Large-cap outperforming small-cap — risk-on breadth."
                          if broad.get("SPY", 0) > broad.get("IWM", 0)
                          else "Small-cap outperforming large-cap — breadth expansion signal.")
            raw.append(("macro-broad-market", ["trend_prediction", "full_analysis"],
                "yfinance", f"Broad market 30-day: {b_lines}. {spy_vs_iwm}"
            ))
 
        self._chunks = [
            {
                "citation_id":  cid,
                "kb_source":    "macro",
                "intent_tags":  tags,
                "source_url":   f"https://fred.stlouisfed.org" if src == "fred" else "https://finance.yahoo.com",
                "source_type":  src,
                "updated_at":   now,
                "text":         _clean_chunk(raw_text),
            }
            for cid, tags, src, raw_text in raw
        ]
        return self._chunks
 
    # ── TXT export ────────────────────────────────────────────────────────────
 
    def export_txt(self, output_path: str = OUTPUT_TXT) -> str:
        chunks = self.generate_chunks()
        now    = self._data.get("fetched_at", datetime.datetime.now().isoformat())[:16]
        regime = self._data.get("regime", "unknown").upper()
 
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"# kb3 MACRO REGIME — {regime} — generated {now}\n\n")
            for c in chunks:
                f.write(f"--- {c['citation_id']} ---\n")
                f.write(f"Intent tags: {', '.join(c['intent_tags'])}\n")
                f.write(c["text"] + "\n\n")
 
        print(f"  ✓ kb3 TXT → {output_path}  ({len(chunks)} chunks)")
        return output_path
 
 
# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    ms = MacroStore()
    ms.refresh(force=True)
    chunks = ms.generate_chunks()
    ms.export_txt()
    print(f"\n✅ {len(chunks)} macro chunks ready")
    for c in chunks:
        print(f"  [{c['citation_id']}] {c['text'][:100]}…")
 