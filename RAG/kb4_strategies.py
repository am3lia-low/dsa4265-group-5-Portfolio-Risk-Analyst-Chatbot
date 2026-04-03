"""
kb4_strategies.py 
===============================
Strategy & rebalancing framework knowledge base — Intent 2 (Rebalance justification).

Output
------
  knowledge_base/strategies/strategies.json   — persisted strategy dict
  kb4_strategies.txt                          — human-readable audit
 
Folder
------
  knowledge_base/strategies/
"""

from __future__ import annotations

import os
import re
import json
import datetime

STRATEGIES_DIR  = "knowledge_base/kb4_strategies"
STRATEGIES_FILE = os.path.join(STRATEGIES_DIR, "strategies.json")
OUTPUT_TXT      = "output/kb4_strategies.txt"

os.makedirs(STRATEGIES_DIR, exist_ok=True)

# =============================================================================
# DATA CLEANING
# =============================================================================
 
def _clean_text(text: str) -> str:
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return re.sub(r"\s+", " ", text).strip()
 
 
def _convert_numbers(text: str) -> str:
    """
    Strategy-specific number normalisation.
 
    Rules:
      • "±Xpp" / "±X pp"  → "plus or minus X percentage points"
      • "X-Y%"  ranges    → "X to Y percent"
      • Plain "X%"        → "X percent"
      • "$X,XXX"          → "X dollars"
      • Bullet chars      → removed
    """
    # "±Xpp" → "plus or minus X percentage points"
    text = re.sub(
        r"[±\+\-]?\s*(\d+\.?\d*)\s*pp\b",
        lambda m: f"plus or minus {m.group(1)} percentage points",
        text,
    )
 
    # "X-Y%" ranges
    text = re.sub(
        r"(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)%",
        lambda m: f"{m.group(1)} to {m.group(2)} percent",
        text,
    )
 
    # Plain "X%"
    def _pct(m):
        s = m.group().replace(",", "").rstrip("%")
        try:    return f"{float(s):.4g} percent"
        except: return m.group()
 
    text = re.sub(r"-?\d[\d,]*\.?\d*%", _pct, text)
 
    # "$X,XXX"
    def _dollar(m):
        s = m.group().replace(",", "")[1:]
        try:    return f"{float(s):.2f} dollars"
        except: return m.group()
 
    text = re.sub(r"\$\d[\d,]*\.?\d*", _dollar, text)
 
    # Strip bullet characters
    text = re.sub(r"[•·–—]\s*", " ", text)
 
    return re.sub(r"\s+", " ", text).strip()
 
 
def _clean_chunk(text: str) -> str:
    return _convert_numbers(_clean_text(text))

# =============================================================================
# STRATEGY DEFINITIONS
# =============================================================================

STRATEGIES: dict[str, dict] = {

    # ── Rebalancing Frameworks ────────────────────────────────────────────────

    "threshold_rebalancing": {
        "label":    "Threshold Rebalancing (Bands Strategy)",
        "category": "rebalancing",
        "what":     "Rebalance only when a position drifts beyond a defined percentage-point band from its target weight. Common bands: ±5pp for major asset classes, ±3pp for smaller positions.",
        "when":     "Triggered by price action, not by the calendar. Monitor drift continuously or at fixed review intervals (weekly). Rebalance if any position is outside its band.",
        "how":      "Step 1: Define target weights and bands for each position. Step 2: Calculate current weight of each position. Step 3: Identify positions outside bands. Step 4: Sell the most overweight positions and use proceeds to buy the most underweight positions. Step 5: Execute in a tax-efficient order (use tax-advantaged accounts first, then taxable).",
        "why":      "Research by Vanguard and Dimensional Fund Advisors shows threshold rebalancing outperforms calendar rebalancing on a risk-adjusted basis by 0.1-0.4% annually. It trades less (lower costs), but trades more meaningfully (larger drift corrections). It also naturally exploits mean reversion — selling assets that have risen and buying those that have fallen.",
        "risks":    "Can underperform in strongly trending markets where momentum persists. A trend-following asset that keeps rising will be repeatedly trimmed, missing gains. Mitigate by setting wider bands (±7-10pp) for higher-conviction positions.",
        "trigger_conditions": [
            "Any position drifts more than 5 percentage points from target weight",
            "Portfolio beta deviates more than 0.2 from target beta",
            "Correlation between any two holdings rises above 0.85 (diversification breakdown)",
            "Sector weight exceeds 40% of total portfolio",
        ],
        "related":  ["calendar_rebalancing", "tax_loss_harvesting", "risk_parity"],
    },

    "calendar_rebalancing": {
        "label":    "Calendar Rebalancing",
        "category": "rebalancing",
        "what":     "Rebalance back to target allocations at fixed time intervals — monthly, quarterly, semi-annually, or annually — regardless of how much drift has occurred.",
        "when":     "Applied mechanically at the chosen interval. Annual rebalancing is most common for tax efficiency; quarterly for more active risk management. Monthly is generally over-trading for most portfolios.",
        "how":      "At the rebalancing date: (1) Calculate current portfolio weights. (2) Compare to target. (3) Sell overweight positions, buy underweight. (4) If using dividend income or new contributions, direct cash flow to underweight positions first before selling anything.",
        "why":      "Simple, predictable, and requires no continuous monitoring. Ensures portfolio risk doesn't drift for more than one period. Particularly suitable for tax-advantaged accounts (no capital gains concern) where simplicity of execution is valued.",
        "risks":    "Ignores the magnitude of drift — you might rebalance when positions are only 1% off target (unnecessary trading costs) and miss rebalancing when they are 15% off target mid-period. Less responsive to sudden market events.",
        "trigger_conditions": [
            "Scheduled quarterly review date reached",
            "Annual tax year end (for tax-loss harvesting alignment)",
            "New contributions or withdrawals of >5% of portfolio value",
        ],
        "related":  ["threshold_rebalancing", "tax_loss_harvesting"],
    },

    "tax_loss_harvesting": {
        "label":    "Tax-Loss Harvesting (TLH)",
        "category": "rebalancing",
        "what":     "Selling securities at a loss to realise a capital loss for tax purposes, then reinvesting in a similar (but not substantially identical) security to maintain market exposure. The tax saving offsets some of the loss.",
        "when":     "Applicable only in taxable accounts. Triggered when: (1) A position is down more than 5-10% from cost basis AND (2) The portfolio has realised capital gains elsewhere to offset OR you have a tax loss carryforward. Apply primarily in November-December for year-end tax planning.",
        "how":      "Step 1: Identify positions with unrealised losses. Step 2: Sell the position to realise the loss. Step 3: Immediately buy a correlated but not 'substantially identical' security to maintain market exposure. Example: sell SPY (S&P 500), buy IVV or VOO (also S&P 500 ETFs from different issuers — allowed). Step 4: After 30 days (wash-sale rule), optionally switch back to the original holding.",
        "why":      "A capital loss can offset capital gains dollar-for-dollar, and up to $3,000 per year of ordinary income. In high-tax jurisdictions, harvesting a $10,000 loss can save $2,380 in federal tax (at 23.8% long-term capital gains rate). The savings compound if reinvested. Vanguard estimates TLH adds 0.5-1.5% in after-tax returns annually.",
        "risks":    "Wash-sale rule (US): if you buy the same or 'substantially identical' security within 30 days before or after the sale, the loss is disallowed. Transaction costs eat into savings for small positions. Not applicable in tax-advantaged accounts (IRA, 401k) — no capital gains in these accounts.",
        "trigger_conditions": [
            "Position is down more than 5% from cost basis in a taxable account",
            "Realised capital gains exist elsewhere in the portfolio this tax year",
            "It is October-December (year-end tax planning window)",
            "Loss exceeds transaction costs by at least 2×",
        ],
        "related":  ["threshold_rebalancing", "calendar_rebalancing", "asset_allocation"],
    },

    # ── Asset Allocation Frameworks ───────────────────────────────────────────

    "sixty_forty": {
        "label":    "60/40 Portfolio (Classic Balanced)",
        "category": "asset_allocation",
        "what":     "The 60/40 portfolio allocates 60% to equities (typically broad market index) and 40% to bonds (typically investment-grade). It is designed to provide equity-like growth while bonds buffer drawdowns.",
        "when":     "Appropriate for: moderate risk tolerance, 10-20 year investment horizon, investors who need to avoid severe drawdowns but still want inflation-beating returns. Historically suited to environments where equities and bonds are negatively correlated (the norm from 2000-2020).",
        "how":      "60% SPY or VTI (total US equity), 40% AGG or BND (investment-grade bonds). Rebalance annually or at ±5pp threshold. In practice, extend to include international equities (add VXUS) and TIPS for inflation protection.",
        "why":      "Between 1926 and 2020, 60/40 delivered an average annual return of approximately 8.8% with significantly lower volatility than 100% equity. The bond allocation acted as a shock absorber — bonds rally in recessions when equities fall, because the Fed cuts rates.",
        "risks":    "The 60/40 broke down in 2022: bonds AND equities fell simultaneously as the Fed raised rates aggressively. In a high-inflation, rising-rate environment, bonds fail to hedge equities. The model assumes equity-bond negative correlation which is regime-dependent. Consider adding real assets (commodities, REITs) to reduce this regime dependency.",
        "trigger_conditions": [
            "Moderate risk investor (5-10 year horizon)",
            "Yield curve is positively sloped (bonds provide carry)",
            "Inflation is below 3% (bonds retain real value)",
        ],
        "related":  ["risk_parity", "all_weather", "threshold_rebalancing"],
    },

    "risk_parity": {
        "label":    "Risk Parity",
        "category": "asset_allocation",
        "what":     "Risk parity allocates capital so that each asset class contributes equally to portfolio risk (measured by volatility), rather than contributing equal dollar amounts. Because bonds have much lower volatility than equities, bonds are overweighted (often with leverage) to achieve equal risk contribution.",
        "when":     "Most appropriate for: institutional investors, long time horizons, and environments with stable or declining rates. Risk parity is most compelling when the expected Sharpe ratios of all asset classes are similar — the strategy then maximises diversification without tilting toward any single risk factor.",
        "how":      "Step 1: Estimate volatility for each asset class. Step 2: Calculate risk contribution weights: weight_i = (1/vol_i) / sum(1/vol_j). Step 3: Apply weights (often using leverage to boost bond returns). Example: equities vol 20%, bonds vol 6% → risk-parity weight: bonds get 3.3× more weight than equities. Typical result: ~25% equities, 50%+ bonds (levered), 10-15% commodities.",
        "why":      "Ray Dalio's Bridgewater research showed that risk parity delivered better risk-adjusted returns than 60/40 over long periods because it is more genuinely diversified across economic environments. The diversification benefit comes from holding assets that each perform well in a different macro regime.",
        "risks":    "Leverage amplifies losses in market dislocations. In 2022, rising rates hurt the levered bond component severely. Requires sophisticated rebalancing and access to leverage (margin or futures). Not suitable for simple retail accounts.",
        "trigger_conditions": [
            "Investor seeks maximum diversification across macro environments",
            "Rates are stable or falling (bonds provide both income and appreciation)",
            "Investor has access to leverage and understands the risks",
        ],
        "related":  ["all_weather", "sixty_forty", "threshold_rebalancing"],
    },

    "all_weather": {
        "label":    "All-Weather Portfolio (Dalio)",
        "category": "asset_allocation",
        "what":     "Ray Dalio's All-Weather portfolio is designed to perform in all four economic environments: (1) rising growth, (2) falling growth, (3) rising inflation, (4) falling inflation. Allocation: 30% equities, 40% long-term bonds, 15% intermediate bonds, 7.5% gold, 7.5% commodities.",
        "when":     "Designed as a permanent strategic allocation — not a tactical strategy. Suitable for investors who cannot predict which macro environment comes next and want resilience across all scenarios. Particularly appropriate during periods of macro uncertainty.",
        "how":      "Buy and hold with annual rebalancing. Equity component: VTI or SPY. Bond components: TLT (long-term) and IEF (intermediate). Gold: GLD. Commodities: DJP or GSG. Rebalance annually back to target weights.",
        "why":      "Backtests show All-Weather delivered approximately 7-8% annual return with a maximum drawdown of approximately −12% (2020 COVID crash). It severely underperformed in 2022 (−21%) when both bonds and equities fell, but historically the diversification across gold and commodities provided partial protection.",
        "risks":    "Heavy bond allocation underperforms in rising rate environments. Returns are lower than pure equity in bull markets. The portfolio was designed for an era of lower inflation — future performance with structurally higher inflation is uncertain.",
        "trigger_conditions": [
            "Investor cannot tolerate more than 15-20% drawdown",
            "Macro environment is uncertain or transitioning between regimes",
            "Investor has a very long time horizon (20+ years)",
        ],
        "related":  ["risk_parity", "sixty_forty", "threshold_rebalancing"],
    },

    "core_satellite": {
        "label":    "Core-Satellite Portfolio",
        "category": "asset_allocation",
        "what":     "The core-satellite approach splits the portfolio into a low-cost passive core (70-80% of assets) tracking the market, and a smaller satellite (20-30%) of active or thematic positions for additional return potential.",
        "when":     "Appropriate when: the investor wants market-rate returns as a baseline but also has conviction in specific sectors, themes, or individual stocks. Combines the efficiency of passive investing with the return potential of active selection, while limiting the damage if active bets go wrong.",
        "how":      "Core (70-80%): broad market ETFs — SPY, VTI, VXUS (international), AGG (bonds). Satellite (20-30%): individual stocks, sector ETFs, thematic ETFs, or alternatives. The satellite is the only part requiring active monitoring and periodic rebalancing. The core is held passively.",
        "why":      "Research shows that most active managers underperform their benchmark over 10+ years. The core locks in market returns cheaply. The satellite allows expression of views where the investor has genuine edge or conviction, while limiting total portfolio impact if those bets fail.",
        "risks":    "Requires discipline to keep satellite exposure limited. Investors often increase the satellite after early success, gradually eliminating the core's stabilising effect. The satellite should be treated as a risk budget, not a speculation budget.",
        "trigger_conditions": [
            "Investor has specific sector or company conviction",
            "Investor wants to limit tracking error vs benchmark",
            "Investment horizon >5 years for the core component",
        ],
        "related":  ["sixty_forty", "threshold_rebalancing", "concentration_risk"],
    },

    # ── Risk Management Rules ─────────────────────────────────────────────────

    "position_sizing": {
        "label":    "Position Sizing Rules",
        "category": "risk_management",
        "what":     "Position sizing rules define the maximum allocation to any single position, sector, or correlated group to prevent concentration risk from dominating portfolio outcomes.",
        "when":     "Applied at portfolio construction and enforced at rebalancing. Common rules: single stock ≤5% (conservative) or ≤10% (moderate), single sector ≤25%, correlated cluster (assets with correlation >0.7) ≤30% collectively.",
        "how":      "Define a risk budget per position before buying. Kelly Criterion (academic): optimal position = (expected return − risk-free rate) / variance. In practice, use simplified rules: 1/N equal weight as a starting point, then deviate only with strong justification. For high-conviction positions, accept maximum 2-3× the equal-weight size.",
        "why":      "Single-stock risk is largely uncompensated (company-specific risk that the market doesn't reward). Academic research shows no return premium for concentration — only higher variance. Position limits force diversification across sources of return.",
        "risks":    "Mechanically capping position sizes can prevent full participation in high-conviction winners. A rule that caps NVDA at 5% in 2023 would have significantly reduced gains. The trade-off is asymmetric: large losses from concentration are permanent; missed gains are not.",
        "trigger_conditions": [
            "Any single position grows above 10% of portfolio (sell to trim)",
            "Any sector exceeds 30% of portfolio",
            "Two positions with correlation >0.8 collectively exceed 25%",
        ],
        "related":  ["concentration_risk", "threshold_rebalancing", "diversification"],
    },

    "when_to_rebalance": {
        "label":    "When Rebalancing Is Beneficial — Decision Framework",
        "category": "rebalancing",
        "what":     "A decision framework that identifies when the expected benefit of rebalancing (risk reduction, mean reversion capture) exceeds its costs (transaction costs, taxes, bid-ask spreads).",
        "when":     "Rebalancing is most beneficial when: (1) drift is large (>5pp), (2) the market is volatile (reversion more likely), (3) it can be done tax-efficiently, and (4) assets have low momentum (trending assets benefit from letting drift run). It is LEAST beneficial when: drift is small (<3pp), assets are strongly trending, or transaction costs are high relative to portfolio size.",
        "how":      "Decision tree: STEP 1: Is drift >5pp on any position? If No → do not rebalance. If Yes → STEP 2: Is the drifting asset in a strong momentum regime (>+20% in 90 days)? If Yes → widen the band to 7pp and delay. If No → STEP 3: Can the rebalance be done in a tax-advantaged account? If Yes → rebalance immediately. If No → STEP 4: Do transaction costs + tax cost exceed expected drift correction benefit? If cost > benefit → delay or use new contributions instead of selling.",
        "why":      "Rebalancing too frequently increases costs without proportional risk reduction. Rebalancing too infrequently allows concentration risk to compound. The 5pp threshold is empirically validated by Vanguard and T. Rowe Price research as the approximate crossover point where benefit exceeds cost for typical retail portfolio sizes.",
        "risks":    "Mechanical rebalancing in strongly trending markets (2017 US equities, 2020-2021 tech) consistently sells winners prematurely. Incorporate a momentum filter — if an asset has risen >20% in 90 days, widen the rebalancing band.",
        "trigger_conditions": [
            "Drift exceeds 5pp AND asset is not in strong momentum regime",
            "Portfolio beta has deviated >0.3 from target",
            "Macro regime has changed (e.g. risk_on → risk_off) requiring allocation shift",
            "New contributions or distributions are being made (use cash flows first)",
        ],
        "related":  ["threshold_rebalancing", "calendar_rebalancing", "tax_loss_harvesting", "asset_allocation"],
    },
}


# =============================================================================
# STRATEGY STORE
# =============================================================================

class StrategyStore:
    """
    Manages the static strategy and framework knowledge base.

    Provides:
      generate_chunks() — for RAG ingestion (Intent 2 primarily)
      lookup(key)       — for specific framework retrieval
      search(query)     — keyword search
      get_rebalance_justification(signals) — generates a justified rebalancing
                          recommendation based on detected drift signals from kb2
    """
 
    def __init__(self):
        self._chunks: list[dict] = []
 
    def lookup(self, key: str) -> dict | None:
        return STRATEGIES.get(key)
 
    def search(self, query: str) -> list[dict]:
        q_words = set(query.lower().split())
        results = []
        for key, s in STRATEGIES.items():
            text = (s["label"] + " " + s["what"] + " " +
                    s.get("when", "") + " " + s.get("why", "") + " " +
                    " ".join(s.get("related", []))).lower()
            score = sum(1 for w in q_words if w in text)
            if score > 0:
                results.append({"key": key, "score": score, **s})
        return sorted(results, key=lambda x: x["score"], reverse=True)
 
    def get_rebalance_justification(
        self,
        drift_signals: list[dict],
        regime:        str = "neutral",
    ) -> str:
        if not drift_signals:
            return ("No rebalancing currently warranted. All positions are within "
                    "their target bands.")
        urgency_map = {
            "risk_off":    "HIGH — risk-off conditions amplify the importance of staying within target risk bands.",
            "stagflation": "HIGH — stagflation is punishing to overweight positions in both equities and bonds.",
            "rate_stress": "MEDIUM-HIGH — rising rates create sharp drawdowns in rate-sensitive positions.",
            "risk_on":     "MEDIUM — risk-on conditions reduce urgency but allow rebalancing at favourable prices.",
            "recovery":    "MEDIUM — recovery phase is a good time to rebalance back to growth allocations.",
            "neutral":     "MEDIUM — standard threshold-based rebalancing applies.",
        }
        urgency = urgency_map.get(regime, "MEDIUM — standard rules apply.")
        actions = [
            f"{s['action_needed']}: current {s['current_weight']*100:.1f}% "
            f"vs target {s['target_weight']*100:.1f}% "
            f"(drift {s['drift']*100:+.1f}pp)"
            for s in drift_signals
        ]
        fw = STRATEGIES["when_to_rebalance"]
        return (
            f"Rebalancing recommendation (urgency: {urgency})\n\n"
            f"Drift detected in {len(drift_signals)} position(s):\n"
            + "\n".join(f"  • {a}" for a in actions)
            + f"\n\nFramework: {fw['label']}. {fw['how']}\n\nJustification: {fw['why']}"
        )
 
    # ── Chunk generation + cleaning ───────────────────────────────────────────
 
    def generate_chunks(self) -> list[dict]:
        if self._chunks:
            return self._chunks
 
        now    = datetime.datetime.now().isoformat()
        chunks = []
 
        for key, s in STRATEGIES.items():
            trigger_str = (
                "Trigger conditions: " + "; ".join(s["trigger_conditions"]) + ". "
                if s.get("trigger_conditions") else ""
            )
            related_str = (
                "Related frameworks: " + ", ".join(s.get("related", [])) + "."
                if s.get("related") else ""
            )
            raw_text = (
                f"{s['label']} ({s.get('category', 'general')}): "
                f"WHAT: {s['what']} "
                f"WHEN: {s.get('when', '')} "
                f"HOW: {s.get('how', '')} "
                f"WHY: {s.get('why', '')} "
                f"RISKS: {s.get('risks', '')} "
                f"{trigger_str}{related_str}"
            )
            chunks.append({
                "citation_id":  f"strategy-{key}",
                "kb_source":    "strategies",
                "intent_tags":  ["rebalance", "full_analysis"],
                "source_url":   "internal://strategies",
                "source_type":  "curated_static",
                "updated_at":   now,
                "strategy_key": key,
                "category":     s.get("category", "general"),
                "label":        s["label"],
                "text":         _clean_chunk(raw_text),
            })
 
        self._chunks = chunks
        return chunks
 
    # ── TXT export ────────────────────────────────────────────────────────────
 
    def export_txt(self, output_path: str = OUTPUT_TXT) -> str:
        chunks = self.generate_chunks()
        now    = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
 
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"# kb4 STRATEGIES — {len(chunks)} frameworks — generated {now}\n\n")
            for c in chunks:
                f.write(f"--- {c['strategy_key']} [{c['category']}] ---\n")
                f.write(f"Label: {c['label']}\n")
                f.write(c["text"] + "\n\n")
 
        print(f"  ✓ kb4 TXT → {output_path}  ({len(chunks)} strategies)")
        return output_path
 
    def save_json(self) -> str:
        with open(STRATEGIES_FILE, "w", encoding="utf-8") as f:
            json.dump(STRATEGIES, f, indent=2)
        print(f"  ✓ kb4 JSON → {STRATEGIES_FILE}")
        return STRATEGIES_FILE
 
 
# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    ss = StrategyStore()
    chunks = ss.generate_chunks()
    ss.export_txt()
    ss.save_json()
    print(f"\n✅ {len(chunks)} strategy chunks ready")
    for c in chunks:
        print(f"  [{c['strategy_key']}] {c['label']}")