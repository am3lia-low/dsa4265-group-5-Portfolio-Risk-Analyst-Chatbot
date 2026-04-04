"""
kb0_ticker_resolver.py 
===============================
Resolves company names / natural language to yfinance ticker symbols.

Strategy (in order of priority):
  1. Exact match against a built-in alias map (fast, no network)
  2. yfinance search API (yf.Search) — works for most listed names
  3. Fallback: Yahoo Finance search endpoint via requests

Usage:
    from kb0_ticker_resolver import resolve_tickers_from_query
    tickers = resolve_tickers_from_query("How is Lululemon progressing recently?")
    # → ["LULU"]
"""

import re
import yfinance as yf

# ---------------------------------------------------------------------------
# 1. Static alias map  (common names/misspellings → ticker)
#    Extend this freely — it's just a dict.
# ---------------------------------------------------------------------------
ALIAS_MAP= { #dict[str, str] 
    # ── Tech ────────────────────────────────────────────────────────────────
    "apple":         "AAPL",
    "microsoft":     "MSFT",
    "google":        "GOOGL",
    "alphabet":      "GOOGL",
    "amazon":        "AMZN",
    "nvidia":        "NVDA",
    "meta":          "META",
    "facebook":      "META",
    "netflix":       "NFLX",
    "salesforce":    "CRM",
    "adobe":         "ADBE",
    "oracle":        "ORCL",
    "intel":         "INTC",
    "amd":           "AMD",
    "advanced micro devices": "AMD",
    "qualcomm":      "QCOM",
    "broadcom":      "AVGO",
    "texas instruments": "TXN",
    "servicenow":    "NOW",
    "snowflake":     "SNOW",
    "palantir":      "PLTR",
    "uber":          "UBER",
    "lyft":          "LYFT",
    "airbnb":        "ABNB",
    "spotify":       "SPOT",
    "paypal":        "PYPL",
    "block":         "SQ",
    "square":        "SQ",
    "shopify":       "SHOP",
    "twilio":        "TWLO",
    "datadog":       "DDOG",
    "crowdstrike":   "CRWD",
    "palo alto":     "PANW",
    "palo alto networks": "PANW",
    "fortinet":      "FTNT",
    "cloudflare":    "NET",
    "mongodb":       "MDB",
    "elastic":       "ESTC",
    "okta":          "OKTA",
    "workday":       "WDAY",
    "veeva":         "VEEV",
    "zoom":          "ZM",
    "slack":         "WORK",
    "twitter":       "X",    # now X, but yf still uses X
    "x":             "X",
    # ── Consumer / Retail ───────────────────────────────────────────────────
    "tesla":         "TSLA",
    "walmart":       "WMT",
    "costco":        "COST",
    "target":        "TGT",
    "home depot":    "HD",
    "lowes":         "LOW",
    "lowe's":        "LOW",
    "dollar general": "DG",
    "dollar tree":   "DLTR",
    "kroger":        "KR",
    "nike":          "NKE",
    "lululemon":     "LULU",
    "lulu":          "LULU",
    "gap":           "GAP",
    "tapestry":      "TPR",
    "ralph lauren":  "RL",
    "pvh":           "PVH",
    "under armour":  "UA",
    "columbia sportswear": "COLM",
    "starbucks":     "SBUX",
    "mcdonald's":    "MCD",
    "mcdonalds":     "MCD",
    "yum brands":    "YUM",
    "chipotle":      "CMG",
    "darden":        "DRI",
    "domino's":      "DPZ",
    "dominos":       "DPZ",
    "papa john's":   "PZZA",
    "dutch bros":    "BROS",
    "wingstop":      "WING",
    # ── Financials ──────────────────────────────────────────────────────────
    "jpmorgan":      "JPM",
    "jp morgan":     "JPM",
    "bank of america": "BAC",
    "goldman sachs": "GS",
    "morgan stanley": "MS",
    "wells fargo":   "WFC",
    "citigroup":     "C",
    "citi":          "C",
    "visa":          "V",
    "mastercard":    "MA",
    "american express": "AXP",
    "amex":          "AXP",
    "blackrock":     "BLK",
    "charles schwab": "SCHW",
    "fidelity":      "FNF",
    "td bank":       "TD",
    "us bancorp":    "USB",
    # ── Healthcare / Pharma ─────────────────────────────────────────────────
    "johnson & johnson": "JNJ",
    "j&j":           "JNJ",
    "pfizer":        "PFE",
    "merck":         "MRK",
    "abbvie":        "ABBV",
    "bristol myers": "BMY",
    "eli lilly":     "LLY",
    "lilly":         "LLY",
    "moderna":       "MRNA",
    "biontech":      "BNTX",
    "unitedhealth":  "UNH",
    "cvs":           "CVS",
    "walgreens":     "WBA",
    "astrazeneca":   "AZN",
    "novartis":      "NVS",
    "roche":         "RHHBY",
    "amgen":         "AMGN",
    "gilead":        "GILD",
    "vertex":        "VRTX",
    "regeneron":     "REGN",
    "illumina":      "ILMN",
    "intuitive surgical": "ISRG",
    "medtronic":     "MDT",
    "abbott":        "ABT",
    "becton dickinson": "BDX",
    "edwards lifesciences": "EW",
    "humana":        "HUM",
    "cigna":         "CI",
    # ── Energy ──────────────────────────────────────────────────────────────
    "exxon":         "XOM",
    "exxon mobil":   "XOM",
    "chevron":       "CVX",
    "shell":         "SHEL",
    "bp":            "BP",
    "conocophillips": "COP",
    "slb":           "SLB",
    "schlumberger":  "SLB",
    "halliburton":   "HAL",
    "pioneer natural": "PXD",
    "devon energy":  "DVN",
    "diamondback":   "FANG",
    "nextera":       "NEE",
    "duke energy":   "DUK",
    "southern company": "SO",
    # ── Industrials ─────────────────────────────────────────────────────────
    "caterpillar":   "CAT",
    "boeing":        "BA",
    "general electric": "GE",
    "ge":            "GE",
    "honeywell":     "HON",
    "3m":            "MMM",
    "lockheed":      "LMT",
    "lockheed martin": "LMT",
    "raytheon":      "RTX",
    "northrop":      "NOC",
    "northrop grumman": "NOC",
    "general dynamics": "GD",
    "deere":         "DE",
    "john deere":    "DE",
    "union pacific": "UNP",
    "ups":           "UPS",
    "fedex":         "FDX",
    "delta":         "DAL",
    "delta air":     "DAL",
    "american airlines": "AAL",
    "united airlines": "UAL",
    "southwest":     "LUV",
    # ── Materials / Real Estate ──────────────────────────────────────────────
    "freeport":      "FCX",
    "freeport mcmoran": "FCX",
    "newmont":       "NEM",
    "barrick":       "GOLD",
    "prologis":      "PLD",
    "simon property": "SPG",
    "american tower": "AMT",
    "crown castle":  "CCI",
    "equinix":       "EQIX",
    "weyerhaeuser":  "WY",
    # ── Communications / Media ───────────────────────────────────────────────
    "at&t":          "T",
    "att":           "T",
    "verizon":       "VZ",
    "t-mobile":      "TMUS",
    "tmobile":       "TMUS",
    "comcast":       "CMCSA",
    "disney":        "DIS",
    "fox":           "FOX",
    "paramount":     "PARA",
    "warner bros":   "WBD",
    "charter":       "CHTR",
    # ── ETFs / Benchmarks ────────────────────────────────────────────────────
    "s&p 500":       "SPY",
    "s&p500":        "SPY",
    "spy":           "SPY",
    "gold":          "GLD",
    "gld":           "GLD",
    "bonds":         "TLT",
    "tlt":           "TLT",
    "qqq":           "QQQ",
    "nasdaq":        "QQQ",
    "dow jones":     "DIA",
    "dia":           "DIA",
    "iwm":           "IWM",
    "russell 2000":  "IWM",
    "vanguard":      "VTI",
    "vti":           "VTI",
    "ark":           "ARKK",
    "ark innovation": "ARKK",
    "arkk":          "ARKK",
    # ── Crypto-adjacent ─────────────────────────────────────────────────────
    "coinbase":      "COIN",
    "microstrategy": "MSTR",
    "bitcoin etf":   "IBIT",
    "ibit":          "IBIT",
}

# ---------------------------------------------------------------------------
# 2. Regex: only match tokens that are ALREADY fully upper-case in the
#    original query, e.g. "NVDA", "TSLA" — NOT "How", "What", "Goldman"
#    which only becomes caps when uppercased for comparison.
# ---------------------------------------------------------------------------
# Matches 2-5 consecutive uppercase ASCII letters surrounded by word boundaries.
# Single letters (I, A) and 6+ letter runs are excluded to reduce false hits.
ALLCAPS_TICKER_PATTERN = re.compile(r"\b([A-Z]{2,5})\b")

# Words to exclude so we don't mistake common English words for tickers
# Stopwords; all uppercase.  These are checked BEFORE the yfinance round-trip.
STOPWORDS = {
    # Question / pronoun words
    "AM", "AN", "AS", "AT", "BE", "BY", "DO", "GO", "IF",
    "IN", "IS", "IT", "ME", "MY", "NO", "OF", "ON", "OR", "SO", "TO",
    "UP", "US", "WE", "AND", "ARE", "FOR", "HAS", "HOW", "ITS", "NOT",
    "THE", "WAS", "WHO", "WHY", "YOU", "WHAT", "WITH", "THIS", "THAT",
    "WILL", "FROM", "HAVE", "THEY", "BEEN", "DOES", "OVER", "THAN",
    "THEN", "WHEN", "INTO", "LIKE", "JUST", "ALSO", "SOME", "VERY",
    "MOST", "MUCH", "BOTH", "EACH", "SUCH", "WELL", "ONLY", "ANY",
    "CAN", "GET", "DID", "HIM", "HIS", "HER", "OUR", "OUT", "NOW",
    "WAY", "MAY", "NEW", "OLD", "ALL", "ONE", "TWO", "BAD", "BIG",
    "SAY", "SEE", "LET", "PUT", "SET", "TELL", "MAKE", "TAKE", "GIVE",
    "COME", "LOOK", "WANT", "KNOW", "THINK", "FEEL", "SEEM", "KEEP",
    "SHOW", "HOLD", "MOVE", "FIND", "CALL", "HIGH", "LOW", "LONG",
    "GOOD", "BEST", "LAST", "NEXT", "SAME", "REAL", "MAIN", "FIRST",
    "YEAR", "WEEK", "DAYS", "TIME", "PAST", "NEAR", "LATE", "BACK",
    "DOWN", "EVEN", "FULL", "OPEN", "SIDE", "HARD", "LIVE", "HAND",
    "MANY", "EVER", "ABLE", "MORE", "LESS", "SINCE",
    # Finance / generic terms that look like tickers
    "BUY", "SELL", "RISK", "RATE", "FUND", "DEBT", "CASH", "EARN",
    "GAIN", "LOSS", "BULL", "BEAR", "MARK", "PLAN", "NEWS", "DATA",
    "VIEW", "HELP", "USED", "FORM", "PART", "PAID", "SAID", "DONE",
    "ROSE", "FELL", "DROP", "JUMP", "BEAT", "MISS", "MEET", "POST",
    "NEED", "MEAN", "BASE", "HALF", "ZERO", "PLUS", "GROW",
    # Time / quantity words
    "FIVE", "FOUR", "NINE", "WEEK", "DAYS",
    # Common abbreviations that are NOT tickers
    "CEO", "CFO", "COO", "IPO", "GDP", "CPI", "PPI", "FED", "SEC",
    "ETF", "NAV", "EPS", "TTM", "YTD", "ATH", "ATL", "RSI", "ATR",
    "SMA", "EMA", "PE", "PB", "PS", "FCF", "ROE", "ROA", "DCF",
    "USA", "USD", "EUR", "GBP", "JPY", "CAD", "AUD",
    "NYC", "LA", "SF", "DC", "UK", "EU", "US",
    # Common English words that yfinance may erroneously validate
    "NOW", "NET", "WORK", "WELL", "BACK", "HOLD", "COST", "OPEN",
    "REAL", "LIVE", "BASE", "MOVE", "CALL", "SHOW",
}

# ---------------------------------------------------------------------------
# 3. yfinance search fallback
# ---------------------------------------------------------------------------

def _yf_search(name: str) -> str | None:
    """
    Use yfinance's Search API to look up a company name.
    Returns the best-match ticker string, or None.
 
    Guard: the returned ticker's longname/shortname must contain at least one
    word from the search term (case-insensitive), otherwise reject it.
    This prevents e.g. searching "Goldman" and getting back a random
    foreign ETF whose name shares no words with the query.
    """
    name_words = set(re.sub(r"[^a-z ]", "", name.lower()).split())
 
    def _name_matches(result: dict) -> bool:
        """True if any word in `name` appears in the result's display name."""
        display = (
            result.get("longname", "") or result.get("shortname", "")
        ).lower()
        return any(w in display for w in name_words if len(w) > 2)
 
    try:
        results = yf.Search(name, max_results=5).quotes
        # First pass: US-listed equity/ETF with name match
        for r in results:
            exchange = r.get("exchange", "")
            q_type   = r.get("quoteType", "")
            symbol   = r.get("symbol", "")
            if not symbol or "." in symbol:   # skip foreign tickers (e.g. G1J.F, WHATS.BR)
                continue
            if q_type in ("EQUITY", "ETF") and exchange in (
                "NMS", "NYQ", "NGM", "NCM", "PCX", "BATS", "ASE"
            ) and _name_matches(r):
                return symbol
        # Second pass: any equity with name match (relaxed exchange filter)
        for r in results:
            symbol = r.get("symbol", "")
            if not symbol or "." in symbol:
                continue
            if r.get("quoteType") in ("EQUITY", "ETF") and _name_matches(r):
                return symbol
    except Exception:
        pass
    return None

# ---------------------------------------------------------------------------
# 4. Main resolver
# ---------------------------------------------------------------------------

def resolve_tickers_from_query(query: str) -> list[str]:
    """
    Parse a natural-language query and return a deduplicated list of
    resolved ticker symbols.
 
    Pass 1 — Alias map, word-boundary matched (longest alias first).
    Pass 2 — Raw ALL-CAPS tokens already present in the original query
             (e.g. "NVDA", "TSLA") validated via yfinance.
    Pass 3 — Multi-word title-case phrases via yf.Search
             (e.g. "Dutch Bros Coffee", "Warner Bros Discovery").
    Pass 4 — Single title-case words not yet matched, via yf.Search,
             with name-match guard.
    """
    found: dict[str, bool] = {}   # ticker → True (ordered dict as set)
    q_lower = query.lower()

    # ── Pass 1: alias map with word-boundary check ──
    # Sort longest-first so "bank of america" is tried before "bank"
    sorted_aliases = sorted(ALIAS_MAP.keys(), key=len, reverse=True)
    # Track which character spans are already consumed so sub-aliases don't
    # double-match inside a longer match (e.g. "gold" inside "goldman sachs")
    
    consumed_spans: list[tuple[int, int]] = []
 
    for alias in sorted_aliases:
        # Use word boundaries; escape special chars in alias (e.g. "s&p 500")
        pattern = r"(?<![a-z])" + re.escape(alias) + r"(?![a-z])"
        for m in re.finditer(pattern, q_lower):
            start, end = m.start(), m.end()
            # Skip if this span overlaps a longer match already consumed
            if any(cs <= start and end <= ce for cs, ce in consumed_spans):
                continue
            found[ALIAS_MAP[alias]] = True
            consumed_spans.append((start, end))
            break   # alias matched — move on to next alias

    # ── Pass 2: ALL-CAPS tokens that were literally typed in the query ──
    # e.g. the user typed "NVDA" or "TSLA" — NOT words like "How" uppercased
    for token in ALLCAPS_TICKER_PATTERN.findall(query):
        if token in STOPWORDS:
            continue
        if token in found:
            continue
        # Validate: yfinance must return market data for this symbol
        try:
            info = yf.Ticker(token).info
            price_field = (
                info.get("regularMarketPrice")
                or info.get("currentPrice")
                or info.get("previousClose")   # ETFs sometimes only have this
            )
            if price_field:
                found[token] = True
        except Exception:
            pass

    # ── Pass 3: multi-word title-case phrases (e.g. "Dutch Bros Coffee") ──
    cap_phrases = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", query)
    for phrase in cap_phrases:
        phrase_lower = phrase.lower()
        # Skip if any word of the phrase was already matched by the alias map
        if any(phrase_lower in a or a in phrase_lower for a in ALIAS_MAP):
            continue
        ticker = _yf_search(phrase)
        if ticker and ticker not in found:
            found[ticker] = True

    # ── Pass 4: single title-case word not yet matched ──
    # e.g. "Lululemon" typed in title-case — alias map handles lowercase,
    # but a user might type it capitalised without it being all-caps.
    cap_words = re.findall(r"\b([A-Z][a-z]{2,})\b", query)
    for word in cap_words:
        word_lower = word.lower()
 
        # Already handled by alias map in Pass 1?
        if word_lower in ALIAS_MAP:
            t = ALIAS_MAP[word_lower]
            if t not in found:
                found[t] = True
            continue

        # Skip common English words — not worth a network call
        if word_lower in {
            "how", "what", "when", "where", "why", "who", "which",
            "this", "that", "these", "those", "will", "would", "could",
            "should", "have", "does", "doing", "been", "being", "were",
            "their", "there", "about", "some", "much", "many", "most",
            "more", "less", "also", "just", "very", "really", "quarter",
            "recently", "today", "week", "month", "year", "stock",
            "market", "price", "trade", "share", "sector", "analyst",
            "outlook", "earnings", "guidance", "momentum", "growth",
            "compare", "tell", "show", "think", "progressing", "working",
            "doing", "with", "from", "into", "over", "than", "then",
            "like", "well", "good", "best", "next", "last",
        }:
            continue
 
        ticker = _yf_search(word)
        if ticker and ticker not in found:
            found[ticker] = True
 
    return list(found.keys())

# ---------------------------------------------------------------------------
# 5. Quick CLI test
# ---------------------------------------------------------------------------
#if __name__ == "__main__":
#    test_queries = [
#        "How is Lululemon progressing recently?",
#        "Compare NVDA and AMD. Which has better momentum?",
#        "Tell me about Apple's earnings and Goldman Sachs guidance",
#        "What's the outlook for Dutch Bros Coffee?",
#        "Is the S&P 500 overbought right now?",
#        "How is Tesla doing this quarter?",
#        "What do analysts think about Shopify?",
#    ]
#    for q in test_queries:
#        tickers = resolve_tickers_from_query(q)
#        print(f"Q: {q!r}\n   → {tickers}\n")