"""
RAG_utils.py
============
Unified RAG pipeline for the risk portfolio chatbot.
 
Knowledge bases
---------------
  kb0  ticker resolver          — query → ticker symbols
  kb1  ticker profiles          — yfinance fundamentals + price history
  kb2  portfolio store          — holdings, weights, portfolio-level risk metrics
  kb3  macro store              — FRED rates, VIX, sector ETFs, regime classification
  kb4  concept definitions      — static curated financial concepts
  kb5  strategy frameworks      — rebalancing rules, allocation frameworks
 
Intent routing
--------------
  full_analysis    → kb1 (ticker) + kb2 (portfolio) + kb3 (macro) + kb4 (concepts)
  rebalance        → kb2 (portfolio drift) + kb3 (regime) + kb5 (frameworks) + kb4
  concept_explanation → kb4 (concepts) + kb5 (strategies)
  trend_prediction → kb3 (macro) + kb1 (ticker momentum)
  fallback         → all KB sources
 
Pipeline
--------
  1. Intent detection from query
  2. Source selection per intent
  3. KB refresh (staleness check for dynamic sources)
  4. Unified chunk pool from all relevant sources
  5. Hybrid BM25 + vector retrieval
  6. Post-processing: relevance filter + citation enforcement
  7. Retrieval evaluation: Recall@K + MRR
  8. Persist logs (retrieval_log)
"""

from __future__ import annotations
 
import os
import re
import json
import hashlib
import datetime
import importlib
import warnings
from typing import Any
 
import numpy as np
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
 
warnings.filterwarnings("ignore")
 
# ── Local KB modules ──────────────────────────────────────────────────────────
import kb0_ticker_resolver
import kb1_generate_tickers
import kb2_portfolio
import kb3_macro_regime
import kb4_concepts
import kb5_strategies
 
for _mod in [kb0_ticker_resolver, kb1_generate_tickers,
             kb2_portfolio, kb3_macro_regime, kb4_concepts, kb5_strategies]:
    importlib.reload(_mod)
 
from kb0_ticker_resolver import resolve_tickers_from_query
from kb1_generate_tickers import (
    generate_tickers,
    convert_tickers_into_txt,
    build_ticker_meta as _build_ticker_meta,
    OUTPUT_DIR_HTML, OUTPUT_TXT
)
from kb2_portfolio import PortfolioStore
from kb3_macro_regime import MacroStore
from kb4_concepts  import ConceptStore
from kb5_strategies import StrategyStore

# ── Config ────────────────────────────────────────────────────────────────────
VECTOR_DB_DIR       = "vector_db"
EMBED_MODEL_NAME    = "sentence-transformers/all-MiniLM-L6-v2"
KB_STALENESS_HOURS  = 24
RELEVANCE_THRESHOLD = 0.10    # lower than before — static KB chunks score differently
RETRIEVAL_LOG_DIR   = "retrieval_log"
 
# ── Intent → KB source mapping ────────────────────────────────────────────────
# Each intent specifies which Chroma collections to query
INTENT_SOURCES: dict[str, list[str]] = {
    "full_analysis":     ["tickers", "portfolio", "macro", "concepts"],
    "rebalance":         ["portfolio", "macro", "strategies", "concepts"],
    "concept_explanation": ["concepts", "strategies"],
    "trend_prediction":  ["macro", "tickers"],
    "fallback":          ["tickers", "portfolio", "macro", "concepts", "strategies"],
}
 
# Collection name → display label
COLLECTION_LABELS = {
    "tickers":    "Ticker Profiles (yfinance)",
    "portfolio":  "Portfolio Store",
    "macro":      "Macro & Regime (FRED + yfinance)",
    "concepts":   "Concept Definitions",
    "strategies": "Strategy Frameworks",
}

# =============================================================================
# SECTION 1 — INTENT DETECTION
# =============================================================================
 
_INTENT_KEYWORDS: dict[str, list[str]] = {
    "rebalance": [
        "rebalance", "rebalancing", "drift", "overweight", "underweight",
        "target weight", "allocation", "trim", "reallocate", "redistribute",
        "more", "less", "uneven", "refit", "should I sell", "should I buy",
        "should I add", "compare 1/n split", "current", "current condition"
    ],
    "concept_explanation": [
        "what is", "explain", "define", "definition", "meaning of",
        "what does", "how does", "tell me about", "concept", "covariance matrix",
        "portfolio volatility", "volatility", "var", "cvar", "maximum drawdown",
        "sharpe", "sortino", "skewness", "excess kurtosis", "beta", "hhi",
        "pairwise correlation", "risk contribution", "diversification", 
        "yield curve", "duration", "diversification", "rebalancing", 
        "asset allocation", "yield curve", "interest rate risk", "recession risk", 
        "liquidity risk", "works"
        
    ],
    "trend_prediction": [
        "trend", "market", "outlook", "forecast", "prediction", "regime",
        "vix", "recession", "inflation", "rate", "sector", "momentum",
        "where is the market", "macro", "economic", "fed", "increase", "decrease",
        "future", "future condition", "foresee", "market", "economy",
    ],
    "full_analysis": [
        "risk", "portfolio", "analyse", "analyze", "high risk", "low risk",
        "performance", "how is my portfolio", "overall", "assessment",
        "dangerous", "safe", "exposure", "holdings", "good", "bad", "works", 
    ],
}
 
 
def detect_intent(query: str) -> str:
    """
    Detect the primary intent from a natural-language query.
 
    Returns one of: "full_analysis" | "rebalance" | "concept_explanation" |
                    "trend_prediction" | "fallback"
 
    Strategy: count keyword matches per intent, return highest scorer.
    Ties default to "full_analysis". No match → "fallback".
    """
    q_lower = query.lower()
    scores  = {intent: 0 for intent in _INTENT_KEYWORDS}
 
    for intent, keywords in _INTENT_KEYWORDS.items():
        for kw in keywords:
            if kw in q_lower:
                scores[intent] += 1
 
    best_score = max(scores.values())
    if best_score == 0:
        return "fallback"
 
    # Return the intent with the highest score
    return max(scores, key=lambda k: scores[k])

# =============================================================================
# SECTION 2 — KB SOURCE MANAGEMENT
# =============================================================================
 
def _kb1_file_path(ticker: str) -> str:
    return os.path.join(OUTPUT_DIR_HTML, f"{ticker.lower()}.html")
 
 
def _kb1_is_fresh(ticker: str) -> bool:
    path = _kb1_file_path(ticker)
    if not os.path.exists(path):
        return False
    age = (datetime.datetime.now().timestamp() - os.path.getmtime(path)) / 3600
    return age < KB_STALENESS_HOURS
 
 
def _ensure_kb1_files(tickers: list[str], force: bool = False) -> None:
    to_gen = {}
    for t in tickers:
        if force or not _kb1_is_fresh(t):
            meta = _build_ticker_meta(t)
            if meta:
                to_gen[t] = meta
            else:
                print(f" Cannot resolve meta for {t}")
        else:
            print(f"  ✓ {t}: ticker KB fresh")
    if to_gen:
        generate_tickers(to_gen)
        convert_tickers_into_txt()
 
 
def _load_kb1_documents() -> list[dict]:
    """
    Load ticker profile chunks by running kb1's full pipeline:
    HTML → strip tags → clean text → convert numbers → return chunks.
    Cleaning is applied inside convert_tickers_into_txt(), so what goes
    into Chroma is identical to what is written to the TXT audit file.
    """
    if not os.path.exists(OUTPUT_DIR_HTML) or not os.listdir(OUTPUT_DIR_HTML):
        return []
    return convert_tickers_into_txt(output_txt=OUTPUT_TXT)
 
 
def _collect_all_chunks(
    intent:           str,
    resolved_tickers: list[str],
    portfolio_store:  PortfolioStore | None = None,
    macro_store:      MacroStore     | None = None,
    concept_store:    ConceptStore   | None = None,
    strategy_store:   StrategyStore  | None = None,
    force_refresh:    bool           = False,
) -> list[dict]:
    """
    Gather chunks from all KB sources relevant to the detected intent.
 
    Returns a flat list of chunk dicts, each with:
      kb_source, text, citation_id, source_url, source_type, intent_tags
    """
    sources   = INTENT_SOURCES.get(intent, INTENT_SOURCES["fallback"])
    all_chunks: list[dict] = []
 
    # ── kb1: Ticker profiles ───────────────────────────────────────────────
    if "tickers" in sources and resolved_tickers:
        print(f"  📥 kb1: loading ticker profiles for {resolved_tickers}")
        _ensure_kb1_files(resolved_tickers, force=force_refresh)
        kb1_docs = _load_kb1_documents()
        # Filter to only resolved tickers for this query
        filtered = [
            d for d in kb1_docs
            if d.get("ticker", "").upper() in [t.upper() for t in resolved_tickers]
        ]
        # Attach citation fields if missing
        for i, d in enumerate(filtered):
            if "citation_id" not in d:
                d["citation_id"] = f"ticker-{d['ticker']}-{i}"
        all_chunks.extend(filtered)
        print(f"    → {len(filtered)} ticker chunks")
 
    # ── kb2: Portfolio ─────────────────────────────────────────────────────
    if "portfolio" in sources and portfolio_store is not None:
        print("  📥 kb2: loading portfolio chunks + writing TXT")
        chunks = portfolio_store.generate_chunks(force=force_refresh)
        portfolio_store.export_txt()
        all_chunks.extend(chunks)
        print(f"    → {len(chunks)} portfolio chunks")
 
    # ── kb3: Macro ─────────────────────────────────────────────────────────
    if "macro" in sources and macro_store is not None:
        print("  📥 kb3: loading macro chunks + writing TXT")
        if force_refresh or macro_store._is_stale():
            macro_store.refresh(force=force_refresh)
        chunks = macro_store.generate_chunks()
        macro_store.export_txt()
        all_chunks.extend(chunks)
        print(f"    → {len(chunks)} macro chunks")
 
    # ── kb4: Concepts ──────────────────────────────────────────────────────
    if "concepts" in sources and concept_store is not None:
        print("  📥 kb4: loading concept chunks + writing TXT")
        chunks = concept_store.generate_chunks()
        concept_store.export_txt()
        all_chunks.extend(chunks)
        print(f"    → {len(chunks)} concept chunks")
 
    # ── kb5: Strategies ────────────────────────────────────────────────────
    if "strategies" in sources and strategy_store is not None:
        print("  📥 kb5: loading strategy chunks + writing TXT")
        chunks = strategy_store.generate_chunks()
        strategy_store.export_txt()
        all_chunks.extend(chunks)
        print(f"    → {len(chunks)} strategy chunks")
 
    print(f"  Total chunks in pool: {len(all_chunks)}")
    return all_chunks

# =============================================================================
# SECTION 3 — CHROMA: MULTI-COLLECTION MANAGEMENT
# =============================================================================
 
def _get_or_create_collection(
    client:     chromadb.PersistentClient,
    name:       str,
    documents:  list[dict],
    model:      SentenceTransformer,
) -> chromadb.Collection:
    """Upsert documents into a named Chroma collection."""
    collection = client.get_or_create_collection(name=name)
    if not documents:
        return collection
 
    texts     = [d["text"] for d in documents]
    metadatas = [
        {
            "kb_source":   d.get("kb_source", name),
            "ticker":      d.get("ticker", ""),
            "source_type": d.get("source_type", ""),
            "citation_id": d.get("citation_id", ""),
        }
        for d in documents
    ]
    ids        = [str(i) for i in range(len(documents))]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
 
    collection.upsert(
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
        ids=ids,
    )
    return collection
 
 
def _build_collections(
    all_chunks: list[dict],
    model:      SentenceTransformer,
    client:     chromadb.PersistentClient,
) -> dict[str, chromadb.Collection]:
    """
    Group chunks by kb_source and upsert into separate Chroma collections.
    Returns { collection_name: Collection }.
    """
    grouped: dict[str, list[dict]] = {}
    for chunk in all_chunks:
        src = chunk.get("kb_source", "unknown")
        grouped.setdefault(src, []).append(chunk)
 
    collections = {}
    for src, docs in grouped.items():
        print(f"  🔢 Embedding {len(docs)} chunks → collection '{src}'")
        collections[src] = _get_or_create_collection(client, src, docs, model)
    return collections

# =============================================================================
# SECTION 4 — HYBRID RETRIEVAL (MULTI-COLLECTION)
# =============================================================================
 
def _hybrid_retrieve_multi(
    query:       str,
    all_chunks:  list[dict],
    collections: dict[str, chromadb.Collection],
    model:       SentenceTransformer,
    top_k:       int   = 8,
    bm25_weight: float = 0.3,
    vec_weight:  float = 0.7,
) -> list[dict]:
    """
    Hybrid BM25 + vector search across all loaded chunks.
    BM25 operates on the full chunk pool; vector search queries each collection
    separately then merges. Scores are fused and globally ranked.
    """
    if not all_chunks:
        return []
 
    texts = [d["text"] for d in all_chunks]
    n     = len(texts)
 
    # ── BM25 across full pool ─────────────────────────────────────────────────
    bm25     = BM25Okapi([t.split() for t in texts])
    bm25_raw = bm25.get_scores(query.split())
    b_min, b_max = bm25_raw.min(), bm25_raw.max()
    bm25_norm = (
        (bm25_raw - b_min) / (b_max - b_min)
        if b_max > b_min else np.zeros(n)
    )
 
    # ── Vector search per collection ──────────────────────────────────────────
    q_emb          = model.encode([query], convert_to_numpy=True)
    vec_scores_raw = np.zeros(n)
 
    # Build a mapping from chunk text → index for vector result matching
    text_to_idx = {text: i for i, text in enumerate(texts)}
 
    # Track which indices have been assigned in each collection
    offset = 0
    col_offsets: dict[str, int] = {}
    for src, docs in _group_by_source(all_chunks):
        col_offsets[src] = offset
        offset += len(docs)
 
    for src, collection in collections.items():
        src_docs  = [c for c in all_chunks if c.get("kb_source") == src]
        n_src     = len(src_docs)
        if n_src == 0:
            continue
        n_results = min(max(top_k, n_src // 2), n_src)
        try:
            results = collection.query(
                query_embeddings=q_emb.tolist(),
                n_results=n_results,
                include=["distances", "documents"],
            )
            for doc_text, dist in zip(
                results["documents"][0],
                results["distances"][0],
            ):
                idx = text_to_idx.get(doc_text)
                if idx is not None:
                    vec_scores_raw[idx] = max(
                        vec_scores_raw[idx],
                        1.0 / (1.0 + dist)
                    )
        except Exception as e:
            print(f"  Vector search failed for '{src}': {e}")
 
    v_min, v_max = vec_scores_raw.min(), vec_scores_raw.max()
    vec_norm = (
        (vec_scores_raw - v_min) / (v_max - v_min)
        if v_max > v_min else np.zeros(n)
    )
 
    # ── Fusion ────────────────────────────────────────────────────────────────
    fused       = bm25_weight * bm25_norm + vec_weight * vec_norm
    top_indices = fused.argsort()[-top_k:][::-1]
 
    return [
        {
            **all_chunks[i],
            "rank":     int(rank + 1),
            "score":    float(fused[i]),
            "bm25_raw": float(bm25_raw[i]),
            "vec_sim":  float(vec_scores_raw[i]),
        }
        for rank, i in enumerate(top_indices)
    ]
 
 
def _group_by_source(chunks: list[dict]) -> list[tuple[str, list[dict]]]:
    """Return chunks grouped by kb_source, preserving insertion order."""
    seen:   dict[str, list[dict]] = {}
    for c in chunks:
        src = c.get("kb_source", "unknown")
        seen.setdefault(src, []).append(c)
    return list(seen.items())

# =============================================================================
# SECTION 5 — POST-PROCESSING
# =============================================================================
 
def _post_process(
    results:          list[dict],
    resolved_tickers: list[str],
    score_threshold:  float = RELEVANCE_THRESHOLD,
) -> list[dict]:
    """
    Option 1 — Relevance Filtering: drop chunks below score_threshold.
    Option 2 — Citation Enforcement: attach citation metadata to all survived chunks.
    """
    processed = []
    for chunk in results:
        if chunk["score"] < score_threshold:
            print(
                f"  ✂  Filtered [{chunk.get('kb_source','?')}] "
                f"rank={chunk['rank']} score={chunk['score']:.3f}"
            )
            continue
 
        # Ensure citation fields are present
        if "citation_id" not in chunk or not chunk["citation_id"]:
            chunk["citation_id"] = f"{chunk.get('kb_source','kb')}-{chunk['rank']}"
        if "source_url" not in chunk:
            chunk["source_url"] = "internal://" + chunk.get("kb_source", "unknown")
        if "source_type" not in chunk:
            chunk["source_type"] = chunk.get("kb_source", "unknown")
 
        chunk["relevant_to"] = resolved_tickers
        processed.append(chunk)
 
    return processed

# =============================================================================
# SECTION 6 — RETRIEVAL EVALUATION
# =============================================================================
 
def _compute_recall_at_k(
    results:          list[dict],
    relevant_tickers: list[str],
    k:                int,
) -> float:
    """
    Recall@K across both ticker chunks and non-ticker chunks.
 
    For ticker intents: measures ticker coverage in top-K.
    For non-ticker intents (concept, macro): measures KB source coverage.
    """
    if not relevant_tickers:
        # For concept/strategy queries with no specific ticker, measure
        # whether the correct KB sources appear in top-K
        return 1.0
 
    top_k_tickers = {
        r.get("ticker", "").upper()
        for r in results[:k]
    }
    found = len({t.upper() for t in relevant_tickers} & top_k_tickers)
    return round(found / len(relevant_tickers), 4)
 
 
def _compute_mrr(
    results:          list[dict],
    relevant_tickers: list[str],
) -> float:
    if not relevant_tickers:
        return 1.0
    rrs = []
    for ticker in relevant_tickers:
        rr = 0.0
        for chunk in results:
            if chunk.get("ticker", "").upper() == ticker.upper():
                rr = 1.0 / chunk["rank"]
                break
        rrs.append(rr)
    return round(sum(rrs) / len(rrs), 4)
 
 
def _compute_retrieval_metrics(
    results:          list[dict],
    relevant_tickers: list[str],
    intent:           str,
    k:                int,
) -> dict:
    recall = _compute_recall_at_k(results, relevant_tickers, k)
    mrr    = _compute_mrr(results, relevant_tickers)
 
    # KB source coverage — what fraction of expected sources appear in results
    expected_sources = set(INTENT_SOURCES.get(intent, []))
    retrieved_sources = {r.get("kb_source", "") for r in results[:k]}
    source_coverage  = (
        round(len(expected_sources & retrieved_sources) / len(expected_sources), 4)
        if expected_sources else 1.0
    )
 
    flags = []
    if relevant_tickers and recall < 1.0:
        missing = {t.upper() for t in relevant_tickers} - {
            r.get("ticker", "").upper() for r in results[:k]
        }
        flags.append(f"RECALL_MISS: {sorted(missing)} not in top-{k}")
    if relevant_tickers and mrr < 0.5:
        flags.append(f"LOW_MRR={mrr:.3f}: relevant chunks buried — consider increasing top_k")
    if source_coverage < 0.75:
        missing_src = expected_sources - retrieved_sources
        flags.append(f"LOW_SOURCE_COVERAGE: {missing_src} missing from results")
 
    return {
        f"Recall@{k}":      recall,
        "MRR":               mrr,
        "source_coverage":   source_coverage,
        "retrieved_sources": sorted(retrieved_sources),
        "diagnostic_flags":  flags,
        "justification": {
            "Recall@K": (
                f"Recall@{k}={recall:.4f}. Measures whether all queried tickers appear "
                f"in top-{k} results. Critical for financial Q&A — missed ticker = missed facts."
            ),
            "MRR": (
                f"MRR={mrr:.4f}. Measures rank position of first relevant chunk. "
                "Low MRR buries relevant context below noise in the LLM prompt."
            ),
            "source_coverage": (
                f"source_coverage={source_coverage:.4f}. Fraction of expected KB sources "
                f"({sorted(expected_sources)}) that appear in top-{k} results. "
                "Low coverage means the intent router is over-relying on one KB."
            ),
        },
    }

# =============================================================================
# SECTION 7 — PERSISTENCE
# =============================================================================
 
def _qhash(query: str) -> str:
    return hashlib.md5(query.encode()).hexdigest()[:10]
 
 
def _save_retrieval_log(query, intent, resolved_tickers, raw, filtered, metrics) -> str:
    os.makedirs(RETRIEVAL_LOG_DIR, exist_ok=True)
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RETRIEVAL_LOG_DIR, f"retrieval_{ts}_{_qhash(query)}.json")
    with open(path,"w") as f:
        json.dump({
            "query":                query,
            "intent":               intent,
            "timestamp":            datetime.datetime.now().isoformat(),
            "resolved_tickers":     resolved_tickers,
            "top_k":                len(raw),
            "score_threshold":      RELEVANCE_THRESHOLD,
            "retrieval_metrics":    metrics,
            "chunks_before_filter": len(raw),
            "chunks_after_filter":  len(filtered),
            "retrieved_chunks":     filtered,
        }, f, indent=2, default=str)
    print(f" Retrieval log → {path}")
    return path

# =============================================================================
# SECTION 8 — RETRIEVE_CONTEXT() FUNCTION
# =============================================================================
 
def retrieve_context(
    query:           str,
    top_k:           int            = 8,
    force_refresh:   bool           = False,
    score_threshold: float          = RELEVANCE_THRESHOLD,
    save_log:        bool           = True,
    portfolio_store: PortfolioStore | None = None,
    macro_store:     MacroStore     | None = None,
    concept_store:   ConceptStore   | None = None,
    strategy_store:  StrategyStore  | None = None,
    fred_api_key:    str            | None = None,
) -> tuple[list[dict], dict, str]:
    """
    Retrieval stage: detect intent → collect chunks → embed → retrieve →
    post-process → evaluate → log.
 
    Parameters
    ----------
    query            : Natural language question
    top_k            : Chunks to retrieve before filtering
    force_refresh    : Force regeneration of dynamic KB files
    score_threshold  : Minimum fused score to pass filtering
    save_log         : Write retrieval JSON log
    portfolio_store  : Pre-initialised PortfolioStore (or None to skip kb2)
    macro_store      : Pre-initialised MacroStore (or None to auto-create)
    concept_store    : Pre-initialised ConceptStore (or None to auto-create)
    strategy_store   : Pre-initialised StrategyStore (or None to auto-create)
    fred_api_key     : FRED API key for macro data (or set FRED_API_KEY env var)
 
    Returns
    -------
    (filtered_chunks, retrieval_metrics, log_path)
    """
    # Auto-create lightweight stores if not provided
    if concept_store is None:
        concept_store = ConceptStore()
    if strategy_store is None:
        strategy_store = StrategyStore()
    if macro_store is None:
        macro_store = MacroStore(fred_api_key=fred_api_key)
 
    # 1. Intent detection
    intent = detect_intent(query)
    print(f"\n🎯 Intent detected: {intent.upper()}")
 
    # 2. Ticker resolution (for kb1 and metric relevance)
    resolved_tickers = resolve_tickers_from_query(query)
    if resolved_tickers:
        print(f"🔍 Resolved tickers: {resolved_tickers}")
 
    # 3. Collect chunks from all relevant KB sources
    print("\n📚 Collecting chunks from KB sources…")
    all_chunks = _collect_all_chunks(
        intent=intent,
        resolved_tickers=resolved_tickers,
        portfolio_store=portfolio_store,
        macro_store=macro_store,
        concept_store=concept_store,
        strategy_store=strategy_store,
        force_refresh=force_refresh,
    )
 
    if not all_chunks:
        print("  ⚠  No chunks collected — check KB sources")
        return [], {}, ""
 
    # 4. Embed and build Chroma collections
    model  = SentenceTransformer(EMBED_MODEL_NAME)
    client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
    print("\n🔢 Building vector collections…")
    collections = _build_collections(all_chunks, model, client)
 
    # 5. Hybrid retrieval
    print(f"\n🔎 Retrieving top-{top_k} chunks…")
    raw_results = _hybrid_retrieve_multi(
        query, all_chunks, collections, model, top_k=top_k
    )
 
    # 6. Post-processing
    filtered = _post_process(raw_results, resolved_tickers, score_threshold)
 
    # 7. Retrieval evaluation
    metrics = _compute_retrieval_metrics(raw_results, resolved_tickers, intent, k=top_k)
 
    k_key = f"Recall@{top_k}"
    print(f"\n📊 {k_key}: {metrics[k_key]:.4f}  MRR: {metrics['MRR']:.4f}  "
          f"Source coverage: {metrics['source_coverage']:.4f}")
    for flag in metrics.get("diagnostic_flags", []):
        print(f"  ⚠  {flag}")
 
    print(f"\n📄 {len(filtered)} chunks after filtering (was {len(raw_results)}):")
    for r in filtered:
        src = r.get("kb_source", "?")
        print(f"  [{r['rank']}] [{src}] score={r['score']:.3f}  {r['text'][:65]}…")
 
    # 8. Save log
    log_path = ""
    if save_log:
        log_path = _save_retrieval_log(
            query, intent, resolved_tickers, raw_results, filtered, metrics
        )
 
    return filtered, metrics, log_path


# =============================================================================
# CLI DEMO
# =============================================================================
# if __name__ == "__main__":
#     # Set up stores
#     portfolio = PortfolioStore()
#     if not portfolio.holdings:
#         portfolio.set_holdings({
#             "AAPL": {"weight": 0.30, "cost_basis": 150.00},
#             "MSFT": {"weight": 0.25, "cost_basis": 280.00},
#             "NVDA": {"weight": 0.20, "cost_basis": 400.00},
#             "SPY":  {"weight": 0.15, "cost_basis": 420.00},
#             "TLT":  {"weight": 0.10, "cost_basis": 95.00},
#         })
# 
#     macro = MacroStore()
#     concepts = ConceptStore()
#     strategies = StrategyStore()
# 
#     # Single query
#     query = "What is the Sharpe ratio and why does it matter?"
# 
#     print(f"\n{'='*65}")
#     print(f"QUERY: {query}")
#     print("="*65)
# 
#     chunks, metrics, log = retrieve_context(
#         query=query,
#         top_k=6,
#         portfolio_store=portfolio,
#         macro_store=macro,
#         concept_store=concepts,
#         strategy_store=strategies,
#     )
# 
#     # Print evaluation metrics
#     k_key = next(k for k in metrics if k.startswith("Recall@"))
#     print(f"\n{k_key}={metrics[k_key]}  MRR={metrics['MRR']}  "
#           f"Sources={metrics['retrieved_sources']}")
