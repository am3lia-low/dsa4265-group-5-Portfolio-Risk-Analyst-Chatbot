"""
RAG_utils.py
============
Unified RAG pipeline for the risk portfolio chatbot.
 
Knowledge bases
---------------
  kb0  ticker resolver          — query → ticker symbols
  kb1  ticker profiles          — yfinance fundamentals + price history
  kb2  macro store              — FRED rates, VIX, sector ETFs, regime classification
  kb3  concept definitions      — static curated financial concepts
  kb4  strategy frameworks      — rebalancing rules, allocation frameworks
 
Intent routing
--------------
  full_analysis    → kb1 (ticker) + kb4 (strategies)
  concept_explanation → kb3 (concepts) + kb4 (strategies)
  trend_prediction → kb1 (ticker) + kb2 (macro)
  "none"           → all KB sources
 
Pipeline
--------
  1. Source selection per intent
  2. KB refresh (staleness check for dynamic sources)
  3. Unified chunk pool from all relevant sources
  4. Hybrid BM25 + vector retrieval
  5. Post-processing: relevance filter + citation enforcement
  6. Retrieval evaluation: Recall@K + MRR
  7. Persist logs (retrieval_log)
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
 
from pathlib import Path
from dotenv import load_dotenv
warnings.filterwarnings("ignore")

# ── Retrieve FRED_API_KEY ──────────────────────────────────────────────────────────
# Automatically find the .env in the main folder
# update: no need to fetch freddie. it has been fetched in kb2_macro_regime.py

# BASE_DIR = Path(__file__).resolve().parent.parent.parent # Go up from RAG/ to main folder
# ENV_PATH = BASE_DIR / ".env"

# if ENV_PATH.exists():
#     load_dotenv(dotenv_path=ENV_PATH)
#     print(f"fetching freddie🐊 from env:{ENV_PATH}")
# else:
#     print("⚠ .env file not found for RAG_utils")

# Access env variable:
FRED_API_KEY = os.getenv("FRED_API_KEY")
 
# ── Local KB modules ──────────────────────────────────────────────────────────
# from . import kb0_ticker_resolver
# from . import kb1_generate_tickers
# from . import kb2_macro_regime
# from . import kb3_concepts
# from . import kb4_strategies

# for _mod in [kb0_ticker_resolver, kb1_generate_tickers,
#              kb2_macro_regime, kb3_concepts, kb4_strategies]:
#     importlib.reload(_mod)
 
from .kb0_ticker_resolver import resolve_tickers_from_query
from .kb1_generate_tickers import (
    generate_tickers,
    convert_tickers_into_txt,
    build_ticker_meta as _build_ticker_meta,
    OUTPUT_DIR_HTML, OUTPUT_TXT
)
from .kb2_macro_regime import MacroStore
from .kb3_concepts  import ConceptStore
from .kb4_strategies import StrategyStore

# ── Config ────────────────────────────────────────────────────────────────────
VECTOR_DB_DIR       = "agent_tools/rag_tools/vector_db"
EMBED_MODEL_NAME    = "sentence-transformers/all-MiniLM-L6-v2"
KB_STALENESS_HOURS  = 24
RELEVANCE_THRESHOLD = 0.10    # lower than before — static KB chunks score differently
RETRIEVAL_LOG_DIR   = "agent_tools/rag_tools/retrieval_log"
 
# ── Intent → KB source mapping ────────────────────────────────────────────────
# Each intent specifies which Chroma collections to query
INTENT_SOURCES: dict[str, list[str]] = {
    "full_analysis":     ["tickers", "strategies"],
    "concept_explanation": ["concepts", "strategies"],
    "trend_prediction":  ["tickers", "macro"],
    "fallback":          ["tickers", "macro", "concepts", "strategies"],
}
 
# Collection name → display label
COLLECTION_LABELS = {
    "tickers":    "Ticker Profiles (yfinance)",
    "macro":      "Macro & Regime (FRED + yfinance)",
    "concepts":   "Concept Definitions",
    "strategies": "Strategy Frameworks",
}

# =============================================================================
# SECTION 1 — KB SOURCE MANAGEMENT
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
    macro_store:      MacroStore     | None = None,
    concept_store:    ConceptStore   | None = None,
    strategy_store:   StrategyStore  | None = None,
    force_refresh:    bool           = False,
    silent:           bool           = False,
) -> tuple[list[dict], list[str]]:
    """
    Gather chunks from all KB sources relevant to the detected intent.

    Returns a flat list of chunk dicts, each with:
      kb_source, text, citation_id, source_url, source_type, intent_tags
    And a list of used stores
    """
    sources   = INTENT_SOURCES.get(intent, INTENT_SOURCES["fallback"])
    all_chunks: list[dict] = []
    used_stores: list[str] = []

    # ── kb1: Ticker profiles ───────────────────────────────────────────────
    if "tickers" in sources and resolved_tickers:
        if not silent:
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
        if not silent:
            print(f"    → {len(filtered)} ticker chunks")

    # ── kb2: Macro ─────────────────────────────────────────────────────────
    if "macro" in sources and macro_store is not None:
        used_stores.append("MacroStore")
        if not silent:
            print("  📥 kb2: loading macro chunks + writing TXT")
        if force_refresh or macro_store._is_stale():
            macro_store.refresh(force=force_refresh)
        chunks = macro_store.generate_chunks()
        macro_store.export_txt()
        all_chunks.extend(chunks)
        if not silent:
            print(f"    → {len(chunks)} macro chunks")

    # ── kb3: Concepts ──────────────────────────────────────────────────────
    if "concepts" in sources and concept_store is not None:
        used_stores.append("ConceptStore")
        if not silent:
            print("  📥 kb3: loading concept chunks + writing TXT")
        chunks = concept_store.generate_chunks()
        concept_store.export_txt()
        all_chunks.extend(chunks)
        if not silent:
            print(f"    → {len(chunks)} concept chunks")

    # ── kb4: Strategies ────────────────────────────────────────────────────
    if "strategies" in sources and strategy_store is not None:
        used_stores.append("StrategyStore")
        if not silent:
            print("  📥 kb4: loading strategy chunks + writing TXT")
        chunks = strategy_store.generate_chunks()
        strategy_store.export_txt()
        all_chunks.extend(chunks)
        if not silent:
            print(f"    → {len(chunks)} strategy chunks")

    if not silent:
        print(f"  Total chunks in pool: {len(all_chunks)}")
    return all_chunks, used_stores

# =============================================================================
# SECTION 2 — CHROMA: MULTI-COLLECTION MANAGEMENT
# =============================================================================
 
def _get_or_create_collection(
    client:     chromadb.PersistentClient,
    name:       str,
    documents:  list[dict],
    model:      SentenceTransformer,
) -> chromadb.Collection:
    collection = client.get_or_create_collection(name=name)
    if not documents:
        return collection

    texts      = [d["text"] for d in documents]
    metadatas  = [
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

    try:
        collection.upsert(
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids,
        )
    except Exception as e:
        print(f"  ⚠ Upsert failed for '{name}': {e}")
        print(f"  🔄 Deleting and recreating collection '{name}'...")
        client.delete_collection(name)
        collection = client.get_or_create_collection(name=name)
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
# SECTION 3 — HYBRID RETRIEVAL (MULTI-COLLECTION)
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
# SECTION 4 — POST-PROCESSING
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
# SECTION 5 — RETRIEVAL EVALUATION
# =============================================================================
 
def _is_relevant(
    intent:           str,
    chunk:            dict,
    relevant_tickers: list[str],
    expected_sources: set[str],
) -> bool:
    """
    Tiered relevance:
      - Ticker chunks: must match a resolved ticker (strict)
      - Macro chunks:  always relevant if macro is an expected source
      - Concept/strategy chunks: only relevant if no tickers were resolved
        (i.e. pure concept/strategy query)
    """
    source = chunk.get("kb_source", "")
    
    # 1. Must belong to expected sources for this intent
    if source not in expected_sources:
        return False

    # 2. Intent-specific relevance rules
    # ─────────────────────────────────────

    # FULL ANALYSIS
    if intent == "full_analysis":
        if source == "tickers":
            return (
                bool(relevant_tickers) and
                chunk.get("ticker", "").upper() in {t.upper() for t in relevant_tickers}
            )
        if source == "strategies":
            return True   # supporting info
        return False      # exclude others

    # CONCEPT EXPLANATION
    if intent == "concept_explanation":
        if source == "concepts":
            return True
        if source == "strategies":
            return True   # supporting examples
        return False

    # TREND PREDICTION
    if intent == "trend_prediction":
        if source == "macro":
            return True
        if source == "tickers":
            return (
                bool(relevant_tickers) and
                chunk.get("ticker", "").upper() in {t.upper() for t in relevant_tickers}
            )
        return False

    # FALLBACK (broad but still controlled)
    if intent == "fallback":
        if source == "tickers":
            return (
                not relevant_tickers or
                chunk.get("ticker", "").upper() in {t.upper() for t in relevant_tickers}
            )
        return True

    return False

def _compute_recall_at_k( #fixed recall to actual formula
    intent:           str,
    results:          list[dict],
    all_chunks:       list[dict],
    relevant_tickers: list[str],
    expected_sources: set[str],
    k:                int,
) -> float:
    """
    No more ticker and non-ticker intents
    Recall@k measures relevant in top-k / total relevant in full point
    Always return a value in [0,1] regardless of intent
    """
    if not all_chunks:
        return 0.0

    total_relevant = sum(
        1 for c in all_chunks
        if _is_relevant(intent, c, relevant_tickers, expected_sources)
    ) 
    if total_relevant == 0:
        return 0
 
    relevant_in_top_k = sum(
        1 for r in results[:k]
        if _is_relevant(intent, r, relevant_tickers, expected_sources)
    )
    return round (relevant_in_top_k / total_relevant, 4)

def _compute_mrr(
    intent:           str,
    results:          list[dict],
    relevant_tickers: list[str],
    expected_sources: set[str],
) -> float:
    """
    No more ticker and non-ticker intents
    MRR measures the reciprocal rank of the FIRST relevant chunk
    """
    if not results:
        return 0.0

    for chunk in results:
        if _is_relevant(intent, chunk, relevant_tickers, expected_sources):
            return round(1.0 / chunk["rank"], 4)

    return 0.0
 
def _compute_retrieval_metrics(
    results:          list[dict],
    all_chunks:       list[dict],
    relevant_tickers: list[str],
    intent:           str,
    k:                int,
) -> dict:
    expected_sources  = set(INTENT_SOURCES.get(intent, INTENT_SOURCES["fallback"]))
    retrieved_sources = sorted({r.get("kb_source", "") for r in results[:k]})

    recall = _compute_recall_at_k(intent, results, all_chunks, relevant_tickers, expected_sources, k)
    mrr    = _compute_mrr(intent, results, relevant_tickers, expected_sources)

    flags = []
    if recall < 0.5:
        flags.append(f"LOW_RECALL@{k}={recall:.3f}: too many relevant chunks missed in top-{k}")
    if mrr < 0.5:
        flags.append(f"LOW_MRR={mrr:.3f}: relevant chunks ranked poorly — consider increasing top_k")

    return {
        f"Recall@{k}":       recall,
        "MRR":               mrr,
        "retrieved_sources": retrieved_sources,
        "diagnostic_flags":  flags,
        "justification": {
            "Recall@K": (
                f"Recall@{k}={recall:.4f}. Relevant chunks in top-{k} divided by total "
                f"relevant chunks in the full pool. Expected sources: {sorted(expected_sources)}."
            ),
            "MRR": (
                f"MRR={mrr:.4f}. Average reciprocal rank across all relevant chunks in results. "
                "Low MRR means relevant chunks are consistently ranked low."
            ),
        },
    }

# =============================================================================
# SECTION 6 — PERSISTENCE
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
# SECTION 7 — RETRIEVE RESULTS IN PYDANTIC MODEL
# =============================================================================

from pydantic import BaseModel, Field
from typing import Optional

# ── Single retrieved chunk ────────────────────────────────────────────────────
class RetrievedChunk(BaseModel):
    # Core content
    text:        str
    kb_source:   str
    
    # Citation fields (enforced in _post_process)
    citation_id: str  = ""
    source_url:  str  = ""
    source_type: str  = ""
    
    # Ticker-specific (empty string for macro/concept chunks)
    ticker:      str  = ""
    relevant_to: list[str] = Field(default_factory=list)
    
    # Retrieval scores (added in _hybrid_retrieve_multi)
    rank:     int   = 0
    score:    float = 0.0
    bm25_raw: float = 0.0
    vec_sim:  float = 0.0


# ── Retrieval metrics ─────────────────────────────────────────────────────────
class RetrievalMetrics(BaseModel):
    intent:           str
    recall_at_k:      float
    mrr:              float
    retrieved_sources: list[str]   = Field(default_factory=list)
    diagnostic_flags:  list[str]   = Field(default_factory=list)
    justification:     dict[str, str] = Field(default_factory=dict)


# ── Full retrieval result (wraps retrieve_context return value) ────────────────
class RetrievalResult(BaseModel):
    query:             str
    intent:            str
    resolved_tickers:  list[str]          = Field(default_factory=list)
    chunks:            list[RetrievedChunk]
    metrics:           RetrievalMetrics
    log_path:          str                = ""

# =============================================================================
# SECTION 8 — RETRIEVE_CONTEXT() FUNCTION
# =============================================================================
 
def retrieve_context(
    intent:          str,
    query:           str,
    top_k:           int            = 8,
    force_refresh:   bool           = False,
    score_threshold: float          = RELEVANCE_THRESHOLD,
    save_log:        bool           = True,
    macro_store:     MacroStore     | None = None,
    concept_store:   ConceptStore   | None = None,
    strategy_store:  StrategyStore  | None = None,
    fred_api_key:    str            | None = None,
    silent:          bool           = False,
) -> RetrievalResult:
    """
    Retrieval stage: look at which intent → collect chunks → embed → retrieve →
    post-process → evaluate → log.

    Parameters
    ----------
    intent           : Detect intent string (full_analysis, concept_explanation, trend_prediction)
    query            : Natural language question
    top_k            : Chunks to retrieve before filtering
    force_refresh    : Force regeneration of dynamic KB files
    score_threshold  : Minimum fused score to pass filtering
    save_log         : Write retrieval JSON log
    macro_store      : Pre-initialised MacroStore (or None to auto-create)
    concept_store    : Pre-initialised ConceptStore (or None to auto-create)
    strategy_store   : Pre-initialised StrategyStore (or None to auto-create)
    fred_api_key     : FRED API key for macro data (or set FRED_API_KEY env var)
    silent           : Whether to suppress print statements and only return MRR/store info

    Returns
    -------
    (filtered_chunks, retrieval_metrics, log_path)
    """
    resolved_tickers = []

    # 1. Intent detection (intent needed from LLM)
    if intent == "full_analysis":
        resolved_tickers = resolve_tickers_from_query(query)
        strategy_store   = strategy_store or StrategyStore()

    elif intent == "concept_explanation":
        concept_store    = concept_store or ConceptStore()
        strategy_store   = strategy_store or StrategyStore()

    elif intent == "trend_prediction":
        resolved_tickers = resolve_tickers_from_query(query)
        macro_store      = macro_store or MacroStore(fred_api_key=fred_api_key)

    else:
        resolved_tickers = resolve_tickers_from_query(query)
        macro_store      = macro_store or MacroStore(fred_api_key=fred_api_key)
        concept_store    = concept_store or ConceptStore()
        strategy_store   = strategy_store or StrategyStore()


    # 2. Collect chunks from all relevant KB sources
    if not silent:
        print("\n📚 Collecting chunks from KB sources…")
    all_chunks, used_stores = _collect_all_chunks(
        intent = intent,
        resolved_tickers=resolved_tickers,
        macro_store=macro_store,
        concept_store=concept_store,
        strategy_store=strategy_store,
        force_refresh=force_refresh,
        silent=silent,
    )

    if not all_chunks:
        if not silent:
            print("  ⚠  No chunks collected — check KB sources")
        return RetrievalResult(
            query            = query,
            intent           = intent,
            resolved_tickers = resolved_tickers,
            chunks           = [],
            metrics          = RetrievalMetrics(
                recall_at_k       = 0.0,
                mrr               = 0.0,
            ),
            log_path         = "",
        )

    # 3. Embed and build Chroma collections
    model  = SentenceTransformer(EMBED_MODEL_NAME)
    client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
    if not silent:
        print("\n🔢 Building vector collections…")
    collections = _build_collections(all_chunks, model, client)

    # 4. Hybrid retrieval
    if not silent:
        print(f"\n🔎 Retrieving top-{top_k} chunks…")
    raw_results = _hybrid_retrieve_multi(
        query, all_chunks, collections, model, top_k=top_k
    )

    # 5. Post-processing
    filtered = _post_process(raw_results, resolved_tickers, score_threshold)

    # 6. Retrieval evaluation
    metrics = _compute_retrieval_metrics(raw_results, all_chunks, resolved_tickers, intent, k=top_k)

    k_key = f"Recall@{top_k}"
    if not silent:
        print(f"\n📊 {k_key}: {metrics[k_key]:.4f}  MRR: {metrics['MRR']:.4f} ")
        for flag in metrics.get("diagnostic_flags", []):
            print(f"  ⚠  {flag}")

        print(f"\n📄 {len(filtered)} chunks after filtering (was {len(raw_results)}):")
        for r in filtered:
            src = r.get("kb_source", "?")
            print(f"  [{r['rank']}] [{src}] score={r['score']:.3f}  {r['text'][:65]}…")
    else:
        # Print only MRR and used stores when silent mode is on
        print(f"MRR: {metrics['MRR']:.4f}, Used Stores: {', '.join(used_stores) if used_stores else 'None'}")

    # 7. Save log
    log_path = ""
    if save_log:
        log_path = _save_retrieval_log(
            query, intent, resolved_tickers, raw_results, filtered, metrics
        )

    # 8. Parse into Pydantic models
    typed_chunks = [RetrievedChunk(**c) for c in filtered]

    typed_metrics = RetrievalMetrics(
        intent            = intent,
        recall_at_k       = metrics[f"Recall@{top_k}"],
        mrr               = metrics["MRR"],
        retrieved_sources = metrics["retrieved_sources"],
        diagnostic_flags  = metrics["diagnostic_flags"],
        justification     = metrics["justification"],
    )

    return RetrievalResult(
        query            = query,
        intent           = intent,
        resolved_tickers = resolved_tickers,
        chunks           = typed_chunks,
        metrics          = typed_metrics,
        log_path         = log_path,
    )

# =============================================================================
# CLI DEMO
# =============================================================================
import json
if __name__ == "__main__":
    # Set up stores
    macro = MacroStore(fred_api_key="06a5067f80033b7cf40e36c48224f59a")
    concepts = ConceptStore()
    strategies = StrategyStore()

    # Single query
    print(f"\n{'='*65}")
    print("QUERY")
    print("="*65)

    result = retrieve_context(
        intent         = "trend_prediction",
        query          = "How is AAPL so far, and is it risky?",
        top_k          = 10,
        force_refresh  = False,
        score_threshold= 0.10,
        save_log       = True,
        macro_store    = macro,
        concept_store  = concepts,
        strategy_store = strategies,
    )
    
    # print the full result
    print(json.dumps(result.dict(), indent=2))