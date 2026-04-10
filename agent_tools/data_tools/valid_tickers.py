"""Cheap checks that symbols exist and return recent history."""

from __future__ import annotations

from typing import Iterable, Sequence

import yfinance as yf


def _clean_symbols(tickers: str | Sequence[str]) -> list[str]:
    if isinstance(tickers, str):
        tickers = [tickers]
    return [str(t).strip().upper() for t in tickers if str(t).strip()]


def valid_tickers(tickers: str | Sequence[str]) -> tuple[bool, list[str]]:
    """
    Return whether every ticker looks tradeable on Yahoo and has recent bars.

    Fetches all tickers in a single batched yfinance request.

    Returns
    -------
    ok : bool
        True if all symbols passed.
    bad : list[str]
        Symbols that failed (empty if ``ok``).
    """
    syms = _clean_symbols(tickers)
    if not syms:
        return False, ["<empty>"]

    data = yf.download(syms, period="5d", auto_adjust=True, progress=False, threads=True)

    if len(syms) == 1:
        close = data["Close"].dropna()
        bad = [syms[0]] if close.empty else []
    else:
        close = data["Close"]
        bad = [
            sym for sym in syms
            if sym not in close.columns or close[sym].dropna().empty
        ]

    return (len(bad) == 0, bad)
