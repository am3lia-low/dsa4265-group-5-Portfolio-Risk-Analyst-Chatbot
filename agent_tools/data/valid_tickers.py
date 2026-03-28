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

    bad: list[str] = []
    for sym in syms:
        t = yf.Ticker(sym)
        hist = t.history(period="5d")
        if hist.empty:
            bad.append(sym)
            continue
        # extra guard: yfinance sometimes returns a row of NaNs
        if hist["Close"].dropna().empty:
            bad.append(sym)

    return (len(bad) == 0, bad)
