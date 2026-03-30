"""Pull adjusted close prices once; reuse for all quant / ML steps."""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd
import yfinance as yf


def _normalise_tickers(tickers: str | Sequence[str]) -> list[str]:
    if isinstance(tickers, str):
        tickers = [tickers]
    out: list[str] = []
    seen: set[str] = set()
    for t in tickers:
        t = str(t).strip().upper()
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _extract_adj_close(df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if df.empty:
        raise ValueError("yfinance returned no rows for the given range / tickers.")

    if isinstance(df.columns, pd.MultiIndex):
        if "Adj Close" in df.columns.get_level_values(0):
            out = df["Adj Close"].copy()
        else:
            out = df.xs("Adj Close", axis=1, level=0, drop_level=True)
            if isinstance(out, pd.Series):
                out = out.to_frame(name=tickers[0])
    else:
        out = df.rename(columns={"Adj Close": tickers[0]})

    out = out.sort_index()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out[tickers]


def fetch_price_data(
    tickers: str | Iterable[str],
    start: str | None = None,
    end: str | None = None,
    *,
    include_spy: bool = True,
) -> pd.DataFrame:
    """
    Download **Adj Close** for the given tickers (splits/dividends adjusted).

    If ``include_spy`` is True and ``SPY`` is not already requested, SPY is added
    so workflows have a cheap equity benchmark column without a second download later.

    Parameters
    ----------
    tickers : str or iterable of str
        Yahoo symbols, e.g. ``\"AAPL\"`` or ``[\"AAPL\", \"MSFT\"]``.
    start, end : str or None
        Passed to yfinance ``download``. Use ``None`` for yfinance defaults
        (``end=None`` means “up to today” when ``start`` is set).
    include_spy : bool
        If True, ensure ``SPY`` is in the download list.

    Returns
    -------
    pd.DataFrame
        Index: dates (naive). Columns: tickers (Adj Close).
    """
    tlist = _normalise_tickers(tickers)  # type: ignore[arg-type]
    if not tlist:
        raise ValueError("tickers is empty after cleaning.")

    if include_spy and "SPY" not in tlist:
        tlist = list(tlist) + ["SPY"]

    df = yf.download(
        list(tlist),
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    return _extract_adj_close(df, tlist)
