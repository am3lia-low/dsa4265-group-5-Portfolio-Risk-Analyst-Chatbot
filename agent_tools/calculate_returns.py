"""Turn price levels into return series (one place for quant + ML)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_returns(price_df: pd.DataFrame, method: str = "simple") -> pd.DataFrame:
    """
    Convert level prices to returns.

    Parameters
    ----------
    price_df : pd.DataFrame
        Columns = assets, index = dates (sorted ascending).
    method : str
        ``\"simple\"`` — pct change: P_t / P_{t-1} - 1
        ``\"log\"`` — log return: log P_t - log P_{t-1}

    Returns
    -------
    pd.DataFrame
        Same columns as ``price_df``; first row is dropped (no prior price).
    """
    m = str(method).strip().lower()
    if m in ("simple", "pct", "arithmetic"):
        return price_df.pct_change().dropna(how="any")
    if m in ("log", "continuous"):
        return np.log(price_df).diff().dropna(how="any")
    raise ValueError(f"Unknown method {method!r}; use 'simple' or 'log'.")
