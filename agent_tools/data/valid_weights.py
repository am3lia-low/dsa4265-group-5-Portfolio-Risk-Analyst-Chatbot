"""Portfolio weight sanity check."""

from __future__ import annotations

from typing import Iterable, Sequence, SupportsFloat

import numpy as np


def valid_weights(
    weights: Sequence[SupportsFloat] | Iterable[SupportsFloat],
    *,
    tol: float = 1e-6,
) -> tuple[bool, str | None]:
    """
    Check that weights sum to 1 (fractions) or 100 (percentages).

    Parameters
    ----------
    weights : sequence of numbers
        Must be finite; negative values fail.
    tol : float
        Absolute tolerance on the sum.

    Returns
    -------
    ok : bool
    message : str or None
        Error text if not ok.
    """
    w = np.asarray(list(weights), dtype=float)
    if w.size == 0:
        return False, "weights is empty."
    if not np.all(np.isfinite(w)):
        return False, "weights must be finite numbers."
    if np.any(w < 0):
        return False, "negative weights not allowed here."

    s = float(w.sum())
    # interpret as fractions in [0, 1] or percentages summing to ~100
    if abs(s - 1.0) <= tol:
        return True, None
    if abs(s - 100.0) <= tol * 100:
        return True, None

    return False, f"weights sum to {s:.6g}; expected ~1.0 (fractions) or ~100.0 (percent)."
