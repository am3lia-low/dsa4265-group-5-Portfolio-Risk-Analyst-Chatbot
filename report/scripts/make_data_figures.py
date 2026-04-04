"""Build figures for the report from data/processed CSVs."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
FIG = ROOT / "report" / "figures"


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    simple = pd.read_csv(PROC / "simple_returns.csv", index_col=0, parse_dates=True)
    adj = pd.read_csv(PROC / "adj_close_prices.csv", index_col=0, parse_dates=True)

    # Figure: correlation heatmap (matches preprocessing notebook, cleaned returns)
    corr = simple.corr()
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
    ax.set_title("Correlation matrix — cleaned simple daily returns")
    plt.tight_layout()
    fig.savefig(FIG / "data_corr_heatmap.png", dpi=150)
    plt.close(fig)

    # Figure: drawdowns from adjusted closes (notebook-style diagnostic)
    dd = adj.div(adj.cummax()).sub(1.0)
    fig, ax = plt.subplots(figsize=(11, 5))
    for col in dd.columns:
        ax.plot(dd.index, dd[col], label=col, linewidth=1.0, alpha=0.88)
    ax.set_title("Drawdown from running high (adjusted close)")
    ax.set_ylabel("Drawdown")
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.45)
    ax.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    fig.savefig(FIG / "data_drawdowns.png", dpi=150)
    plt.close(fig)

    print("Wrote:", FIG / "data_corr_heatmap.png")
    print("Wrote:", FIG / "data_drawdowns.png")


if __name__ == "__main__":
    main()
