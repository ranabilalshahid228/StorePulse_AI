"""Evaluation plots rendered headlessly with matplotlib (Agg)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def actual_vs_forecast(val: pd.DataFrame, out: Path) -> Path | None:
    if val.empty:
        return None
    daily = val.groupby("ds")[["y", "yhat"]].sum()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(daily.index, daily["y"], label="actual", linewidth=1.6)
    ax.plot(daily.index, daily["yhat"], label="forecast", linewidth=1.6, linestyle="--")
    ax.set_title("Validation window - aggregated actual vs forecast")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def learning_curve(logs: list[dict[str, Any]], out: Path) -> Path | None:
    if len(logs) < 2:
        return None
    df = pd.DataFrame(logs).set_index("epoch")
    cols = [
        c for c in df.columns if c.startswith(("val_", "train_")) and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not cols:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    for c in cols[:4]:
        ax.plot(df.index, df[c], label=c)
    ax.set_xlabel("epoch / boosting round")
    ax.set_title("Learning curve")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def feature_importance(importance: dict[str, float], out: Path, top: int = 20) -> Path:
    items = sorted(importance.items(), key=lambda kv: kv[1])[-top:]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(items))))
    ax.barh([k for k, _ in items], [v for _, v in items])
    ax.set_title("Feature importance (gain)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
