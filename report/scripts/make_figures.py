"""Draw every results chart of the report from report/data/*.csv (vector PDF, IEEE column width).

cd backend
.venv\\Scripts\\python ..\\report\\scripts\\make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "report" / "data"
FIG = ROOT / "report" / "figures"
COL = 3.45  # IEEE single-column width in inches

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 7.5,
        "axes.titlesize": 8,
        "axes.labelsize": 7.5,
        "legend.fontsize": 6.5,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)
# Colour-blind-safe palette (Okabe-Ito).
C = {
    "seasonal_naive": "#999999",
    "arima": "#E69F00",
    "sarima": "#D55E00",
    "lightgbm": "#0072B2",
    "torch_mlp": "#56B4E9",
    "torch_gru": "#CC79A7",
    "ensemble": "#009E73",
}
LABEL = {
    "seasonal_naive": "Seasonal naive",
    "arima": "ARIMA(1,1,1)",
    "sarima": "SARIMA",
    "lightgbm": "LightGBM",
    "torch_mlp": "MLP",
    "torch_gru": "GRU",
    "ensemble": "Ensemble",
}
ORDER = ["seasonal_naive", "arima", "sarima", "lightgbm", "torch_mlp", "torch_gru", "ensemble"]


def comparison_table() -> pd.DataFrame:
    """One WAPE per (model, horizon) on the identical validation window, all 1,115 stores."""
    gpu = pd.read_csv(DATA / "model_comparison.csv")
    rows = []
    for _, r in gpu.iterrows():
        model = r["model"]
        if model == "lightgbm (local CPU rerun)":
            continue
        rows.append({"model": model, "horizon": int(r["horizon_days"]), "wape": r["wape"], "source": r["source"]})
    # ARIMA / SARIMA at 42 days were trained in the local app (same data, stores and validation window).
    local = pd.read_csv(DATA / "model_runs_local.csv")
    local = local[(local["dataset"] == "Rossmann full (all stores)") & (local["status"] == "succeeded")]
    for fam in ("arima", "sarima"):
        hit = local[(local["family"] == fam) & (local["horizon_days"] == 42)]
        if len(hit):
            rows.append({"model": fam, "horizon": 42, "wape": hit.iloc[0]["wape"], "source": "local app (CPU)"})
    out = pd.DataFrame(rows).drop_duplicates(["model", "horizon"])
    out.to_csv(DATA / "fig_accuracy_source.csv", index=False)
    return out


def fig_accuracy() -> None:
    df = comparison_table()
    fig, ax = plt.subplots(figsize=(COL, 2.1))
    width = 0.38
    for i, h in enumerate((30, 42)):
        sub = df[df["horizon"] == h].set_index("model").reindex(ORDER)
        x = [k + (i - 0.5) * width for k in range(len(ORDER))]
        bars = ax.bar(
            x,
            sub["wape"] * 100,
            width,
            label=f"{h}-day horizon",
            color=[C[m] for m in ORDER],
            alpha=0.55 if h == 30 else 1.0,
            edgecolor="black",
            linewidth=0.3,
            hatch="///" if h == 30 else None,
        )
        for b, v in zip(bars, sub["wape"]):
            if pd.notna(v):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    v * 100 + 0.4,
                    f"{v * 100:.1f}",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=5,
                )
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([LABEL[m] for m in ORDER], rotation=25, ha="right")
    ax.set_ylabel("Validation WAPE (%)")
    ax.set_ylim(0, 27)
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor="white", edgecolor="black", hatch="///", label="30-day horizon"),
            Patch(facecolor="#555555", edgecolor="black", label="42-day horizon"),
        ],
        loc="upper right",
        frameon=False,
    )
    fig.savefig(FIG / "fig_accuracy.pdf")
    plt.close(fig)


def fig_week_ahead() -> None:
    df = pd.read_csv(DATA / "residuals_by_week_ahead.csv")
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.9), sharey=True)
    learned = ["lightgbm", "torch_mlp", "torch_gru", "ensemble"]  # baselines (19-32 %) would flatten the scale
    for ax, h in zip(axes, (42, 30)):
        sub = df[df["horizon_days"] == h]
        for m in learned:
            s = sub[sub["model"] == m]
            if len(s) and not (m == "ensemble" and h == 30):  # h30 ensemble == GRU (weight 1.0)
                ax.plot(s["week"], s["wape"] * 100, marker="o", ms=2.5, lw=1, color=C[m], label=LABEL[m])
        ax.set_title(f"{h}-day model")
        ax.set_xlabel("Week ahead")
        ax.set_xticks(sorted(sub["week"].unique()))
    axes[0].set_ylabel("WAPE (%)")
    axes[0].set_ylim(5, 13)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.1))
    fig.savefig(FIG / "fig_week_ahead.pdf")
    plt.close(fig)


def fig_weekday() -> None:
    df = pd.read_csv(DATA / "residuals_by_weekday.csv")
    sub = df[(df["horizon_days"] == 42) & (df["model"].isin(["lightgbm", "torch_gru", "ensemble"]))]
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig, ax = plt.subplots(figsize=(COL, 1.6))
    width = 0.27
    for i, m in enumerate(["lightgbm", "torch_gru", "ensemble"]):
        s = sub[sub["model"] == m].set_index("dow").reindex(names)
        x = [k + (i - 1) * width for k in range(7)]
        ax.bar(x, s["wape"].fillna(0) * 100, width, color=C[m], label=LABEL[m])
    ax.set_xticks(range(7))
    ax.set_xticklabels(names)
    ax.set_ylabel("WAPE (%)")
    ax.set_ylim(0, 15.5)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.savefig(FIG / "fig_weekday.pdf")
    plt.close(fig)


def fig_coverage() -> None:
    df = pd.read_csv(DATA / "interval_coverage.csv")
    df = df[~df["model"].str.contains("local")]
    df = df[~((df["model"] == "ensemble") & (df["horizon_days"] == 30))]
    labels, values, colors = [], [], []
    for h in (42, 30):
        for m in ORDER:
            s = df[(df["model"] == m) & (df["horizon_days"] == h)]
            if len(s):
                labels.append(f"{LABEL[m]} ({h} d)")
                values.append(s.iloc[0]["coverage_holdout"] * 100)
                colors.append(C[m])
    fig, ax = plt.subplots(figsize=(COL, 2.3))
    y = list(range(len(values)))
    ax.barh(y, values, color=colors, edgecolor="black", linewidth=0.3)
    for i, v in enumerate(values):
        ax.text(v + 0.3, i, f"{v:.1f}", va="center", fontsize=5.5)
    ax.axvline(80, color="black", lw=0.8, ls="--")
    ax.text(80.3, -0.75, "nominal 80 %", fontsize=5.5, va="center")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_ylim(len(values) - 0.5, -1.3)
    ax.set_xlim(70, 88)
    ax.set_xlabel("Hold-out coverage of the P10–P90 band (%)")
    fig.savefig(FIG / "fig_coverage.pdf")
    plt.close(fig)


def fig_ensemble() -> None:
    df = pd.read_csv(DATA / "ensemble_holdout.csv")
    sub = df[df["horizon_days"] == 42].set_index("variant")
    order = [k for k in ("lightgbm", "torch_gru", "equal_weight_blend", "fitted_blend") if k in sub.index]
    names = {
        "lightgbm": "LightGBM",
        "torch_gru": "GRU",
        "equal_weight_blend": "Equal blend",
        "fitted_blend": "Fitted blend\n(0.75 / 0.25)",
    }
    fig, ax = plt.subplots(figsize=(COL, 1.5))
    vals = sub.loc[order, "wape"] * 100
    colors = [C["lightgbm"], C["torch_gru"], "#7f7f7f", C["ensemble"]][: len(order)]
    bars = ax.barh([names[k] for k in order], vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(v + 0.03, b.get_y() + b.get_height() / 2, f"{v:.2f}", va="center", fontsize=6)
    ax.set_xlim(7.5, 8.6)
    ax.invert_yaxis()
    ax.set_xlabel("WAPE (%) on the untouched second half of the 42-day window")
    fig.savefig(FIG / "fig_ensemble.pdf")
    plt.close(fig)


def fig_latency() -> None:
    df = pd.read_csv(DATA / "api_latency.csv")
    fig, ax = plt.subplots(figsize=(COL, 1.75))
    y = range(len(df))
    ax.barh(y, df["median_ms"], color=C["lightgbm"], label="median")
    ax.errorbar(df["p95_ms"], y, fmt="|", color="black", ms=6, label="95th percentile")
    ax.set_yticks(list(y))
    ax.set_yticklabels(df["endpoint"].str.replace("GET ", ""), fontsize=6)
    ax.set_xscale("log")
    ax.set_xlabel("Latency (ms, log scale), n = 30 sequential calls")
    ax.invert_yaxis()
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(FIG / "fig_latency.pdf")
    plt.close(fig)


def fig_fit_time() -> None:
    gpu = pd.read_csv(DATA / "model_comparison.csv")
    gpu = gpu[gpu["horizon_days"] == 42]
    local = pd.read_csv(DATA / "model_runs_local.csv")
    local = local[
        (local["dataset"] == "Rossmann full (all stores)")
        & (local["horizon_days"] == 42)
        & (local["status"] == "succeeded")
        & (local["device"] == "cpu")
    ]
    rows = []
    for m in ("lightgbm", "torch_mlp", "torch_gru", "ensemble"):
        g = gpu[gpu["model"] == m]
        if len(g):
            dev = "GPU host (RTX 6000 Ada)" if m != "lightgbm" else "GPU host (CPU, 16 threads)"
            rows.append({"model": LABEL[m], "where": dev, "seconds": g.iloc[0]["fit_seconds"]})
    for m in ("seasonal_naive", "lightgbm", "torch_mlp", "arima", "sarima"):
        lo = local[local["family"] == m]
        if len(lo):
            rows.append(
                {"model": LABEL[m], "where": "laptop CPU (i7-8665U)", "seconds": lo.iloc[0]["fit_seconds"]}
            )
    df = pd.DataFrame(rows)
    df.to_csv(DATA / "fig_fit_time_source.csv", index=False)
    fig, ax = plt.subplots(figsize=(COL, 1.9))
    colors = {
        "GPU host (RTX 6000 Ada)": "#009E73",
        "GPU host (CPU, 16 threads)": "#56B4E9",
        "laptop CPU (i7-8665U)": "#E69F00",
    }
    labels = [f"{r.model} — {r.where}" for r in df.itertuples()]
    ax.barh(labels, df["seconds"].clip(lower=0.05), color=[colors[w] for w in df["where"]])
    for i, v in enumerate(df["seconds"]):
        ax.text(max(v, 0.05) * 1.15, i, f"{v:,.1f} s", va="center", fontsize=5.5)
    ax.set_xscale("log")
    ax.set_xlim(0.03, 20000)
    ax.set_xlabel("Training time (s, log scale), 42-day horizon, 1,115 stores")
    ax.tick_params(axis="y", labelsize=5.6)
    ax.invert_yaxis()
    fig.savefig(FIG / "fig_fit_time.pdf")
    plt.close(fig)


def fig_backtest() -> None:
    path = DATA / "backtest_per_origin.csv"
    if not path.exists():
        return
    po = pd.read_csv(path).sort_values("test_start")
    labels = [pd.Timestamp(s).strftime("%d %b\n%Y") for s in po["test_start"]]
    fig, ax = plt.subplots(figsize=(COL, 1.9))
    x = range(len(po))
    ax.plot(
        x,
        po["wape_seasonal_naive"] * 100,
        marker="o",
        ms=3,
        lw=1,
        color=C["seasonal_naive"],
        label="Seasonal naive",
    )
    ax.plot(
        x,
        po["wape_lightgbm_honest"] * 100,
        marker="o",
        ms=3,
        lw=1.2,
        color=C["lightgbm"],
        label="LightGBM (honest stopping)",
    )
    ax.plot(
        x,
        po["wape_lightgbm_teststop"] * 100,
        marker="x",
        ms=3,
        lw=0.8,
        ls="--",
        color="#555555",
        label="LightGBM (stopping on test window)",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=5.5)
    ax.set_xlabel("Start of the 28-day test window (origin)")
    ax.set_ylabel("WAPE (%)")
    ax.set_ylim(0, po["wape_seasonal_naive"].max() * 100 + 12)
    ax.legend(frameon=False, fontsize=5.8, loc="upper right")
    fig.savefig(FIG / "fig_backtest.pdf")
    plt.close(fig)


def fig_staffing() -> None:
    path = DATA / "staffing_frontier.csv"
    if not path.exists():
        return
    fr = pd.read_csv(path)
    s = pd.read_csv(DATA / "staffing_ip_summary.csv").set_index("plan")
    fig, ax = plt.subplots(figsize=(COL, 2.0))
    ax.plot(
        fr["shortfall_staff_hours"] / 1000,
        fr["paid_hours"] / 1000,
        marker="o",
        ms=3,
        lw=1.2,
        color=C["ensemble"],
        label="Integer program, safety factor $\\lambda$",
    )
    for _, r in fr.iterrows():
        if r["lambda"] in (1.0, 1.06, 1.117, 1.14):
            ax.annotate(
                f"$\\lambda$={r['lambda']:.3g}",
                (r["shortfall_staff_hours"] / 1000, r["paid_hours"] / 1000),
                textcoords="offset points",
                xytext=(4, 3),
                fontsize=5.5,
            )
    rule = s.loc["rule_p50"]
    ax.scatter(
        [rule["shortfall_staff_hours"] / 1000],
        [rule["paid_hours"] / 1000],
        color=C["sarima"],
        zorder=3,
        s=22,
        marker="s",
        label="Rule-based plan (P50)",
    )
    ax.set_xlabel("Understaffing vs realised footfall (thousand staff-hours)")
    ax.set_ylabel("Paid staff-hours (thousand)")
    ax.legend(frameon=False, fontsize=6, loc="upper right")
    fig.savefig(FIG / "fig_staffing.pdf")
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for f in (
        fig_accuracy,
        fig_week_ahead,
        fig_weekday,
        fig_coverage,
        fig_ensemble,
        fig_latency,
        fig_fit_time,
        fig_backtest,
        fig_staffing,
    ):
        f()
        print("drew", f.__name__)


if __name__ == "__main__":
    main()
