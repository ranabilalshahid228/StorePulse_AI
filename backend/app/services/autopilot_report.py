"""Store Autopilot exports: executive PDF (charts, KPIs, model comparison, P&L, shift plan,
actions), Excel workbook and CSV - all rendered from a stored Autopilot result."""

from __future__ import annotations

import io
from datetime import UTC, datetime
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4, landscape  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

from app.services.reports import _table  # noqa: E402

PRIORITY_COLORS = {"critical": "#b91c1c", "high": "#dc2626", "medium": "#d97706", "low": "#2563eb"}
SERIES = {"p50": "#2563eb", "band": "#93c5fd", "baseline": "#64748b", "profit": "#16a34a", "loss": "#dc2626"}
DAILY_COLUMNS = [
    "ds",
    "is_open",
    "is_promo",
    "event",
    "revenue_p10",
    "revenue_p50",
    "revenue_p90",
    "baseline_revenue_p50",
    "footfall",
    "basket_size",
    "gross_revenue",
    "promo_discount",
    "cogs",
    "gross_profit",
    "labor_hours",
    "labor_cost",
    "fixed_overhead",
    "net_profit_p10",
    "net_profit_p50",
    "net_profit_p90",
    "break_even_revenue",
    "surge_staff",
    "utilization",
    "staffing_status",
]


def _has_interval(result: dict[str, Any]) -> bool:
    if "interval_available" in result:
        return bool(result["interval_available"])
    return not any("no interval available" in line for line in result.get("assumptions", []))


def daily_frame(result: dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame(result["days"])
    staff = [f"staff_{n}" for n in result.get("shift_names", [])]
    cols = [c for c in DAILY_COLUMNS[:15] if c in df] + [c for c in staff if c in df]
    cols += [c for c in DAILY_COLUMNS[15:] if c in df]
    if not _has_interval(result):
        df = df.assign(**{c: None for c in cols if c.endswith(("_p10", "_p90"))})
    return df[cols]


def to_csv(result: dict[str, Any]) -> bytes:
    return daily_frame(result).to_csv(index=False).encode("utf-8")


def to_xlsx(result: dict[str, Any]) -> bytes:
    k = result["kpis"]
    if not _has_interval(result):
        k = {key: None if key.endswith(("_p10", "_p90")) else value for key, value in k.items()}
    summary = pd.DataFrame(
        [{"metric": key.replace("_", " "), "value": v} for key, v in k.items() if not isinstance(v, dict)]
    )
    sheets = {
        "Summary": summary,
        "Daily plan": daily_frame(result),
        "Shift plan": pd.DataFrame(result["shifts"]),
        "Recommendations": pd.DataFrame(
            [{**r, "dates": ", ".join(r["dates"])} for r in result["recommendations"]]
            or [{"priority": "", "title": "No actions"}]
        ),
        "Model comparison": pd.DataFrame(result.get("model_comparison") or [{"model_family": "n/a"}]),
        "Assumptions": pd.DataFrame({"assumption": result["assumptions"]}),
    }
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        for name, df in sheets.items():
            df.to_excel(xl, sheet_name=name, index=False)
            ws = xl.sheets[name]
            ws.freeze_panes = "A2"
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col[:200])
                ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 60)
    return buf.getvalue()


# ----------------------------------------------------------------------------------------
# PDF
# ----------------------------------------------------------------------------------------


def _png(fig: Any) -> io.BytesIO:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


def _style(ax: Any, title: str) -> None:
    ax.set_title(title, fontsize=10, loc="left")
    ax.grid(axis="y", alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=7)


def revenue_chart(df: pd.DataFrame, currency: str, *, show_band: bool = True) -> io.BytesIO:
    ds = pd.to_datetime(df["ds"])
    fig, ax = plt.subplots(figsize=(10, 3.2))
    if show_band:
        ax.fill_between(
            ds, df["revenue_p10"], df["revenue_p90"], color=SERIES["band"], alpha=0.5, label="P10-P90"
        )
    ax.plot(ds, df["revenue_p50"], color=SERIES["p50"], linewidth=2, label="Revenue P50")
    if "baseline_revenue_p50" in df and df["baseline_revenue_p50"].notna().any():
        ax.plot(ds, df["baseline_revenue_p50"], color=SERIES["baseline"], linestyle="--", label="Baseline")
    be = pd.to_numeric(df["break_even_revenue"], errors="coerce")
    if be.notna().any():
        ax.plot(ds, be, color=SERIES["loss"], linewidth=1, linestyle=":", label="Break-even")
    _style(ax, f"Daily revenue forecast ({currency})")
    ax.legend(fontsize=7, frameon=False, ncol=4)
    fig.autofmt_xdate()
    return _png(fig)


def profit_chart(df: pd.DataFrame, currency: str) -> io.BytesIO:
    ds = pd.to_datetime(df["ds"])
    np50 = df["net_profit_p50"].astype(float)
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.bar(ds, np50, color=[SERIES["profit"] if v >= 0 else SERIES["loss"] for v in np50], width=0.8)
    ax.axhline(0, color="#334155", linewidth=0.8)
    _style(ax, f"Daily net profit P50 ({currency})")
    fig.autofmt_xdate()
    return _png(fig)


def staffing_chart(df: pd.DataFrame, shift_names: list[str]) -> io.BytesIO:
    ds = pd.to_datetime(df["ds"])
    palette = ["#1d4ed8", "#0891b2", "#7c3aed", "#ea580c", "#16a34a", "#db2777"]
    fig, ax = plt.subplots(figsize=(10, 2.8))
    bottom = None
    for i, name in enumerate(shift_names):
        v = df[f"staff_{name}"].astype(float)
        ax.bar(ds, v, bottom=bottom, label=name, color=palette[i % len(palette)], width=0.8)
        bottom = v if bottom is None else bottom + v
    if "surge_staff" in df and df["surge_staff"].astype(float).gt(0).any():
        ax.bar(
            ds, df["surge_staff"].astype(float), bottom=bottom, label="Peak cover", color="#fbbf24", width=0.8
        )
    _style(ax, "Headcount per shift")
    ax.legend(fontsize=7, frameon=False, ncol=len(shift_names) + 1)
    fig.autofmt_xdate()
    return _png(fig)


def _m(v: Any, cur: str = "") -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int | float):
        return f"{cur + ' ' if cur else ''}{v:,.0f}"
    return str(v)


def _pct(v: Any) -> str:
    return "-" if v is None else f"{v:.1%}"


def executive_pdf(result: dict[str, Any], *, title: str, org_name: str, store_name: str) -> bytes:
    cur = result["currency"]
    k, t, w = result["kpis"], result["totals"], result["window"]
    df = pd.DataFrame(result["days"])
    show_band = _has_interval(result)

    def interval_text(row: dict[str, Any], metric: str, currency: str = "") -> str:
        if not show_band:
            return "Unavailable"
        return f"{_m(row[f'{metric}_p10'], currency)} - {_m(row[f'{metric}_p90'], currency)}"

    styles = getSampleStyleSheet()
    small = ParagraphStyle("small", parent=styles["Normal"], fontSize=8, leading=10)
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=landscape(A4),
        leftMargin=12 * mm,
        rightMargin=12 * mm,
        topMargin=10 * mm,
        bottomMargin=10 * mm,
        title=title,
    )
    width = doc.width
    story: list[Any] = [
        Paragraph(title, styles["Title"]),
        Paragraph(
            f"{org_name} &middot; {store_name} (store {result['store_code']}) &middot; "
            f"{w['start']} to {w['end']} ({w['days']} days, {w['kind'].replace('_', ' ')}) &middot; "
            f"generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC",
            styles["Normal"],
        ),
        Spacer(1, 4 * mm),
    ]
    kpi = [
        ["Revenue P50", "Revenue P10 - P90", "Net profit P50", "Net margin", "Break-even revenue"],
        [
            _m(k["revenue_p50"], cur),
            interval_text(k, "revenue", cur),
            _m(k["net_profit_p50"], cur),
            _pct(k["net_margin_pct"]),
            _m(k["break_even_revenue"], cur),
        ],
        ["Footfall", "Labour hours", "Labour cost", "Open days", "Actions (high / medium / low)"],
        [
            _m(k["footfall"]),
            _m(k["labor_hours"]),
            _m(k["labor_cost"], cur),
            f"{k['open_days']} / {k['days']}",
            " / ".join(str(k["recommendations"][p]) for p in ("high", "medium", "low")),
        ],
    ]
    kt = _table(kpi, [width / 5] * 5)
    kt.setStyle(
        [
            ("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#0f172a")),
            ("TEXTCOLOR", (0, 2), (-1, 2), colors.white),
        ]
    )
    story += [kt, Spacer(1, 3 * mm)]
    if result.get("scenario_impact"):
        imp = result["scenario_impact"]
        story.append(
            Paragraph(
                f"<b>Scenario vs baseline:</b> revenue {imp['revenue_p50']:+,.0f} {cur}, net profit "
                f"{imp['net_profit_p50']:+,.0f} {cur}, labour {imp['labor_cost']:+,.0f} {cur}.",
                styles["Normal"],
            )
        )
    if result.get("cold_start"):
        story.append(Paragraph(f"<b>Cold start:</b> {result['cold_start']['note']}", styles["Normal"]))
    story.append(Image(revenue_chart(df, cur, show_band=show_band), width=width, height=width * 0.32))

    story += [Paragraph("Autopilot actions", styles["Heading2"])]
    recs = result["recommendations"] or [
        {"priority": "low", "title": "No actions needed", "detail": "", "impact": None}
    ]
    rows = [["Priority", "Action", "Detail", f"Impact ({cur})"]]
    for r in recs[:14]:
        rows.append(
            [
                Paragraph(
                    f"<font color='{PRIORITY_COLORS[r['priority']]}'><b>{r['priority'].upper()}</b></font>",
                    small,
                ),
                Paragraph(f"<b>{r['title']}</b>", small),
                Paragraph(r["detail"], small),
                _m(r.get("impact")),
            ]
        )
    story.append(_table(rows, [20 * mm, 60 * mm, width - 105 * mm, 25 * mm]))

    story += [PageBreak(), Paragraph("Profit & loss", styles["Heading2"])]
    pnl_rows = [
        ["Line", f"Amount ({cur})", "Notes"],
        ["Gross (list-price) revenue", _m(t["gross_revenue"]), ""],
        ["Promo discount", _m(-(t["promo_discount"] or 0)), "realized vs list price on promo days"],
        ["Net revenue P50", _m(t["revenue_p50"]), interval_text(t, "revenue")],
        ["COGS", _m(-(t["cogs"] or 0)), ""],
        ["Gross profit", _m(t["gross_profit"]), f"margin {_pct(t['gross_margin_pct'])}"],
        ["Labour cost", _m(-(t["labor_cost"] or 0)), f"{_m(k['labor_hours'])} scheduled hours"],
        ["Fixed overhead", _m(-(t["fixed_overhead"] or 0)), "every calendar day"],
        [
            "Net profit P50",
            _m(t["net_profit_p50"]),
            interval_text(t, "net_profit"),
        ],
        [
            "Break-even revenue",
            _m(t["break_even_revenue"]),
            f"margin of safety {_pct(t['margin_of_safety_pct'])}",
        ],
    ]
    story += [
        KeepTogether([_table(pnl_rows, [70 * mm, 40 * mm, 110 * mm])]),
        Spacer(1, 3 * mm),
        Image(profit_chart(df, cur), width=width, height=width * 0.28),
        Paragraph("Workforce plan", styles["Heading2"]),
        Image(staffing_chart(df, result["shift_names"]), width=width, height=width * 0.28),
    ]

    story += [PageBreak(), Paragraph("Daily plan", styles["Heading2"])]
    names = result["shift_names"]
    head = [
        "Date",
        "Open",
        "Promo",
        "Revenue P50",
        "P10 - P90",
        "Footfall",
        *names,
        "Peak +",
        "Labour",
        "Net P50",
        "Status",
    ]
    rows = [head]
    for d in result["days"][:62]:
        rows.append(
            [
                d["ds"],
                _m(d["is_open"]),
                _m(d["is_promo"]),
                _m(d["revenue_p50"]),
                interval_text(d, "revenue"),
                _m(d["footfall"]),
                *[str(d.get(f"staff_{n}", 0)) for n in names],
                str(d.get("surge_staff", 0)),
                _m(d["labor_cost"]),
                _m(d["net_profit_p50"]),
                d.get("staffing_status") or "",
            ]
        )
    story.append(_table(rows))

    comp = result.get("model_comparison") or []
    story += [Spacer(1, 4 * mm), Paragraph("Model comparison (validation)", styles["Heading2"])]
    used = (result.get("model") or {}).get("model_run_id")
    if comp:
        rows = [["Model", "Family", "Horizon", "Accuracy", "WAPE", "RMSPE", "RMSSE", "R2", "P10-P90 cov."]]
        for c in comp[:10]:
            rows.append(
                [
                    ("> " if c["model_run_id"] == used else "") + (c.get("name") or c["model_run_id"][:8]),
                    c["model_family"],
                    str(c["horizon_days"]),
                    _pct(c.get("accuracy")),
                    *[
                        f"{c[x]:.4f}" if c.get(x) is not None else "-"
                        for x in ("wape", "rmspe", "rmsse", "r2")
                    ],
                    _pct(c.get("interval_coverage")),
                ]
            )
        story += [_table(rows), Paragraph("&gt; marks the model used for this plan.", small)]
    else:
        story.append(Paragraph("No finished model runs to compare.", small))

    story += [Paragraph("Assumptions", styles["Heading2"])]
    story += [Paragraph(f"&bull; {a}", small) for a in result["assumptions"]]
    doc.build(story)
    return buf.getvalue()
