"""CSV and PDF report rendering for forecast runs."""

from __future__ import annotations

import io
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _table(data: list[list[Any]], col_widths: list[float] | None = None) -> Table:
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
            ]
        )
    )
    return t


def _fmt(v: Any) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:,.2f}"
    return str(v)


def forecast_pdf(
    *,
    title: str,
    org_name: str,
    store: dict[str, Any] | None,
    model: dict[str, Any],
    forecast: dict[str, Any],
    daily: pd.DataFrame,
    totals: dict[str, Any] | None,
    assumptions: list[str],
    max_rows: int = 120,
) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=landscape(A4),
        leftMargin=12 * mm,
        rightMargin=12 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
        title=title,
    )
    styles = getSampleStyleSheet()
    story: list[Any] = [
        Paragraph(title, styles["Title"]),
        Paragraph(f"{org_name} &middot; generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC", styles["Normal"]),
        Spacer(1, 6 * mm),
    ]
    meta = [["Field", "Value"]]
    if store:
        meta += [
            ["Store", f"{store['code']} - {store['name']}"],
            ["Currency", store["currency"]],
            ["Target", store["target_type"]],
        ]
    meta += [
        ["Model", f"{model['model_family']} ({model['run_type']})"],
        ["Validation WAPE", _fmt(model["metrics"].get("wape"))],
        ["Validation RMSPE", _fmt(model["metrics"].get("rmspe"))],
        [
            "Forecast window",
            f"{forecast['summary'].get('start_date')} -> {forecast['summary'].get('end_date')}",
        ],
        ["Kind", forecast["kind"]],
    ]
    story += [_table(meta, [60 * mm, 120 * mm]), Spacer(1, 6 * mm)]

    if totals:
        story.append(Paragraph("Totals", styles["Heading2"]))
        rows = [["Metric", "Value"]] + [[k.replace("_", " "), _fmt(v)] for k, v in totals.items()]
        story += [_table(rows, [80 * mm, 60 * mm]), Spacer(1, 6 * mm)]

    if assumptions:
        story.append(Paragraph("Assumptions", styles["Heading2"]))
        story += [Paragraph(f"&bull; {a}", styles["Normal"]) for a in assumptions]

    story += [PageBreak(), Paragraph("Daily detail", styles["Heading2"])]
    shown = daily.head(max_rows)
    data = [list(shown.columns)] + [[_fmt(v) for v in row] for row in shown.itertuples(index=False)]
    story.append(_table(data))
    if len(daily) > max_rows:
        story.append(Paragraph(f"... {len(daily) - max_rows} more rows in the CSV export.", styles["Italic"]))
    doc.build(story)
    return buf.getvalue()
