"use client";

import { useMemo, useState, type ReactNode } from "react";
import { AlertTriangle, CalendarX2, Database, Download, FileSpreadsheet, FileText, Info, Sparkles } from "lucide-react";
import { api } from "@/lib/api";
import type { AutopilotResult, HealthReport, Priority } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Badge, Card, CardContent, CardDescription, CardHeader, CardTitle, Table, TD, TH, THead, TR } from "@/components/ui/primitives";
import { ErrorNotice } from "@/components/shared";
import { RevenuePlanChart, SimpleBarChart, WaterfallChart } from "@/components/charts";
import { cn, formatCompact, formatCurrency, formatNumber, formatPct } from "@/lib/utils";

export const PRIORITY_TONE: Record<Priority, "danger" | "warning" | "info" | "neutral"> = {
  critical: "danger",
  high: "danger",
  medium: "warning",
  low: "info",
};

const STATUS_TONE = { understaffed: "danger", overstaffed: "warning", closed: "neutral", ok: "success" } as const;

/* ---- KPI with a P10–P90 range --------------------------------------------------------------- */

export function KpiRange({
  label,
  value,
  low,
  high,
  sub,
  tone,
}: {
  label: string;
  value: string;
  low?: string;
  high?: string;
  sub?: ReactNode;
  tone?: "loss" | "gain";
}) {
  return (
    <Card>
      <CardContent className="p-5">
        <p className="text-xs font-medium text-muted-foreground">{label}</p>
        <p className={cn("mt-2 text-2xl font-semibold tracking-tight tabular-nums", tone === "loss" && "text-destructive")}>{value}</p>
        {low !== undefined && high !== undefined ? (
          <p className="mt-1 text-xs tabular-nums text-muted-foreground">
            P10 <span className="font-medium text-foreground">{low}</span> · P90 <span className="font-medium text-foreground">{high}</span>
          </p>
        ) : null}
        {sub ? <p className="mt-1 text-xs text-muted-foreground">{sub}</p> : null}
      </CardContent>
    </Card>
  );
}

export function hasPlanInterval(res: AutopilotResult) {
  return res.interval_available ?? !res.assumptions.some((line) => line.includes("no interval available"));
}

export function PlanKpis({ res }: { res: AutopilotResult }) {
  const k = res.kpis;
  const cur = res.currency;
  const hasInterval = hasPlanInterval(res);
  const isSales = res.target_type === "sales";
  const demand = (v: number | undefined) => (v === undefined ? "—" : isSales ? formatCurrency(v, cur) : formatCompact(v));
  const delta = res.scenario_impact;
  return (
    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
      <KpiRange
        label={isSales ? "Forecasted sales" : res.target_type === "units" ? "Forecasted units" : "Forecasted customers"}
        value={demand(k.demand_p50 ?? k.revenue_p50)}
        low={hasInterval ? demand(k.demand_p10 ?? k.revenue_p10) : undefined}
        high={hasInterval ? demand(k.demand_p90 ?? k.revenue_p90) : undefined}
        sub={
          delta
            ? `${delta.revenue_p50 >= 0 ? "+" : ""}${formatCurrency(delta.revenue_p50, cur)} revenue vs baseline`
            : `${k.open_days} of ${k.days} days open`
        }
      />
      <KpiRange
        label="Estimated footfall"
        value={formatNumber(k.footfall)}
        low={hasInterval && k.footfall_p10 !== undefined ? formatNumber(k.footfall_p10) : undefined}
        high={hasInterval && k.footfall_p90 !== undefined ? formatNumber(k.footfall_p90) : undefined}
        sub="customers over the window"
      />
      <KpiRange
        label={k.net_profit_p50 < 0 ? "Projected net loss" : "Projected net profit"}
        value={formatCurrency(k.net_profit_p50, cur)}
        low={hasInterval ? formatCurrency(k.net_profit_p10, cur) : undefined}
        high={hasInterval ? formatCurrency(k.net_profit_p90, cur) : undefined}
        tone={k.net_profit_p50 < 0 ? "loss" : undefined}
        sub={
          delta
            ? `${delta.net_profit_p50 >= 0 ? "+" : ""}${formatCurrency(delta.net_profit_p50, cur)} vs baseline`
            : `margin ${formatPct(k.net_margin_pct)} · break-even ${formatCurrency(k.break_even_revenue, cur)}`
        }
      />
      <KpiRange
        label="Staff assignments per open day"
        value={`${formatNumber(k.avg_staff_per_open_day ?? null, 1)} assignments`}
        sub={`${formatNumber(k.labor_hours)} h · ${formatCurrency(k.labor_cost, cur)} · peak ${k.peak_staff_per_shift}/shift`}
      />
    </div>
  );
}

/* ---- Alerts ----------------------------------------------------------------------------------- */

function Banner({ tone, icon, title, children }: { tone: "warning" | "danger" | "info"; icon: ReactNode; title: string; children: ReactNode }) {
  return (
    <div
      className={cn(
        "flex items-start gap-3 rounded-lg border p-3 text-sm",
        tone === "danger" && "border-destructive/30 bg-destructive/5",
        tone === "warning" && "border-warning/40 bg-warning/10",
        tone === "info" && "border-border bg-secondary/60",
      )}
    >
      <span className="mt-0.5 shrink-0 [&_svg]:size-4">{icon}</span>
      <div>
        <p className="font-medium">{title}</p>
        <div className="text-muted-foreground">{children}</div>
      </div>
    </div>
  );
}

export function AlertBanners({ res, health }: { res: AutopilotResult; health?: HealthReport | null }) {
  const closed = res.days.filter((d) => !d.is_open);
  const dataIssues = (health?.issues ?? []).filter((i) => i.severity !== "info");
  const anomalies = res.anomalies ?? [];
  const under = res.kpis.understaffed_shifts;
  if (!closed.length && !dataIssues.length && !anomalies.length && !res.cold_start && !under) return null;
  return (
    <div className="flex flex-col gap-2">
      {res.cold_start ? (
        <Banner tone="warning" icon={<Sparkles />} title="New store — cohort-based forecast">
          {res.cold_start.note}
        </Banner>
      ) : null}
      {closed.length ? (
        <Banner tone="info" icon={<CalendarX2 />} title={`Store closed on ${closed.length} day(s)`}>
          {closed.slice(0, 6).map((d) => d.ds).join(", ")}
          {closed.length > 6 ? ` +${closed.length - 6} more` : ""} — no sales, footfall or shifts; fixed overhead still accrues.
        </Banner>
      ) : null}
      {under ? (
        <Banner tone="danger" icon={<AlertTriangle />} title={`${under} shift(s) exceed store capacity`}>
          Demand needs more staff than the configured maximum per shift — see the actions below.
        </Banner>
      ) : null}
      {anomalies.length ? (
        <Banner tone="warning" icon={<Info />} title={`${anomalies.length} forecast / data anomaly warning(s)`}>
          {anomalies.slice(0, 3).map((a) => a.detail).join(" ")}
        </Banner>
      ) : null}
      {dataIssues.length ? (
        <Banner tone={dataIssues.some((i) => i.severity === "critical") ? "danger" : "warning"} icon={<Database />} title="Data quality">
          {dataIssues.slice(0, 3).map((i) => i.message).join(" · ")}
          {health ? ` (health score ${health.score}/100)` : ""}
        </Banner>
      ) : null}
    </div>
  );
}

/* ---- Recommendations ----------------------------------------------------------------------- */

export function RecommendationsFeed({ res, limit }: { res: AutopilotResult; limit?: number }) {
  const recs = limit ? res.recommendations.slice(0, limit) : res.recommendations;
  return (
    <Card>
      <CardHeader>
        <CardTitle>Autopilot actions</CardTitle>
        <CardDescription>Prioritized by severity, then money at stake.</CardDescription>
      </CardHeader>
      <CardContent>
        {recs.length === 0 ? (
          <p className="text-sm text-muted-foreground">No actions needed for this window.</p>
        ) : (
          <ul className="flex flex-col divide-y">
            {recs.map((rec, i) => (
              <li key={i} className="flex gap-3 py-3">
                <Badge tone={PRIORITY_TONE[rec.priority]} className="h-fit shrink-0 uppercase">{rec.priority}</Badge>
                <div className="min-w-0 flex-1">
                  <p className="text-sm font-medium">{rec.title}</p>
                  <p className="text-sm text-muted-foreground">{rec.detail}</p>
                  <p className="mt-0.5 text-[11px] uppercase tracking-wide text-muted-foreground">{rec.category.replace("_", " ")}</p>
                </div>
                {rec.impact !== null ? (
                  <span className={cn("shrink-0 text-sm tabular-nums", rec.impact < 0 ? "text-destructive" : "text-success")}>
                    {formatCurrency(rec.impact, res.currency)}
                  </span>
                ) : null}
              </li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}

/* ---- Charts ------------------------------------------------------------------------------------ */

export function usePlanChartData(res: AutopilotResult) {
  return useMemo(
    () => res.days.map((d) => ({ ...d, band: [d.revenue_p10, d.revenue_p90], surge: d.surge_staff })),
    [res.days],
  );
}

export function RevenueCard({ res }: { res: AutopilotResult }) {
  const data = usePlanChartData(res);
  const showBand = hasPlanInterval(res);
  return (
    <Card>
      <CardHeader>
        <CardTitle>Revenue forecast</CardTitle>
        <CardDescription>P50{showBand ? " with the P10–P90 band" : " (forecast interval unavailable)"}{res.scenario_impact ? ", scenario baseline" : ""} and daily break-even</CardDescription>
      </CardHeader>
      <CardContent><RevenuePlanChart data={data} showBand={showBand} /></CardContent>
    </Card>
  );
}

export function StaffingCard({ res }: { res: AutopilotResult }) {
  const data = usePlanChartData(res);
  return (
    <Card>
      <CardHeader><CardTitle>Headcount per shift</CardTitle><CardDescription>Stacked by shift, plus peak-hour cover</CardDescription></CardHeader>
      <CardContent>
        <SimpleBarChart
          data={data}
          xKey="ds"
          stacked
          series={[...res.shift_names.map((n) => ({ key: `staff_${n}`, label: n })), { key: "surge", label: "Peak cover", color: "var(--chart-4)" }]}
        />
      </CardContent>
    </Card>
  );
}

export function ProfitCard({ res }: { res: AutopilotResult }) {
  const data = usePlanChartData(res);
  return (
    <Card>
      <CardHeader><CardTitle>Daily net profit</CardTitle><CardDescription>P50 after COGS, labour and overhead</CardDescription></CardHeader>
      <CardContent>
        <SimpleBarChart
          data={data}
          xKey="ds"
          series={[
            { key: "net_profit_p50", label: "Net profit P50" },
            ...(res.scenario_impact ? [{ key: "baseline_net_profit_p50", label: "Baseline", color: "var(--chart-5)" }] : []),
          ]}
        />
      </CardContent>
    </Card>
  );
}

/* ---- P&L ----------------------------------------------------------------------------------------- */

export function PnlWaterfallCard({ res }: { res: AutopilotResult }) {
  const t = res.totals;
  const cur = res.currency;
  const fmt = (v: number) => formatCurrency(v, cur);
  return (
    <Card>
      <CardHeader>
        <CardTitle>P&L breakdown</CardTitle>
        <CardDescription>
          {res.window.start} → {res.window.end} · gross margin {formatPct(t.gross_margin_pct)} · net margin {formatPct(t.net_margin_pct)}
        </CardDescription>
      </CardHeader>
      <CardContent className="grid gap-6 lg:grid-cols-5">
        <div className="lg:col-span-3">
          <WaterfallChart
            valueFormatter={formatCompact}
            steps={[
              { label: "List revenue", value: t.gross_revenue, total: true },
              { label: "Promo disc.", value: -t.promo_discount },
              { label: "COGS", value: -t.cogs },
              { label: "Labour", value: -t.labor_cost },
              { label: "Overhead", value: -t.fixed_overhead },
              { label: "Net profit", value: t.net_profit_p50, total: true },
            ]}
          />
        </div>
        <div className="min-w-0 lg:col-span-2">
        <Table>
          <tbody>
            {(
              [
                ["Gross (list-price) revenue", t.gross_revenue],
                ["Promo discount", -t.promo_discount],
                ["Net revenue P50", t.revenue_p50],
                ["COGS", -t.cogs],
                ["Gross profit", t.gross_profit],
                ["Labour", -t.labor_cost],
                ["Fixed overhead", -t.fixed_overhead],
                ["Net profit P50", t.net_profit_p50],
              ] as [string, number][]
            ).map(([label, v]) => (
              <TR key={label} className={label.startsWith("Net profit") || label === "Gross profit" ? "font-medium" : ""}>
                <TD>{label}</TD>
                <TD className={cn("text-right tabular-nums", v < 0 && "text-destructive")}>{fmt(v)}</TD>
              </TR>
            ))}
            {hasPlanInterval(res) && <TR>
              <TD className="text-muted-foreground">Net profit P10 / P90</TD>
              <TD className="text-right tabular-nums text-muted-foreground">{fmt(t.net_profit_p10)} / {fmt(t.net_profit_p90)}</TD>
            </TR>}
            <TR>
              <TD className="text-muted-foreground">Break-even revenue · safety</TD>
              <TD className="text-right tabular-nums text-muted-foreground">{fmt(t.break_even_revenue ?? NaN)} · {formatPct(t.margin_of_safety_pct)}</TD>
            </TR>
          </tbody>
        </Table>
        </div>
      </CardContent>
    </Card>
  );
}

/* ---- Workforce roster ------------------------------------------------------------------------- */

export function ShiftRoster({ res }: { res: AutopilotResult }) {
  const [detail, setDetail] = useState(false);
  const byDay = useMemo(() => {
    const m = new Map<string, Record<string, (typeof res.shifts)[number]>>();
    for (const s of res.shifts) m.set(s.ds, { ...(m.get(s.ds) ?? {}), [s.shift]: s });
    return m;
  }, [res.shifts]);
  const cur = res.currency;
  return (
    <Card>
      <CardHeader className="flex flex-row items-start justify-between gap-4">
        <div>
          <CardTitle>Workforce roster</CardTitle>
          <CardDescription>Headcount per shift; {detail ? "hours, expected customers and utilization" : "toggle detail for hours and utilization"}</CardDescription>
        </div>
        <Button size="sm" variant="outline" onClick={() => setDetail((d) => !d)}>{detail ? "Compact" : "Detail"}</Button>
      </CardHeader>
      <CardContent className="overflow-x-auto">
        <Table>
          <THead>
            <tr>
              <TH>Date</TH>
              <TH className="text-right">Footfall</TH>
              {res.shift_names.map((n) => <TH key={n} className="text-right">{n}</TH>)}
              <TH className="text-right">Peak +</TH>
              <TH className="text-right">Hours</TH>
              <TH className="text-right">Labour</TH>
              <TH>Status</TH>
            </tr>
          </THead>
          <tbody>
            {res.days.map((d) => {
              const shifts = byDay.get(d.ds) ?? {};
              return (
                <TR key={d.ds} className={d.is_open ? "" : "text-muted-foreground"}>
                  <TD className="whitespace-nowrap">
                    {d.ds}
                    {d.is_promo ? <Badge tone="info" className="ml-2">promo</Badge> : null}
                    {d.event ? <span className="ml-2 text-xs text-muted-foreground">{d.event}</span> : null}
                  </TD>
                  <TD className="text-right tabular-nums">{formatNumber(d.footfall)}</TD>
                  {res.shift_names.map((n) => {
                    const s = shifts[n];
                    return (
                      <TD key={n} className={cn("text-right tabular-nums", s?.status === "understaffed" && "text-destructive", s?.status === "overstaffed" && "text-[oklch(0.5_0.12_70)]")}>
                        {s ? s.headcount : 0}
                        {detail && s && s.hours ? (
                          <span className="block text-[11px] text-muted-foreground">
                            {formatNumber(s.hours, 1)} h · {formatNumber(s.expected_customers)} cust · {formatPct(s.utilization, 0)}
                          </span>
                        ) : null}
                      </TD>
                    );
                  })}
                  <TD className="text-right tabular-nums">{d.surge_staff}</TD>
                  <TD className="text-right tabular-nums">{formatNumber(d.labor_hours)}</TD>
                  <TD className="text-right tabular-nums">{formatCurrency(d.labor_cost, cur)}</TD>
                  <TD><Badge tone={STATUS_TONE[d.staffing_status]}>{d.staffing_status}</Badge></TD>
                </TR>
              );
            })}
          </tbody>
        </Table>
      </CardContent>
    </Card>
  );
}

/* ---- Exports ------------------------------------------------------------------------------------- */

export function ExportButtons({ runId, res, size = "sm" }: { runId: string; res: AutopilotResult; size?: "sm" | "default" }) {
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const stem = `autopilot_${res.store_code}_${res.window.start}_${res.window.end}`;
  const dl = (path: string, ext: string) => {
    setError(null);
    setBusy(ext);
    api
      .download(`/autopilot/runs/${runId}/${path}`, `${stem}.${ext}`)
      .catch(setError)
      .finally(() => setBusy(null));
  };
  return (
    <div className="flex flex-col items-end gap-2">
      <div className="flex flex-wrap gap-2">
        <Button size={size} onClick={() => dl("report.pdf", "pdf")} disabled={busy !== null}>
          <FileText /> {busy === "pdf" ? "Rendering…" : "PDF report"}
        </Button>
        <Button size={size} variant="outline" onClick={() => dl("plan.xlsx", "xlsx")} disabled={busy !== null}>
          <FileSpreadsheet /> Excel
        </Button>
        <Button size={size} variant="outline" onClick={() => dl("plan.csv", "csv")} disabled={busy !== null}>
          <Download /> CSV
        </Button>
      </div>
      <ErrorNotice error={error} />
    </div>
  );
}

export function AssumptionsCard({ items }: { items: string[] }) {
  return (
    <Card>
      <CardHeader><CardTitle>Assumptions</CardTitle><CardDescription>Every input and convention this plan used</CardDescription></CardHeader>
      <CardContent>
        <ul className="list-disc space-y-1 pl-5 text-sm text-muted-foreground">
          {items.map((a) => <li key={a}>{a}</li>)}
        </ul>
      </CardContent>
    </Card>
  );
}
