"use client";

import { useMemo, useState } from "react";
import type { ModelRunDetail } from "@/lib/types";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, Select, Table, TD, TH, THead, TR } from "@/components/ui/primitives";
import { HorizontalBarChart, SimpleBarChart, TimeSeriesChart } from "@/components/charts";
import { cn, formatNumber, formatPct } from "@/lib/utils";

const SKIP = new Set(["epoch", "seconds", "iteration"]);

/** Per-epoch / per-iteration curves. Logs can interleave several models (quantile boosters,
 * ensemble members): pick one; losses and validation metrics are charted separately. */
export function EpochCurves({ logs }: { logs: Record<string, unknown>[] }) {
  const tag = (l: Record<string, unknown>) => String(l.member ?? l.model ?? "model");
  const tags = useMemo(() => [...new Set(logs.map(tag))], [logs]);
  const [picked, setPicked] = useState<string>("");
  const current = picked || tags[0] || "";
  const curve = useMemo(
    () => logs.filter((l) => tag(l) === current).map((l): Record<string, unknown> => ({ ...l, ds: l.epoch ?? l.iteration })),
    [logs, current],
  );
  const keys = curve.length ? Object.keys(curve[0]!).filter((k) => !SKIP.has(k) && k !== "ds" && typeof curve[0]![k] === "number") : [];
  const lossKeys = keys.filter((k) => /loss|l2|l1|tweedie|poisson|quantile|rmse|mae/i.test(k) && !k.startsWith("val_wape"));
  const valKeys = keys.filter((k) => !lossKeys.includes(k));
  const secs = curve.map((c) => Number(c.seconds)).filter((v) => Number.isFinite(v));

  return (
    <Card>
      <CardHeader className="flex flex-row items-start justify-between gap-3">
        <div>
          <CardTitle>Training curves</CardTitle>
          <CardDescription>
            {curve.length ? `${curve.length} ${curve[0]?.epoch !== undefined ? "epochs" : "iterations"}` : "No logs yet"}
            {secs.length ? ` · ${formatNumber(secs.reduce((a, b) => a + b, 0), 0)} s total` : ""}
          </CardDescription>
        </div>
        {tags.length > 1 ? (
          <Select value={current} onChange={(e) => setPicked(e.target.value)} className="w-40" aria-label="Model">
            {tags.map((t) => <option key={t}>{t}</option>)}
          </Select>
        ) : null}
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        {!curve.length ? (
          <p className="text-sm text-muted-foreground">Curves appear while the model trains.</p>
        ) : (
          <>
            {lossKeys.length ? <TimeSeriesChart data={curve} series={lossKeys.map((k) => ({ key: k, label: k }))} height={200} /> : null}
            {valKeys.length ? (
              <TimeSeriesChart data={curve} series={valKeys.map((k, i) => ({ key: k, label: k, color: `var(--chart-${((i + 2) % 5) + 1})` }))} height={200} />
            ) : null}
          </>
        )}
      </CardContent>
    </Card>
  );
}

/** Residual, interval-decay and feature-importance diagnostics stored with the run's metrics. */
export function NeuralDiagnostics({ run }: { run: ModelRunDetail }) {
  const res = run.metrics.residuals;
  const fi = run.metrics.feature_importance;
  const fiData = useMemo(() => {
    if (!fi) return [];
    const entries = Object.entries(fi).sort((a, b) => b[1] - a[1]).slice(0, 20);
    const total = entries.reduce((a, [, v]) => a + v, 0) || 1;
    return entries.map(([feature, v]) => ({ feature, share: v / total }));
  }, [fi]);
  if (!res && !fiData.length) return null;

  return (
    <div className="mb-6 grid gap-6 lg:grid-cols-2">
      {res ? (
        <>
          <Card>
            <CardHeader>
              <CardTitle>Error by week ahead</CardTitle>
              <CardDescription>WAPE per week of the validation window: how accuracy decays with lead time</CardDescription>
            </CardHeader>
            <CardContent>
              <SimpleBarChart
                data={res.by_week_ahead.map((w) => ({ week: `wk ${w.week}`, wape: w.wape !== null ? w.wape * 100 : null }))}
                xKey="week"
                series={[{ key: "wape", label: "WAPE %" }]}
                height={220}
              />
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>Residuals by weekday</CardTitle>
              <CardDescription>WAPE and bias (positive = over-forecast)</CardDescription>
            </CardHeader>
            <CardContent>
              <Table>
                <THead><tr><TH>Day</TH><TH className="text-right">WAPE</TH><TH className="text-right">Bias</TH><TH className="text-right">Rows</TH></tr></THead>
                <tbody>
                  {res.by_weekday.map((d) => (
                    <TR key={d.dow}>
                      <TD>{d.dow}</TD>
                      <TD className="text-right tabular-nums">{formatPct(d.wape, 1)}</TD>
                      <TD className={cn("text-right tabular-nums", Math.abs(d.bias_pct ?? 0) > 0.03 && "font-medium text-[oklch(0.5_0.12_70)]")}>
                        {d.bias_pct !== null ? `${d.bias_pct >= 0 ? "+" : ""}${formatPct(d.bias_pct, 1)}` : "—"}
                      </TD>
                      <TD className="text-right tabular-nums text-muted-foreground">{formatNumber(d.n)}</TD>
                    </TR>
                  ))}
                </tbody>
              </Table>
              {Object.keys(res.pct_error_quantiles).length ? (
                <div className="mt-4">
                  <p className="mb-2 text-xs font-medium text-muted-foreground">Percentage error distribution (non-zero actuals)</p>
                  <div className="grid grid-cols-7 gap-1 text-center text-xs">
                    {Object.entries(res.pct_error_quantiles).map(([q, v]) => (
                      <div key={q} className="rounded-md bg-muted px-1 py-1.5">
                        <p className="text-muted-foreground">{q.toUpperCase()}</p>
                        <p className="font-medium tabular-nums">{`${v >= 0 ? "+" : ""}${(v * 100).toFixed(1)}%`}</p>
                      </div>
                    ))}
                  </div>
                </div>
              ) : null}
            </CardContent>
          </Card>
        </>
      ) : null}
      {fiData.length ? (
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Feature importance</CardTitle>
            <CardDescription>Top {fiData.length} features by share of total gain</CardDescription>
          </CardHeader>
          <CardContent>
            <HorizontalBarChart data={fiData} labelKey="feature" valueKey="share" valueFormatter={(v) => formatPct(v, 1)} />
          </CardContent>
        </Card>
      ) : null}
    </div>
  );
}
