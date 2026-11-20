"use client";

import Link from "next/link";
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { PnLReport, WorkforcePlan } from "@/lib/types";
import { useForecasts, useStores } from "@/hooks/queries";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, Field, Select } from "@/components/ui/primitives";
import { EmptyState, ErrorNotice, Kpi } from "@/components/shared";
import { SimpleBarChart, TimeSeriesChart } from "@/components/charts";
import { formatCurrency, formatNumber, formatPct } from "@/lib/utils";

/** Forecast-run based P&L / workforce (the pre-Autopilot view, kept for forecast runs). */
export function ForecastPnl() {
  const forecasts = useForecasts();
  const stores = useStores();
  const done = forecasts.data?.items.filter((f) => f.status === "succeeded") ?? [];
  const [forecastId, setForecastId] = useState("");
  const [storeId, setStoreId] = useState("");
  const fid = forecastId || done[0]?.id || "";
  const sid = storeId || stores.data?.items[0]?.id || "";
  const enabled = !!fid && !!sid;

  const pnl = useQuery({ queryKey: ["pnl", fid, sid], enabled, queryFn: () => api.get<PnLReport>(`/forecasts/${fid}/pnl?store_id=${sid}`) });
  const wf = useQuery({ queryKey: ["workforce", fid, sid], enabled, queryFn: () => api.get<WorkforcePlan>(`/forecasts/${fid}/workforce?store_id=${sid}`) });
  const cur = pnl.data?.currency ?? "EUR";

  return (
    <>

      {!done.length ? (
        <EmptyState title="No completed forecasts" description="Run a forecast from a trained model in Model Lab." action={<Button asChild size="sm"><Link href="/model-lab">Model Lab</Link></Button>} />
      ) : (
        <>
          <Card className="mb-6">
            <CardContent className="grid gap-4 p-5 md:grid-cols-2">
              <Field label="Forecast">
                <Select value={fid} onChange={(e) => setForecastId(e.target.value)}>
                  {done.map((f) => <option key={f.id} value={f.id}>{f.summary.start_date} → {f.summary.end_date} · {f.kind}</option>)}
                </Select>
              </Field>
              <Field label="Store">
                <Select value={sid} onChange={(e) => setStoreId(e.target.value)}>
                  {stores.data?.items.map((s) => (
                    <option key={s.id} value={s.id}>{s.code} — {s.name}{s.has_financial_config ? "" : " (no config)"}</option>
                  ))}
                </Select>
              </Field>
            </CardContent>
          </Card>
          <ErrorNotice error={pnl.error ?? wf.error} />
          {(pnl.error || wf.error) ? (
            <p className="mt-2 text-xs text-muted-foreground">Configure unit economics on the <Link className="underline" href="/stores">Stores</Link> page.</p>
          ) : null}
          {pnl.data && wf.data ? (
            <>
              <div className="mb-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                <Kpi label="Net revenue" value={formatCurrency(pnl.data.totals.net_revenue, cur)} sub={`promo discounts ${formatCurrency(pnl.data.totals.promo_discount, cur)}`} />
                <Kpi label="Gross profit" value={formatCurrency(pnl.data.totals.gross_profit, cur)} sub={`margin ${formatPct(pnl.data.totals.realized_gross_margin_pct)}`} />
                <Kpi label="Operating profit" value={formatCurrency(pnl.data.totals.operating_profit, cur)} sub={`margin ${formatPct(pnl.data.totals.operating_margin_pct)}`} />
                <Kpi label="Labour" value={formatCurrency(wf.data.total_labor_cost, cur)} sub={`${formatNumber(wf.data.total_scheduled_hours)} h · peak ${wf.data.peak_staff_per_shift}/shift`} />
              </div>
              <div className="mb-6 grid gap-6 lg:grid-cols-2">
                <Card>
                  <CardHeader><CardTitle>Daily P&L</CardTitle></CardHeader>
                  <CardContent>
                    <TimeSeriesChart
                      data={pnl.data.days}
                      series={[
                        { key: "net_revenue", label: "Net revenue" },
                        { key: "gross_profit", label: "Gross profit" },
                        { key: "operating_profit", label: "Operating profit" },
                      ]}
                    />
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader><CardTitle>Staff per shift</CardTitle><CardDescription>From forecast footfall and productivity</CardDescription></CardHeader>
                  <CardContent>
                    <SimpleBarChart data={wf.data.days} xKey="ds" series={[{ key: "staff_per_shift", label: "Staff / shift" }]} height={280} />
                  </CardContent>
                </Card>
              </div>
              <Card>
                <CardHeader><CardTitle>Assumptions</CardTitle></CardHeader>
                <CardContent>
                  <ul className="list-disc space-y-1 pl-5 text-sm text-muted-foreground">
                    {[...pnl.data.assumptions, ...wf.data.assumptions].map((a) => <li key={a}>{a}</li>)}
                  </ul>
                </CardContent>
              </Card>
            </>
          ) : null}
        </>
      )}
    </>
  );
}
