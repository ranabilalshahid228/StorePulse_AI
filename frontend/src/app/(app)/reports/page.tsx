"use client";

import { useState } from "react";
import { Download, FileSpreadsheet, FileText } from "lucide-react";
import { api } from "@/lib/api";
import { useAutopilotRuns, useForecasts, useStores } from "@/hooks/queries";
import { Button } from "@/components/ui/button";
import { Badge, Card, CardContent, CardDescription, CardHeader, CardTitle, Select, Skeleton, Table, TD, TH, THead, TR } from "@/components/ui/primitives";
import { EmptyState, ErrorNotice, JobProgress, PageHeader } from "@/components/shared";
import { formatCompact, formatCurrency, formatDateTime } from "@/lib/utils";

export default function ReportsPage() {
  const forecasts = useForecasts();
  const stores = useStores();
  const [storeId, setStoreId] = useState("");
  const [error, setError] = useState<unknown>(null);
  const sid = storeId || stores.data?.items[0]?.id || "";

  const dl = (path: string, name: string) => {
    setError(null);
    api.download(path, name).catch(setError);
  };

  return (
    <>
      <PageHeader
        title="Report export center"
        description="Executive Autopilot reports (PDF with charts, Excel workbook, CSV) and raw forecast exports."
        actions={
          <Select value={sid} onChange={(e) => setStoreId(e.target.value)} className="w-56">
            {stores.data?.items.map((s) => <option key={s.id} value={s.id}>Store plan for {s.code}</option>)}
          </Select>
        }
      />
      <div className="mb-4"><ErrorNotice error={error} /></div>
      <AutopilotExports dl={dl} />
      <Card>
        <CardHeader><CardTitle>Forecast runs</CardTitle><CardDescription>Raw forecasts, and per-store plans for the store picked above</CardDescription></CardHeader>
        <CardContent>
          {forecasts.data?.items.length ? (
            <Table>
              <THead><tr><TH>Window</TH><TH>Kind</TH><TH>Status</TH><TH>Total</TH><TH>Created</TH><TH className="text-right">Downloads</TH></tr></THead>
              <tbody>
                {forecasts.data.items.map((f) => {
                  const totals = Object.values(f.summary.horizon_totals ?? {});
                  return (
                    <TR key={f.id}>
                      <TD className="font-medium">{f.summary.start_date ? `${f.summary.start_date} → ${f.summary.end_date}` : `${f.horizon_days} days`}</TD>
                      <TD>{f.kind.replace("_", " ")}</TD>
                      <TD><JobProgress status={f.status} pct={f.progress_pct} stage={f.stage} /></TD>
                      <TD>{formatCompact(totals[totals.length - 1]?.yhat)}</TD>
                      <TD className="text-muted-foreground">{formatDateTime(f.created_at)}</TD>
                      <TD className="text-right">
                        {f.status === "succeeded" ? (
                          <div className="flex justify-end gap-2">
                            <Button size="sm" variant="outline" onClick={() => dl(`/reports/forecasts/${f.id}/forecast.csv`, `forecast_${f.id}.csv`)}>
                              <Download /> Forecast CSV
                            </Button>
                            <Button size="sm" variant="outline" disabled={!sid} onClick={() => dl(`/reports/forecasts/${f.id}/store-plan.csv?store_id=${sid}`, `store_plan_${f.id}.csv`)}>
                              <Download /> Plan CSV
                            </Button>
                            <Button size="sm" disabled={!sid} onClick={() => dl(`/reports/forecasts/${f.id}/store-plan.pdf?store_id=${sid}`, `store_plan_${f.id}.pdf`)}>
                              <FileText /> Plan PDF
                            </Button>
                          </div>
                        ) : null}
                      </TD>
                    </TR>
                  );
                })}
              </tbody>
            </Table>
          ) : (
            <EmptyState title="No forecasts yet" description="Forecasts you run appear here for export." />
          )}
        </CardContent>
      </Card>
    </>
  );
}

function AutopilotExports({ dl }: { dl: (path: string, name: string) => void }) {
  const runs = useAutopilotRuns();
  const stores = useStores();
  const code = new Map((stores.data?.items ?? []).map((s) => [s.id, s.code]));
  const items = runs.data?.items ?? [];
  return (
    <Card className="mb-6">
      <CardHeader>
        <CardTitle>Store Autopilot plans</CardTitle>
        <CardDescription>PDF: KPIs, charts, actions, P&L, shift plan and model comparison. Excel: one sheet per section. CSV: daily plan.</CardDescription>
      </CardHeader>
      <CardContent>
        <ErrorNotice error={runs.error} />
        {runs.isLoading ? (
          <Skeleton className="h-24" />
        ) : items.length ? (
          <Table>
            <THead><tr><TH>Store</TH><TH>Window</TH><TH>Status</TH><TH className="text-right">Revenue P50</TH><TH className="text-right">Net profit P50</TH><TH>Created</TH><TH className="text-right">Downloads</TH></tr></THead>
            <tbody>
              {items.map((r) => {
                const stem = `autopilot_${code.get(r.store_id) ?? r.store_id}_${r.summary.window_start ?? r.id}`;
                const scenario = Boolean(r.scenario.levers?.length || r.scenario.hourly_wage || r.scenario.cogs_pct || r.scenario.promo2 != null);
                return (
                  <TR key={r.id}>
                    <TD className="font-medium">{code.get(r.store_id) ?? "—"}</TD>
                    <TD>
                      {r.summary.window_start ? `${r.summary.window_start} → ${r.summary.window_end}` : r.horizon.replace("_", " ")}
                      {scenario ? <Badge tone="info" className="ml-2">scenario</Badge> : null}
                    </TD>
                    <TD><JobProgress status={r.status} pct={r.progress_pct} stage={r.stage} /></TD>
                    <TD className="text-right tabular-nums">{formatCurrency(r.summary.revenue_p50 ?? null)}</TD>
                    <TD className="text-right tabular-nums">{formatCurrency(r.summary.net_profit_p50 ?? null)}</TD>
                    <TD className="text-muted-foreground">{formatDateTime(r.created_at)}</TD>
                    <TD className="text-right">
                      {r.status === "succeeded" ? (
                        <div className="flex justify-end gap-2">
                          <Button size="sm" onClick={() => dl(`/autopilot/runs/${r.id}/report.pdf`, `${stem}.pdf`)}><FileText /> PDF</Button>
                          <Button size="sm" variant="outline" onClick={() => dl(`/autopilot/runs/${r.id}/plan.xlsx`, `${stem}.xlsx`)}><FileSpreadsheet /> Excel</Button>
                          <Button size="sm" variant="outline" onClick={() => dl(`/autopilot/runs/${r.id}/plan.csv`, `${stem}.csv`)}><Download /> CSV</Button>
                        </div>
                      ) : null}
                    </TD>
                  </TR>
                );
              })}
            </tbody>
          </Table>
        ) : (
          <EmptyState title="No Autopilot plans yet" description="Plans you run in the Store Autopilot appear here for export." />
        )}
      </CardContent>
    </Card>
  );
}
