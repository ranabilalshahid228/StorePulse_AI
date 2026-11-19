"use client";

import { useState } from "react";
import { Download, FileText } from "lucide-react";
import { api } from "@/lib/api";
import { useForecasts, useStores } from "@/hooks/queries";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, Select, Table, TD, TH, THead, TR } from "@/components/ui/primitives";
import { EmptyState, ErrorNotice, JobProgress, PageHeader } from "@/components/shared";
import { formatCompact, formatDateTime } from "@/lib/utils";

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
        title="Reports"
        description="Download forecasts as CSV, or per-store plans (P&L + staffing) as CSV and PDF."
        actions={
          <Select value={sid} onChange={(e) => setStoreId(e.target.value)} className="w-56">
            {stores.data?.items.map((s) => <option key={s.id} value={s.id}>Store plan for {s.code}</option>)}
          </Select>
        }
      />
      <div className="mb-4"><ErrorNotice error={error} /></div>
      <Card>
        <CardHeader><CardTitle>Forecast runs</CardTitle></CardHeader>
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
