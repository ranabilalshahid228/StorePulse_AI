"use client";

import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { ForecastPoint } from "@/lib/types";
import { useDatasets, useForecasts, useModelRuns, useStores } from "@/hooks/queries";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, Table, TD, TH, THead, TR } from "@/components/ui/primitives";
import { EmptyState, JobProgress, Kpi, PageHeader } from "@/components/shared";
import { ForecastBandChart } from "@/components/charts";
import { formatCompact, formatDateTime, formatPct } from "@/lib/utils";

export default function CommandCenter() {
  const stores = useStores();
  const datasets = useDatasets();
  const runs = useModelRuns();
  const forecasts = useForecasts();

  const latest = forecasts.data?.items.find((f) => f.status === "succeeded");
  const series = useQuery({
    queryKey: ["forecast-series", latest?.id],
    enabled: !!latest,
    queryFn: () => api.get<{ points: ForecastPoint[] }>(`/forecasts/${latest!.id}/series`),
  });

  const bestRun = runs.data?.items
    .filter((r) => r.status === "succeeded" && typeof r.metrics.wape === "number")
    .sort((a, b) => (a.metrics.wape as number) - (b.metrics.wape as number))[0];
  type Job = { id: string; status: string; progress_pct: number; stage: string | null; created_at: string };
  const job = (j: Job, jobType: string, label: string) => ({ ...j, jobType, label });
  const active = [
    ...(datasets.data?.items ?? []).filter((d) => ["queued", "processing"].includes(d.status)).map((d) => job(d, "Ingestion", d.name)),
    ...(runs.data?.items ?? [])
      .filter((r) => ["queued", "running"].includes(r.status))
      .map((r) => job(r, r.run_type === "tune" ? "Tuning" : "Training", r.name ?? r.model_family)),
    ...(forecasts.data?.items ?? [])
      .filter((f) => ["queued", "running"].includes(f.status))
      .map((f) => job(f, "Forecast", `${f.horizon_days}-day ${f.kind.replace("_", " ")}`)),
  ];
  const totals = latest?.summary.horizon_totals ?? {};
  const totalKeys = Object.keys(totals);
  const fullHorizon = totalKeys.length ? totals[totalKeys[totalKeys.length - 1]!] : undefined;

  const chartData = (series.data?.points ?? []).map((p) => ({
    ds: p.ds,
    yhat: p.yhat,
    baseline_yhat: p.baseline_yhat,
    band: p.yhat_lower !== null && p.yhat_upper !== null ? [p.yhat_lower, p.yhat_upper] : null,
  }));

  return (
    <>
      <PageHeader title="Command Center" description="Live view of forecasts, model quality and background jobs across your stores." />
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Kpi label="Stores" value={stores.data?.total ?? "—"} sub={`${datasets.data?.total ?? 0} datasets`} />
        <Kpi
          label="Forecast demand (full horizon)"
          value={fullHorizon ? formatCompact(fullHorizon.yhat) : "—"}
          sub={latest ? `${latest.summary.start_date} → ${latest.summary.end_date}` : "No forecast yet"}
        />
        <Kpi label="Best validation WAPE" value={bestRun ? formatPct(bestRun.metrics.wape as number) : "—"} sub={bestRun ? bestRun.model_family : "Train a model in Model Lab"} />
        <Kpi label="Active jobs" value={active.length} sub="ingestion · training · forecasting" />
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Latest forecast</CardTitle>
            <CardDescription>Aggregated across all series, with the empirical 90% interval from validation.</CardDescription>
          </CardHeader>
          <CardContent>
            {latest ? (
              <ForecastBandChart data={chartData} />
            ) : (
              <EmptyState
                title="No forecasts yet"
                description="Upload a dataset, train a model, then run a forecast."
                action={<Button asChild size="sm"><Link href="/data">Upload data</Link></Button>}
              />
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Top stores by forecast</CardTitle>
          </CardHeader>
          <CardContent>
            {latest?.summary.top_stores?.length ? (
              <ul className="flex flex-col gap-2 text-sm">
                {latest.summary.top_stores.map((s) => (
                  <li key={s.store_code} className="flex justify-between">
                    <span className="text-muted-foreground">Store {s.store_code}</span>
                    <span className="font-medium">{formatCompact(s.yhat)}</span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-sm text-muted-foreground">Nothing to rank yet.</p>
            )}
          </CardContent>
        </Card>
      </div>

      <Card className="mt-6">
        <CardHeader>
          <CardTitle>Background jobs</CardTitle>
        </CardHeader>
        <CardContent>
          {active.length ? (
            <Table>
              <THead>
                <tr><TH>Type</TH><TH>Name</TH><TH>Progress</TH><TH>Submitted</TH></tr>
              </THead>
              <tbody>
                {active.map((j) => (
                  <TR key={j.id}>
                    <TD>{j.jobType}</TD>
                    <TD className="font-medium">{j.label}</TD>
                    <TD><JobProgress status={j.status} pct={j.progress_pct} stage={j.stage} /></TD>
                    <TD className="text-muted-foreground">{formatDateTime(j.created_at)}</TD>
                  </TR>
                ))}
              </tbody>
            </Table>
          ) : (
            <p className="text-sm text-muted-foreground">No jobs running.</p>
          )}
        </CardContent>
      </Card>
    </>
  );
}
