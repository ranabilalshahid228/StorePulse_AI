"use client";

import { use, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api, isActive } from "@/lib/api";
import type { ModelRunDetail } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, Field, Input, Skeleton, Table, TD, TH, THead, TR } from "@/components/ui/primitives";
import { ErrorNotice, JobProgress, Kpi, PageHeader } from "@/components/shared";
import { TimeSeriesChart } from "@/components/charts";
import { formatNumber, formatPct } from "@/lib/utils";

export default function RunPage({ params }: { params: Promise<{ runId: string }> }) {
  const { runId } = use(params);
  const router = useRouter();
  const run = useQuery({
    queryKey: ["model-run", runId],
    queryFn: () => api.get<ModelRunDetail>(`/model-runs/${runId}`),
    refetchInterval: (q) => (q.state.data && isActive(q.state.data.status) ? 2000 : false),
  });
  const forecast = useMutation({
    mutationFn: (horizon: number) => api.post<{ id: string }>("/forecasts", { model_run_id: runId, horizon_days: horizon }),
    onSuccess: () => router.push("/reports"),
  });

  const r = run.data;
  if (run.error) return <ErrorNotice error={run.error} />;
  if (!r) return <Skeleton className="h-64" />;

  // Logs may interleave several models (quantile boosters, ensemble members): plot the first one.
  const tag = (l: Record<string, unknown>) => String(l.member ?? l.model ?? "");
  const firstTag = r.epoch_logs.length ? tag(r.epoch_logs[0]!) : "";
  const curve = r.epoch_logs.filter((l) => tag(l) === firstTag);
  const curveKeys = curve.length
    ? Object.keys(curve[0]!).filter((k) => !["epoch", "seconds"].includes(k) && typeof curve[0]![k] === "number").slice(0, 3)
    : [];
  const m = r.metrics as Record<string, number | undefined>;
  const interval = r.metrics.interval;
  const ensemble = r.metrics.ensemble;

  return (
    <>
      <PageHeader
        title={r.name ?? `${r.model_family} ${r.run_type}`}
        description={`${r.model_family} · horizon ${r.horizon_days}d · validation ${r.validation_days}d · device ${r.device ?? "—"}`}
        actions={<JobProgress status={r.status} pct={r.progress_pct} stage={r.stage} />}
      />
      {r.error ? <div className="mb-6"><ErrorNotice error={new Error(r.error)} /></div> : null}
      <div className="mb-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4 xl:grid-cols-7">
        <Kpi label="WAPE" value={formatPct(m.wape, 2)} />
        <Kpi label="RMSSE" value={formatNumber(m.rmsse, 3)} sub="vs. naive, per series" />
        <Kpi label="RMSE" value={formatNumber(m.rmse, 1)} />
        <Kpi label="MAE" value={formatNumber(m.mae, 1)} />
        <Kpi label="R²" value={formatNumber(m.r2, 3)} />
        <Kpi label="Bias" value={formatPct(m.bias_pct, 2)} />
        <Kpi label="Inference" value={r.inference_ms ? `${r.inference_ms.toFixed(1)} ms` : "—"} sub="per 1,000 rows" />
      </div>

      {interval || ensemble ? (
        <div className="mb-6 grid gap-6 lg:grid-cols-2">
          {interval ? (
            <Card>
              <CardHeader>
                <CardTitle>Prediction interval</CardTitle>
                <CardDescription>
                  {interval.method === "quantile_model"
                    ? "Learned P10 / P50 / P90 quantile models (pinball loss: lower is better)."
                    : "Empirical P10–P90 band from validation residuals (coverage is in-sample)."}
                </CardDescription>
              </CardHeader>
              <CardContent className="grid grid-cols-2 gap-3 text-sm">
                <span className="text-muted-foreground">P10–P90 coverage</span>
                <span className="text-right font-medium">{formatPct(interval.coverage)} <span className="text-xs text-muted-foreground">(target 80%)</span></span>
                <span className="text-muted-foreground">Mean band width</span>
                <span className="text-right">{formatNumber(interval.mean_width, 2)}</span>
                {interval.pinball_p50 !== undefined ? (
                  <>
                    <span className="text-muted-foreground">Pinball P10 / P50 / P90</span>
                    <span className="text-right">
                      {[interval.pinball_p10, interval.pinball_p50, interval.pinball_p90].map((v) => formatNumber(v, 3)).join(" / ")}
                    </span>
                  </>
                ) : null}
              </CardContent>
            </Card>
          ) : null}
          {ensemble ? (
            <Card>
              <CardHeader>
                <CardTitle>Ensemble comparison</CardTitle>
                <CardDescription>
                  Weights fitted on the {ensemble.weights_fitted_on}; all rows below scored on the {ensemble.compared_on} ({formatNumber(ensemble.holdout_rows)} rows).
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Table>
                  <THead><tr><TH>Model</TH><TH>Weight</TH><TH>WAPE</TH><TH>RMSE</TH><TH>Bias</TH></tr></THead>
                  <tbody>
                    {Object.entries(ensemble.models).map(([name, mm]) => (
                      <TR key={name} className={name === "fitted_blend" ? "font-medium" : ""}>
                        <TD>{name.replaceAll("_", " ")}</TD>
                        <TD>{ensemble.weights[name] !== undefined ? formatPct(ensemble.weights[name], 0) : "—"}</TD>
                        <TD>{formatPct(mm.wape, 2)}</TD>
                        <TD>{formatNumber(mm.rmse, 2)}</TD>
                        <TD>{formatPct(mm.bias_pct, 1)}</TD>
                      </TR>
                    ))}
                  </tbody>
                </Table>
              </CardContent>
            </Card>
          ) : null}
        </div>
      ) : null}

      {r.status === "succeeded" ? (
        <Card className="mb-6">
          <CardHeader><CardTitle>Run a forecast</CardTitle><CardDescription>Batch-forecast every series in the dataset up to the trained horizon.</CardDescription></CardHeader>
          <CardContent>
            <form
              className="flex items-end gap-3"
              onSubmit={(e) => {
                e.preventDefault();
                forecast.mutate(Number(new FormData(e.currentTarget).get("h")));
              }}
            >
              <Field label="Horizon (days)"><Input name="h" type="number" min={1} max={r.horizon_days} defaultValue={r.horizon_days} /></Field>
              <Button type="submit" disabled={forecast.isPending}>Forecast</Button>
            </form>
            <div className="mt-3"><ErrorNotice error={forecast.error} /></div>
          </CardContent>
        </Card>
      ) : null}

      <div className="mb-6 grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader><CardTitle>Learning curve</CardTitle></CardHeader>
          <CardContent>
            {curveKeys.length ? (
              <TimeSeriesChart data={r.epoch_logs.map((l) => ({ ...l, ds: l.epoch }))} series={curveKeys.map((k) => ({ key: k, label: k }))} />
            ) : (
              <p className="text-sm text-muted-foreground">No epoch logs yet.</p>
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle>Hyperparameters</CardTitle></CardHeader>
          <CardContent>
            <pre className="max-h-72 overflow-auto rounded-md bg-muted p-3 text-xs">{JSON.stringify(r.hyperparameters, null, 2)}</pre>
          </CardContent>
        </Card>
      </div>

      {r.plot_paths.length ? (
        <div className="mb-6 grid gap-6 lg:grid-cols-2">
          {r.plot_paths.map((p) => {
            const name = p.split("/").pop()!.replace(".png", "");
            return <PlotCard key={p} runId={r.id} name={name} />;
          })}
        </div>
      ) : null}

      {r.trials.length ? (
        <Card>
          <CardHeader><CardTitle>Optuna trials</CardTitle></CardHeader>
          <CardContent>
            <Table>
              <THead><tr><TH>#</TH><TH>Objective</TH><TH>Params</TH></tr></THead>
              <tbody>
                {[...r.trials].sort((a, b) => a.value - b.value).map((t) => (
                  <TR key={t.number}>
                    <TD>{t.number}</TD>
                    <TD>{t.value.toFixed(4)}</TD>
                    <TD className="max-w-xl truncate font-mono text-xs text-muted-foreground">{JSON.stringify(t.params)}</TD>
                  </TR>
                ))}
              </tbody>
            </Table>
          </CardContent>
        </Card>
      ) : null}
    </>
  );
}

function PlotCard({ runId, name }: { runId: string; name: string }) {
  const [src, setSrc] = useState<string | null>(null);
  useEffect(() => {
    let url: string | null = null;
    api.blobUrl(`/model-runs/${runId}/plots/${name}`).then((u) => setSrc((url = u))).catch(() => setSrc(null));
    return () => {
      if (url) URL.revokeObjectURL(url);
    };
  }, [runId, name]);
  return (
    <Card>
      <CardHeader><CardTitle>{name.replaceAll("_", " ")}</CardTitle></CardHeader>
      <CardContent>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        {src ? <img src={src} alt={name} className="w-full rounded-md" /> : <Skeleton className="h-48" />}
      </CardContent>
    </Card>
  );
}
