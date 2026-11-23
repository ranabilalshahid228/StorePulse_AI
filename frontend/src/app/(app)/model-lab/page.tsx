"use client";

import Link from "next/link";
import { useMemo, useState, type FormEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowDown, ArrowUp, Play, Sparkles } from "lucide-react";
import { api } from "@/lib/api";
import type { ModelFamilyInfo, ModelRun } from "@/lib/types";
import { useDatasets, useModelRuns, useStores } from "@/hooks/queries";
import { Button } from "@/components/ui/button";
import { Badge, Card, CardContent, CardDescription, CardHeader, CardTitle, Field, Input, Select, Skeleton, Table, TD, TH, THead, TR } from "@/components/ui/primitives";
import { EmptyState, ErrorNotice, JobProgress, PageHeader } from "@/components/shared";
import { SimpleBarChart } from "@/components/charts";
import { cn, formatDateTime, formatNumber, formatPct } from "@/lib/utils";

type SortKey = "accuracy" | "wape" | "rmspe" | "rmsse" | "rmse" | "mae" | "r2" | "fit" | "horizon";
const HIGHER_BETTER: SortKey[] = ["accuracy", "r2"];

const fitSeconds = (r: ModelRun): number | null => {
  if (typeof r.metrics.fit_seconds === "number") return r.metrics.fit_seconds;
  if (r.started_at && r.finished_at) return (new Date(r.finished_at).getTime() - new Date(r.started_at).getTime()) / 1000;
  return null;
};
const formatDuration = (s: number | null) =>
  s === null ? "—" : s < 90 ? `${s.toFixed(0)} s` : s < 5400 ? `${(s / 60).toFixed(1)} min` : `${(s / 3600).toFixed(1)} h`;

function metricOf(r: ModelRun, k: SortKey): number | null {
  const m = r.metrics as Record<string, unknown>;
  const n = (v: unknown) => (typeof v === "number" ? v : null);
  switch (k) {
    case "accuracy":
      return n(m.wape) === null ? null : 1 - (m.wape as number);
    case "fit":
      return fitSeconds(r);
    case "horizon":
      return r.horizon_days;
    default:
      return n(m[k]);
  }
}

export default function ModelLabPage() {
  const datasets = useDatasets();
  const ready = datasets.data?.items.filter((d) => d.status === "ready" && d.role === "primary") ?? [];
  const [datasetId, setDatasetId] = useState<string>("");
  const selected = datasetId || ready[0]?.id || "";
  const runs = useModelRuns(selected || undefined);

  return (
    <>
      <PageHeader
        title="Model Lab"
        description="Train or tune forecasting models, compare them on accuracy, error metrics and training time across horizons, and drill into neural diagnostics."
        actions={
          ready.length ? (
            <Select value={selected} onChange={(e) => setDatasetId(e.target.value)} className="w-64" aria-label="Dataset">
              {ready.map((d) => <option key={d.id} value={d.id}>{d.name}</option>)}
            </Select>
          ) : null
        }
      />
      {datasets.isLoading ? (
        <Skeleton className="h-48" />
      ) : !ready.length ? (
        <EmptyState title="No ready datasets" description="Upload and map a sales history first." action={<Button asChild size="sm"><Link href="/data">Go to Data</Link></Button>} />
      ) : (
        <>
          <LaunchForm datasetId={selected} />
          <ErrorNotice error={runs.error} />
          <Comparison runs={runs.data?.items ?? []} loading={runs.isLoading} />
          <RunsTable runs={runs.data?.items ?? []} />
        </>
      )}
    </>
  );
}

function LaunchForm({ datasetId }: { datasetId: string }) {
  const qc = useQueryClient();
  const stores = useStores();
  const families = useQuery({ queryKey: ["families"], queryFn: () => api.get<ModelFamilyInfo[]>("/model-families") });
  const [tune, setTune] = useState(false);
  const [family, setFamily] = useState("lightgbm");
  const submit = useMutation({
    mutationFn: (body: object) => api.post(`/model-runs/${tune ? "tune" : "train"}`, body),
    onSuccess: () => void qc.invalidateQueries({ queryKey: ["model-runs"] }),
  });

  function onSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const f = new FormData(e.currentTarget);
    const opt = (k: string) => (f.get(k) ? Number(f.get(k)) : null);
    submit.mutate({
      dataset_id: datasetId,
      model_family: f.get("family"),
      name: f.get("name") || null,
      store_id: f.get("store_id") || null,
      horizon_days: Number(f.get("horizon")),
      validation_days: opt("validation_days"),
      history_days: opt("history_days"),
      use_store_features: f.get("store_features") === "on",
      ...(tune
        ? { n_trials: Number(f.get("trials")), metric: f.get("metric"), timeout_minutes: opt("timeout") }
        : {}),
    });
  }

  return (
    <Card className="mb-6">
      <CardHeader>
        <CardTitle>New experiment</CardTitle>
        <CardDescription>
          The last <em>validation days</em> (default = horizon) are held out; target features are lagged by the horizon so nothing leaks.
          Tuning scores Optuna trials on an earlier split of the training period, so the reported metrics stay out-of-sample.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form className="flex flex-col gap-4" onSubmit={onSubmit}>
          <div className="grid gap-4 md:grid-cols-3 xl:grid-cols-6">
            <Field label="Model family">
              <Select name="family" value={family} onChange={(e) => setFamily(e.target.value)}>
                {families.data?.map((f) => (
                  <option key={f.family} value={f.family} disabled={!f.available}>
                    {f.family === "arima" || f.family === "sarima" ? f.family.toUpperCase() : f.family}{f.supports_gpu ? " (GPU)" : ""}{f.available ? "" : " — not installed"}
                  </option>
                ))}
              </Select>
            </Field>
            <Field label="Run name"><Input name="name" placeholder="optional" /></Field>
            <Field label="Horizon (days)" hint="Autopilot month plans need ≥ 30"><Input name="horizon" type="number" min={1} max={365} defaultValue={42} required /></Field>
            <Field label="Validation days" hint="Empty = horizon"><Input name="validation_days" type="number" min={7} max={365} placeholder="= horizon" /></Field>
            <Field label="Train on last N days" hint="Empty = full history"><Input name="history_days" type="number" min={56} max={3650} placeholder="all" /></Field>
            <Field label="Stores" hint="One store = single-store model">
              <Select name="store_id" defaultValue="">
                <option value="">All stores</option>
                {stores.data?.items.map((s) => <option key={s.id} value={s.id}>Only {s.code}</option>)}
              </Select>
            </Field>
          </div>
          <div className="flex flex-wrap items-center gap-6">
            <label className="flex items-center gap-2 text-sm">
              <input type="checkbox" name="store_features" defaultChecked className="accent-[var(--primary)]" /> Store metadata features
            </label>
            <label className="flex items-center gap-2 text-sm">
              <input type="checkbox" checked={tune} onChange={(e) => setTune(e.target.checked)} className="accent-[var(--primary)]" /> Optuna tuning
            </label>
            {tune ? (
              <div className="flex flex-wrap items-end gap-3">
                <Field label="Trials"><Input name="trials" type="number" min={1} max={500} defaultValue={25} className="w-24" /></Field>
                <Field label="Optimize">
                  <Select name="metric" defaultValue="wape" className="w-28">
                    {["wape", "rmspe", "rmsse", "rmse", "mae"].map((m) => <option key={m}>{m}</option>)}
                  </Select>
                </Field>
                <Field label="Time limit (min)"><Input name="timeout" type="number" min={1} placeholder="none" className="w-28" /></Field>
              </div>
            ) : null}
            <Button type="submit" className="ml-auto" disabled={submit.isPending || !datasetId}>
              {tune ? <Sparkles /> : <Play />} {submit.isPending ? "Queuing…" : tune ? "Launch tuning" : "Launch training"}
            </Button>
          </div>
          <ErrorNotice error={submit.error} />
          {submit.isSuccess ? <p className="text-xs text-success">Queued — progress appears in the runs table below.</p> : null}
        </form>
      </CardContent>
    </Card>
  );
}

function Comparison({ runs, loading }: { runs: ModelRun[]; loading: boolean }) {
  const [sort, setSort] = useState<{ key: SortKey; desc: boolean }>({ key: "accuracy", desc: true });
  const done = runs.filter((r) => r.status === "succeeded" && typeof r.metrics.wape === "number");
  const sorted = useMemo(() => {
    const v = (r: ModelRun) => metricOf(r, sort.key) ?? (sort.desc ? -Infinity : Infinity);
    return [...done].sort((a, b) => (sort.desc ? v(b) - v(a) : v(a) - v(b)));
  }, [done, sort]);

  // Best accuracy per (horizon, family) for the horizon comparison chart.
  const byHorizon = useMemo(() => {
    const fams = [...new Set(done.map((r) => r.model_family))];
    const rows = new Map<number, Record<string, number | string>>();
    for (const r of done) {
      const row = rows.get(r.horizon_days) ?? { horizon: `${r.horizon_days} d` };
      const acc = 1 - (r.metrics.wape as number);
      if (!(r.model_family in row) || acc > (row[r.model_family] as number)) row[r.model_family] = Math.round(acc * 10000) / 100;
      rows.set(r.horizon_days, row);
    }
    return { fams, data: [...rows.entries()].sort((a, b) => a[0] - b[0]).map(([, v]) => v) };
  }, [done]);

  if (loading) return <Skeleton className="mb-6 h-64" />;
  if (!done.length) {
    return (
      <Card className="mb-6">
        <CardHeader><CardTitle>Model comparison</CardTitle></CardHeader>
        <CardContent><p className="text-sm text-muted-foreground">Finished runs appear here with accuracy, error metrics and training time.</p></CardContent>
      </Card>
    );
  }

  const head = (key: SortKey, label: string) => (
    <TH className="cursor-pointer select-none text-right" onClick={() => setSort((s) => ({ key, desc: s.key === key ? !s.desc : HIGHER_BETTER.includes(key) }))}>
      <span className="inline-flex items-center gap-1">
        {label}
        {sort.key === key ? sort.desc ? <ArrowDown className="size-3" /> : <ArrowUp className="size-3" /> : null}
      </span>
    </TH>
  );
  const best = sorted[0];

  return (
    <div className="mb-6 grid gap-6 xl:grid-cols-3">
      <Card className="xl:col-span-2">
        <CardHeader>
          <CardTitle>Model comparison</CardTitle>
          <CardDescription>Accuracy = 1 − WAPE on each run&apos;s validation window. Click a column to sort.</CardDescription>
        </CardHeader>
        <CardContent className="overflow-x-auto">
          <Table>
            <THead>
              <tr>
                <TH>Model</TH>
                {head("horizon", "Horizon")}
                {head("accuracy", "Accuracy")}
                {head("wape", "WAPE")}
                {head("rmspe", "RMSPE")}
                {head("rmsse", "RMSSE")}
                {head("rmse", "RMSE")}
                {head("mae", "MAE")}
                {head("r2", "R²")}
                {head("fit", "Train time")}
              </tr>
            </THead>
            <tbody>
              {sorted.map((r) => {
                const m = r.metrics as Record<string, number | undefined>;
                return (
                  <TR key={r.id} className={cn(r.id === best?.id && "bg-muted/60")}>
                    <TD>
                      <Link className="font-medium hover:underline" href={`/model-lab/${r.id}`}>{r.name ?? r.model_family}</Link>
                      <span className="block text-xs text-muted-foreground">
                        {r.model_family} · {r.run_type}{r.store_id ? " · single store" : ""}{r.device === "cuda" ? " · GPU" : ""}
                      </span>
                    </TD>
                    <TD className="text-right tabular-nums">{r.horizon_days} d</TD>
                    <TD className="text-right font-medium tabular-nums">{formatPct(metricOf(r, "accuracy"), 2)}</TD>
                    <TD className="text-right tabular-nums">{formatNumber(m.wape, 4)}</TD>
                    <TD className="text-right tabular-nums">{formatNumber(m.rmspe, 4)}</TD>
                    <TD className="text-right tabular-nums">{formatNumber(m.rmsse, 4)}</TD>
                    <TD className="text-right tabular-nums">{formatNumber(m.rmse, 1)}</TD>
                    <TD className="text-right tabular-nums">{formatNumber(m.mae, 1)}</TD>
                    <TD className="text-right tabular-nums">{formatNumber(m.r2, 3)}</TD>
                    <TD className="text-right tabular-nums">{formatDuration(fitSeconds(r))}</TD>
                  </TR>
                );
              })}
            </tbody>
          </Table>
        </CardContent>
      </Card>
      <Card>
        <CardHeader><CardTitle>Accuracy by horizon</CardTitle><CardDescription>Best run per model family (%)</CardDescription></CardHeader>
        <CardContent>
          <SimpleBarChart data={byHorizon.data} xKey="horizon" series={byHorizon.fams.map((f) => ({ key: f, label: f }))} height={280} />
          <p className="mt-2 text-xs text-muted-foreground">Compare horizons on the same validation window (set validation days) for a fair read.</p>
        </CardContent>
      </Card>
    </div>
  );
}

function RunsTable({ runs }: { runs: ModelRun[] }) {
  return (
    <Card>
      <CardHeader><CardTitle>All runs</CardTitle></CardHeader>
      <CardContent>
        {runs.length ? (
          <Table>
            <THead><tr><TH>Run</TH><TH>Type</TH><TH>Status</TH><TH>Accuracy</TH><TH>Horizon</TH><TH>Started</TH></tr></THead>
            <tbody>
              {runs.map((r) => (
                <TR key={r.id}>
                  <TD><Link className="font-medium hover:underline" href={`/model-lab/${r.id}`}>{r.name ?? r.model_family}</Link></TD>
                  <TD><Badge>{r.run_type}</Badge></TD>
                  <TD><JobProgress status={r.status} pct={r.progress_pct} stage={r.stage} /></TD>
                  <TD className="tabular-nums">{formatPct(metricOf(r, "accuracy"), 2)}</TD>
                  <TD>{r.horizon_days} d</TD>
                  <TD className="text-muted-foreground">{formatDateTime(r.created_at)}</TD>
                </TR>
              ))}
            </tbody>
          </Table>
        ) : (
          <p className="text-sm text-muted-foreground">No runs for this dataset yet.</p>
        )}
      </CardContent>
    </Card>
  );
}
