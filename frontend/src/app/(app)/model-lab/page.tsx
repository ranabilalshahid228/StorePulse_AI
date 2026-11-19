"use client";

import Link from "next/link";
import { useState, type FormEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Play, Sparkles } from "lucide-react";
import { api } from "@/lib/api";
import type { LeaderboardEntry, ModelFamilyInfo } from "@/lib/types";
import { useDatasets, useModelRuns } from "@/hooks/queries";
import { Button } from "@/components/ui/button";
import { Badge, Card, CardContent, CardDescription, CardHeader, CardTitle, Field, Input, Select, Table, TD, TH, THead, TR } from "@/components/ui/primitives";
import { EmptyState, ErrorNotice, JobProgress, PageHeader } from "@/components/shared";
import { formatDateTime, formatNumber, formatPct } from "@/lib/utils";

export default function ModelLabPage() {
  const qc = useQueryClient();
  const datasets = useDatasets();
  const ready = datasets.data?.items.filter((d) => d.status === "ready") ?? [];
  const [datasetId, setDatasetId] = useState<string>("");
  const selected = datasetId || ready[0]?.id || "";
  const runs = useModelRuns(selected || undefined);
  const families = useQuery({ queryKey: ["families"], queryFn: () => api.get<ModelFamilyInfo[]>("/model-families") });
  const leaderboard = useQuery({
    queryKey: ["leaderboard", selected],
    enabled: !!selected,
    queryFn: () => api.get<{ entries: LeaderboardEntry[] }>(`/model-runs/leaderboard?dataset_id=${selected}&metric=wape`),
    refetchInterval: 10_000,
  });

  const submit = useMutation({
    mutationFn: ({ mode, body }: { mode: "train" | "tune"; body: object }) => api.post(`/model-runs/${mode}`, body),
    onSuccess: () => void qc.invalidateQueries({ queryKey: ["model-runs"] }),
  });

  function submitForm(form: HTMLFormElement, mode: "train" | "tune") {
    const f = new FormData(form);
    submit.mutate({
      mode,
      body: {
        dataset_id: selected,
        model_family: f.get("family"),
        name: f.get("name") || null,
        horizon_days: Number(f.get("horizon")),
        history_days: f.get("history_days") ? Number(f.get("history_days")) : null,
        use_store_features: f.get("store_features") === "on",
        ...(mode === "tune" ? { n_trials: Number(f.get("trials")), metric: "wape" } : {}),
      },
    });
  }

  return (
    <>
      <PageHeader title="Model Lab" description="Train and tune forecasting models, compare them on a shared validation window, and promote the best one." />
      {!ready.length ? (
        <EmptyState title="No ready datasets" description="Upload and map a dataset first." action={<Button asChild size="sm"><Link href="/data">Go to Data</Link></Button>} />
      ) : (
        <>
          <Card className="mb-6">
            <CardHeader>
              <CardTitle>New experiment</CardTitle>
              <CardDescription>Validation uses the last N days (N = horizon) as a hold-out; lags are offset so no future data leaks.</CardDescription>
            </CardHeader>
            <CardContent>
              <form
                className="grid items-end gap-4 md:grid-cols-4 xl:grid-cols-8"
                onSubmit={(e: FormEvent<HTMLFormElement>) => {
                  e.preventDefault();
                  submitForm(e.currentTarget, "train");
                }}
              >
                <Field label="Dataset">
                  <Select value={selected} onChange={(e) => setDatasetId(e.target.value)}>
                    {ready.map((d) => <option key={d.id} value={d.id}>{d.name}</option>)}
                  </Select>
                </Field>
                <Field label="Model family">
                  <Select name="family">
                    {families.data?.map((f) => (
                      <option key={f.family} value={f.family} disabled={!f.available}>
                        {f.family}{f.supports_gpu ? " (GPU)" : ""}{f.available ? "" : " — not installed"}
                      </option>
                    ))}
                  </Select>
                </Field>
                <Field label="Run name"><Input name="name" placeholder="optional" /></Field>
                <Field label="Horizon (days)"><Input name="horizon" type="number" min={1} max={365} defaultValue={42} /></Field>
                <Field label="Optuna trials"><Input name="trials" type="number" min={1} max={500} defaultValue={30} /></Field>
                <Field label="Train on last N days" hint="Empty = full history"><Input name="history_days" type="number" min={56} max={3650} placeholder="all" /></Field>
                <label className="flex h-9 items-center gap-2 self-end text-xs text-muted-foreground">
                  <input type="checkbox" name="store_features" defaultChecked className="accent-[var(--primary)]" /> Store metadata features
                </label>
                <div className="flex gap-2">
                  <Button type="submit" disabled={submit.isPending}><Play /> Train</Button>
                  <Button
                    type="button"
                    variant="secondary"
                    disabled={submit.isPending}
                    onClick={(e) => e.currentTarget.form && submitForm(e.currentTarget.form, "tune")}
                  >
                    <Sparkles /> Tune
                  </Button>
                </div>
              </form>
              <div className="mt-3"><ErrorNotice error={submit.error} /></div>
            </CardContent>
          </Card>

          <div className="grid gap-6 lg:grid-cols-5">
            <Card className="lg:col-span-2">
              <CardHeader><CardTitle>Leaderboard</CardTitle><CardDescription>Ranked by validation WAPE (lower is better)</CardDescription></CardHeader>
              <CardContent>
                {leaderboard.data?.entries.length ? (
                  <Table>
                    <THead><tr><TH>#</TH><TH>Model</TH><TH>WAPE</TH><TH>RMSSE</TH><TH>RMSPE</TH><TH>ms/1k</TH></tr></THead>
                    <tbody>
                      {leaderboard.data.entries.map((e) => (
                        <TR key={e.model_run_id}>
                          <TD>{e.rank}</TD>
                          <TD><Link className="font-medium hover:underline" href={`/model-lab/${e.model_run_id}`}>{e.name ?? e.model_family}</Link></TD>
                          <TD>{formatPct(e.metric_value, 2)}</TD>
                          <TD>{formatNumber(e.metrics.rmsse, 3)}</TD>
                          <TD>{formatPct(e.metrics.rmspe, 2)}</TD>
                          <TD>{formatNumber(e.inference_ms, 1)}</TD>
                        </TR>
                      ))}
                    </tbody>
                  </Table>
                ) : (
                  <p className="text-sm text-muted-foreground">No finished runs yet.</p>
                )}
              </CardContent>
            </Card>
            <Card className="lg:col-span-3">
              <CardHeader><CardTitle>Runs</CardTitle></CardHeader>
              <CardContent>
                {runs.data?.items.length ? (
                  <Table>
                    <THead><tr><TH>Run</TH><TH>Type</TH><TH>Status</TH><TH>WAPE</TH><TH>Started</TH></tr></THead>
                    <tbody>
                      {runs.data.items.map((r) => (
                        <TR key={r.id}>
                          <TD><Link className="font-medium hover:underline" href={`/model-lab/${r.id}`}>{r.name ?? r.model_family}</Link></TD>
                          <TD><Badge>{r.run_type}</Badge></TD>
                          <TD><JobProgress status={r.status} pct={r.progress_pct} stage={r.stage} /></TD>
                          <TD>{formatPct(r.metrics.wape as number | undefined, 2)}</TD>
                          <TD className="text-muted-foreground">{formatDateTime(r.created_at)}</TD>
                        </TR>
                      ))}
                    </tbody>
                  </Table>
                ) : (
                  <p className="text-sm text-muted-foreground">No runs for this dataset.</p>
                )}
              </CardContent>
            </Card>
          </div>
        </>
      )}
    </>
  );
}
