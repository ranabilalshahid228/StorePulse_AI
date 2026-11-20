"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, Download, FileSpreadsheet, FileText, Plus, Rocket, Trash2 } from "lucide-react";
import { api } from "@/lib/api";
import type {
  AutopilotRequest,
  AutopilotResult,
  AutopilotRun,
  HorizonKind,
  LeverKind,
  Priority,
  ScenarioLever,
} from "@/lib/types";
import { useAutopilotRun, useAutopilotRuns, useModelRuns, useStores } from "@/hooks/queries";
import { Button } from "@/components/ui/button";
import {
  Badge,
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  Field,
  Input,
  Select,
  Skeleton,
  Table,
  TD,
  TH,
  THead,
  TR,
} from "@/components/ui/primitives";
import { EmptyState, ErrorNotice, JobProgress, Kpi, PageHeader } from "@/components/shared";
import { RevenuePlanChart, SimpleBarChart } from "@/components/charts";
import { formatCurrency, formatDateTime, formatNumber, formatPct } from "@/lib/utils";

const HORIZONS: { value: HorizonKind; label: string }[] = [
  { value: "next_day", label: "Next day" },
  { value: "week", label: "Next 7 days" },
  { value: "month", label: "Next 30 days" },
  { value: "custom", label: "Custom range" },
];
const LEVER_KINDS: LeverKind[] = ["promo", "holiday", "closure", "local_event", "competitor", "price", "custom"];
const PRIORITY_TONE: Record<Priority, "danger" | "warning" | "info" | "neutral"> = {
  critical: "danger",
  high: "danger",
  medium: "warning",
  low: "info",
};

type Tri = "" | "on" | "off";
const tri = (v: Tri): boolean | null => (v === "" ? null : v === "on");

interface LeverDraft {
  name: string;
  kind: LeverKind;
  start_date: string;
  end_date: string;
  promo: Tri;
  school_holiday: Tri;
  open: Tri;
  state_holiday: string;
  demand_pct: string;
  price_change_pct: string;
}

const emptyLever = (): LeverDraft => ({
  name: "",
  kind: "promo",
  start_date: "",
  end_date: "",
  promo: "",
  school_holiday: "",
  open: "",
  state_holiday: "",
  demand_pct: "",
  price_change_pct: "",
});

function toLever(d: LeverDraft): ScenarioLever {
  const pct = (v: string) => (v.trim() === "" ? null : Number(v) / 100);
  return {
    name: d.name || d.kind,
    kind: d.kind,
    start_date: d.start_date,
    end_date: d.end_date || d.start_date,
    promo: tri(d.promo),
    school_holiday: tri(d.school_holiday),
    open: tri(d.open),
    state_holiday: d.state_holiday || null,
    demand_pct: pct(d.demand_pct),
    price_change_pct: pct(d.price_change_pct),
  };
}

export default function AutopilotPage() {
  const qc = useQueryClient();
  const stores = useStores();
  const models = useModelRuns();
  const [storeId, setStoreId] = useState("");
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const sid = storeId || stores.data?.items.find((s) => s.has_financial_config)?.id || stores.data?.items[0]?.id || "";
  const runs = useAutopilotRuns(sid || undefined);
  const runId = selectedRun ?? runs.data?.items[0]?.id ?? null;

  const create = useMutation({
    mutationFn: (body: AutopilotRequest) => api.post<{ id: string }>("/autopilot/runs", body),
    onSuccess: (r) => {
      setSelectedRun(r.id);
      void qc.invalidateQueries({ queryKey: ["autopilot-runs"] });
    },
  });

  const loading = stores.isLoading || models.isLoading;
  const finishedModels = models.data?.items.filter((m) => m.status === "succeeded") ?? [];

  return (
    <>
      <PageHeader
        title="Store Autopilot"
        description="Plan a store's next day, week, month or any range: forecast with P10–P90 bands, footfall, P&L, shift plan and prioritized actions — with what-if scenarios."
      />
      {loading ? (
        <div className="grid gap-6 lg:grid-cols-3">
          <Skeleton className="h-96" />
          <Skeleton className="h-96 lg:col-span-2" />
        </div>
      ) : !finishedModels.length ? (
        <EmptyState
          title="No trained models yet"
          description="Train a model on your sales history in Model Lab, then plan stores here."
          action={<Button asChild size="sm"><Link href="/model-lab">Open Model Lab</Link></Button>}
        />
      ) : !stores.data?.items.length ? (
        <EmptyState title="No stores yet" description="Stores appear after a dataset with a store column is processed." />
      ) : (
        <div className="grid gap-6 xl:grid-cols-3">
          <div className="flex flex-col gap-6">
            <PlanForm
              storeId={sid}
              onStore={(id) => {
                setStoreId(id);
                setSelectedRun(null);
              }}
              onSubmit={(body) => create.mutate(body)}
              pending={create.isPending}
              error={create.error}
            />
            <RunList runs={runs.data?.items ?? []} loading={runs.isLoading} error={runs.error} active={runId} onSelect={setSelectedRun} />
          </div>
          <div className="xl:col-span-2">
            {runId ? <RunDetail id={runId} /> : <EmptyState title="No plans for this store yet" description="Configure a horizon and optional scenario, then run the Autopilot." />}
          </div>
        </div>
      )}
    </>
  );
}

function PlanForm({
  storeId,
  onStore,
  onSubmit,
  pending,
  error,
}: {
  storeId: string;
  onStore: (id: string) => void;
  onSubmit: (body: AutopilotRequest) => void;
  pending: boolean;
  error: unknown;
}) {
  const stores = useStores();
  const models = useModelRuns();
  const store = stores.data?.items.find((s) => s.id === storeId);
  const usable = (models.data?.items ?? []).filter(
    (m) => m.status === "succeeded" && (!m.store_id || m.store_id === storeId),
  );
  const [modelId, setModelId] = useState("");
  const mid = modelId || usable[0]?.id || "";
  const model = usable.find((m) => m.id === mid);
  const features = new Set(model?.feature_columns ?? []);
  const [horizon, setHorizon] = useState<HorizonKind>("week");
  const [start, setStart] = useState("");
  const [end, setEnd] = useState("");
  const [noPromoDefault, setNoPromoDefault] = useState(false);
  const [levers, setLevers] = useState<LeverDraft[]>([]);
  const [promo2, setPromo2] = useState<Tri>("");
  const [promo2Interval, setPromo2Interval] = useState("");
  const [cogs, setCogs] = useState("");
  const [elasticity, setElasticity] = useState("");

  const update = (i: number, patch: Partial<LeverDraft>) =>
    setLevers((ls) => ls.map((l, j) => (j === i ? { ...l, ...patch } : l)));

  function submit() {
    onSubmit({
      store_id: storeId,
      model_run_id: mid,
      horizon,
      start_date: horizon === "custom" ? start : null,
      end_date: horizon === "custom" ? end : null,
      scenario: {
        levers: levers.filter((l) => l.start_date).map(toLever),
        promo2: tri(promo2),
        promo2_interval: promo2Interval || null,
        covariate_defaults: noPromoDefault ? { Promo: 0 } : {},
        cogs_pct: cogs.trim() === "" ? null : Number(cogs) / 100,
        price_elasticity: elasticity.trim() === "" ? null : Number(elasticity),
      },
    });
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Plan</CardTitle>
        <CardDescription>Horizons start the day after the latest data and must fit the model&apos;s trained horizon.</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <Field label="Store">
          <Select value={storeId} onChange={(e) => onStore(e.target.value)}>
            {stores.data?.items.map((s) => (
              <option key={s.id} value={s.id}>
                {s.code} — {s.name}
                {s.has_financial_config ? "" : " (no config)"}
              </option>
            ))}
          </Select>
        </Field>
        {store && !store.has_financial_config ? (
          <p className="flex items-start gap-2 rounded-md bg-warning/15 p-3 text-xs">
            <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
            <span>
              This store has no financial & staffing config yet. Set it on the <Link className="underline" href="/stores">Stores</Link> page first.
            </span>
          </p>
        ) : null}
        <Field label="Model" hint={model ? `Validation accuracy ${formatPct(model.metrics.wape !== undefined ? 1 - model.metrics.wape : null)} · horizon ${model.horizon_days} d` : undefined}>
          <Select value={mid} onChange={(e) => setModelId(e.target.value)}>
            {usable.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name ?? m.model_family} · {m.model_family} · {m.horizon_days} d · {formatDateTime(m.finished_at)}
              </option>
            ))}
          </Select>
        </Field>
        <Field label="Horizon">
          <Select value={horizon} onChange={(e) => setHorizon(e.target.value as HorizonKind)}>
            {HORIZONS.map((h) => <option key={h.value} value={h.value}>{h.label}</option>)}
          </Select>
        </Field>
        {horizon === "custom" ? (
          <div className="grid grid-cols-2 gap-3">
            <Field label="Start"><Input type="date" value={start} onChange={(e) => setStart(e.target.value)} /></Field>
            <Field label="End"><Input type="date" value={end} onChange={(e) => setEnd(e.target.value)} /></Field>
          </div>
        ) : null}
        {features.has("Promo") ? (
          <label className="flex items-center gap-2 text-sm">
            <input type="checkbox" checked={noPromoDefault} onChange={(e) => setNoPromoDefault(e.target.checked)} />
            No promos unless a lever adds them
            <span className="text-xs text-muted-foreground">(default repeats the last weeks&apos; promo pattern)</span>
          </label>
        ) : null}

        <div className="border-t pt-4">
          <div className="mb-2 flex items-center justify-between">
            <p className="text-sm font-medium">What-if levers</p>
            <Button size="sm" variant="outline" onClick={() => setLevers((l) => [...l, emptyLever()])}>
              <Plus /> Add lever
            </Button>
          </div>
          {levers.length === 0 ? (
            <p className="text-xs text-muted-foreground">Promos, holidays, closures, local or competitor events, demand shocks and price changes on date ranges.</p>
          ) : null}
          <div className="flex flex-col gap-3">
            {levers.map((l, i) => (
              <div key={i} className="rounded-md border p-3">
                <div className="mb-2 grid grid-cols-[1fr_auto_auto] gap-2">
                  <Input placeholder="Name, e.g. summer promo" value={l.name} onChange={(e) => update(i, { name: e.target.value })} />
                  <Select value={l.kind} onChange={(e) => update(i, { kind: e.target.value as LeverKind })} className="w-32">
                    {LEVER_KINDS.map((k) => <option key={k} value={k}>{k.replace("_", " ")}</option>)}
                  </Select>
                  <Button size="icon" variant="ghost" aria-label="Remove lever" onClick={() => setLevers((ls) => ls.filter((_, j) => j !== i))}>
                    <Trash2 />
                  </Button>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <Field label="From"><Input type="date" value={l.start_date} onChange={(e) => update(i, { start_date: e.target.value })} /></Field>
                  <Field label="To"><Input type="date" value={l.end_date} onChange={(e) => update(i, { end_date: e.target.value })} /></Field>
                  <TriField label="Promo" value={l.promo} onChange={(v) => update(i, { promo: v })} />
                  <TriField label="Open" value={l.open} onChange={(v) => update(i, { open: v })} onLabel="open" offLabel="closed" />
                  {features.has("SchoolHoliday") ? (
                    <TriField label="School holiday" value={l.school_holiday} onChange={(v) => update(i, { school_holiday: v })} />
                  ) : null}
                  {[...features].some((f) => f.startsWith("StateHoliday")) ? (
                    <Field label="State holiday">
                      <Select value={l.state_holiday} onChange={(e) => update(i, { state_holiday: e.target.value })}>
                        <option value="">unchanged</option>
                        <option value="0">none</option>
                        <option value="a">public (a)</option>
                        <option value="b">Easter (b)</option>
                        <option value="c">Christmas (c)</option>
                      </Select>
                    </Field>
                  ) : null}
                  <Field label="Demand change %" hint="e.g. -20 for a shock">
                    <Input type="number" step="1" value={l.demand_pct} onChange={(e) => update(i, { demand_pct: e.target.value })} />
                  </Field>
                  <Field label="Price change %">
                    <Input type="number" step="0.5" value={l.price_change_pct} onChange={(e) => update(i, { price_change_pct: e.target.value })} />
                  </Field>
                </div>
              </div>
            ))}
          </div>
        </div>

        <details className="rounded-md border p-3 text-sm">
          <summary className="cursor-pointer font-medium">Store-level levers</summary>
          <div className="mt-3 grid grid-cols-2 gap-2">
            {features.has("st_promo2_active") ? (
              <>
                <TriField label="Promo2" value={promo2} onChange={setPromo2} />
                <Field label="Promo2 months" hint="e.g. Jan,Apr,Jul,Oct">
                  <Input value={promo2Interval} onChange={(e) => setPromo2Interval(e.target.value)} />
                </Field>
              </>
            ) : null}
            <Field label="COGS override %" hint="margin adjustment">
              <Input type="number" step="0.5" value={cogs} onChange={(e) => setCogs(e.target.value)} />
            </Field>
            <Field label="Price elasticity" hint="e.g. -1.2; blank = volume unchanged">
              <Input type="number" step="0.1" max={0} value={elasticity} onChange={(e) => setElasticity(e.target.value)} />
            </Field>
          </div>
        </details>

        <ErrorNotice error={error} />
        <Button onClick={submit} disabled={pending || !storeId || !mid || !store?.has_financial_config}>
          <Rocket /> {pending ? "Queuing…" : "Run Autopilot"}
        </Button>
      </CardContent>
    </Card>
  );
}

function TriField({
  label,
  value,
  onChange,
  onLabel = "on",
  offLabel = "off",
}: {
  label: string;
  value: Tri;
  onChange: (v: Tri) => void;
  onLabel?: string;
  offLabel?: string;
}) {
  return (
    <Field label={label}>
      <Select value={value} onChange={(e) => onChange(e.target.value as Tri)}>
        <option value="">unchanged</option>
        <option value="on">{onLabel}</option>
        <option value="off">{offLabel}</option>
      </Select>
    </Field>
  );
}

function RunList({
  runs,
  loading,
  error,
  active,
  onSelect,
}: {
  runs: AutopilotRun[];
  loading: boolean;
  error: unknown;
  active: string | null;
  onSelect: (id: string) => void;
}) {
  return (
    <Card>
      <CardHeader><CardTitle>Recent plans</CardTitle></CardHeader>
      <CardContent>
        <ErrorNotice error={error} />
        {loading ? (
          <Skeleton className="h-24" />
        ) : runs.length === 0 ? (
          <p className="text-sm text-muted-foreground">No plans yet for this store.</p>
        ) : (
          <ul className="flex flex-col gap-1">
            {runs.map((r) => (
              <li key={r.id}>
                <button
                  onClick={() => onSelect(r.id)}
                  className={`flex w-full items-center justify-between gap-3 rounded-md px-3 py-2 text-left text-sm hover:bg-muted ${active === r.id ? "bg-muted" : ""}`}
                >
                  <span>
                    <span className="font-medium">{r.name ?? HORIZONS.find((h) => h.value === r.horizon)?.label}</span>
                    <span className="block text-xs text-muted-foreground">
                      {r.summary.window_start ? `${r.summary.window_start} → ${r.summary.window_end}` : formatDateTime(r.created_at)}
                      {r.scenario.levers?.length ? ` · ${r.scenario.levers.length} lever(s)` : ""}
                    </span>
                  </span>
                  <JobProgress status={r.status} pct={r.progress_pct} stage={r.stage} />
                </button>
              </li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}

function RunDetail({ id }: { id: string }) {
  const run = useAutopilotRun(id);
  if (run.isLoading) return <Skeleton className="h-[480px]" />;
  if (run.error) return <ErrorNotice error={run.error} />;
  const r = run.data;
  if (!r) return null;
  if (r.status === "failed") {
    return (
      <Card>
        <CardHeader><CardTitle>Autopilot failed</CardTitle></CardHeader>
        <CardContent><p className="text-sm text-destructive">{r.error}</p></CardContent>
      </Card>
    );
  }
  if (r.status !== "succeeded" || !("kpis" in r.result)) {
    return (
      <Card>
        <CardHeader><CardTitle>Planning…</CardTitle><CardDescription>{r.status_message ?? "Forecasting and building the plan."}</CardDescription></CardHeader>
        <CardContent><JobProgress status={r.status} pct={r.progress_pct} stage={r.stage} /></CardContent>
      </Card>
    );
  }
  return <ResultView id={id} res={r.result as AutopilotResult} />;
}

function ResultView({ id, res }: { id: string; res: AutopilotResult }) {
  const cur = res.currency;
  const k = res.kpis;
  const chartDays = useMemo(
    () =>
      res.days.map((d) => ({
        ...d,
        band: [d.revenue_p10, d.revenue_p90],
        surge: d.surge_staff,
      })),
    [res.days],
  );
  const stem = `autopilot_${res.store_code}_${res.window.start}_${res.window.end}`;
  const [downloadError, setDownloadError] = useState<unknown>(null);
  const download = (path: string, ext: string) =>
    api.download(`/autopilot/runs/${id}/${path}`, `${stem}.${ext}`).catch(setDownloadError);

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-sm font-medium">
            Store {res.store_code} · {res.window.start} → {res.window.end} ({res.window.days} days, {k.open_days} open)
          </p>
          <p className="text-xs text-muted-foreground">Data through {res.window.history_end}{res.model.model_family ? ` · model ${res.model.model_family}` : ""}</p>
        </div>
        <div className="flex gap-2">
          <Button size="sm" variant="outline" onClick={() => download("report.pdf", "pdf")}><FileText /> PDF report</Button>
          <Button size="sm" variant="outline" onClick={() => download("plan.xlsx", "xlsx")}><FileSpreadsheet /> Excel</Button>
          <Button size="sm" variant="outline" onClick={() => download("plan.csv", "csv")}><Download /> CSV</Button>
        </div>
      </div>
      <ErrorNotice error={downloadError} />
      {res.cold_start ? (
        <p className="rounded-md bg-warning/15 p-3 text-xs">{res.cold_start.note}</p>
      ) : null}

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Kpi label="Revenue (P50)" value={formatCurrency(k.revenue_p50, cur)} sub={`P10 ${formatCurrency(k.revenue_p10, cur)} · P90 ${formatCurrency(k.revenue_p90, cur)}`} />
        <Kpi
          label="Net profit (P50)"
          value={formatCurrency(k.net_profit_p50, cur)}
          sub={
            res.scenario_impact
              ? `${res.scenario_impact.net_profit_p50 >= 0 ? "+" : ""}${formatCurrency(res.scenario_impact.net_profit_p50, cur)} vs baseline`
              : `margin ${formatPct(k.net_margin_pct)}`
          }
        />
        <Kpi label="Footfall" value={formatNumber(k.footfall)} sub={`break-even ${formatCurrency(k.break_even_revenue, cur)} · safety ${formatPct(k.margin_of_safety_pct)}`} />
        <Kpi label="Labour" value={formatCurrency(k.labor_cost, cur)} sub={`${formatNumber(k.labor_hours)} h · peak ${k.peak_staff_per_shift}/shift`} />
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Autopilot actions</CardTitle>
          <CardDescription>Prioritized by severity, then money at stake.</CardDescription>
        </CardHeader>
        <CardContent>
          {res.recommendations.length === 0 ? (
            <p className="text-sm text-muted-foreground">No actions needed for this window.</p>
          ) : (
            <ul className="flex flex-col divide-y">
              {res.recommendations.map((rec, i) => (
                <li key={i} className="flex gap-3 py-3">
                  <Badge tone={PRIORITY_TONE[rec.priority]} className="h-fit shrink-0 uppercase">{rec.priority}</Badge>
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-medium">{rec.title}</p>
                    <p className="text-sm text-muted-foreground">{rec.detail}</p>
                  </div>
                  {rec.impact !== null ? (
                    <span className={`shrink-0 text-sm tabular-nums ${rec.impact < 0 ? "text-destructive" : "text-success"}`}>
                      {formatCurrency(rec.impact, cur)}
                    </span>
                  ) : null}
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader><CardTitle>Revenue forecast</CardTitle><CardDescription>P50 with the P10–P90 band, scenario baseline and daily break-even</CardDescription></CardHeader>
        <CardContent><RevenuePlanChart data={chartDays} /></CardContent>
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader><CardTitle>Daily net profit</CardTitle><CardDescription>P50 after COGS, labour and overhead</CardDescription></CardHeader>
          <CardContent><SimpleBarChart data={chartDays} xKey="ds" series={[{ key: "net_profit_p50", label: "Net profit P50" }]} /></CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle>Headcount per shift</CardTitle><CardDescription>Plus peak-hour cover</CardDescription></CardHeader>
          <CardContent>
            <SimpleBarChart
              data={chartDays}
              xKey="ds"
              stacked
              series={[...res.shift_names.map((n) => ({ key: `staff_${n}`, label: n })), { key: "surge", label: "Peak cover", color: "var(--chart-3)" }]}
            />
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader><CardTitle>P&L</CardTitle><CardDescription>{res.window.days} days</CardDescription></CardHeader>
          <CardContent>
            <Table>
              <tbody>
                {[
                  ["Gross (list-price) revenue", res.totals.gross_revenue],
                  ["Promo discount", -res.totals.promo_discount],
                  ["Net revenue P50", res.totals.revenue_p50],
                  ["COGS", -res.totals.cogs],
                  ["Gross profit", res.totals.gross_profit],
                  ["Labour", -res.totals.labor_cost],
                  ["Fixed overhead", -res.totals.fixed_overhead],
                  ["Net profit P50", res.totals.net_profit_p50],
                ].map(([label, v]) => (
                  <TR key={label as string}>
                    <TD>{label}</TD>
                    <TD className="text-right tabular-nums">{formatCurrency(v as number, cur)}</TD>
                  </TR>
                ))}
                <TR>
                  <TD className="text-muted-foreground">Net profit P10 / P90</TD>
                  <TD className="text-right tabular-nums text-muted-foreground">
                    {formatCurrency(res.totals.net_profit_p10, cur)} / {formatCurrency(res.totals.net_profit_p90, cur)}
                  </TD>
                </TR>
              </tbody>
            </Table>
          </CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle>Model comparison</CardTitle><CardDescription>Validation metrics of finished runs on this dataset</CardDescription></CardHeader>
          <CardContent>
            {res.model_comparison.length === 0 ? (
              <p className="text-sm text-muted-foreground">No other finished models.</p>
            ) : (
              <Table>
                <THead><tr><TH>Model</TH><TH>Accuracy</TH><TH>RMSPE</TH><TH>RMSSE</TH></tr></THead>
                <tbody>
                  {res.model_comparison.slice(0, 8).map((m) => (
                    <TR key={m.model_run_id} className={m.model_run_id === res.model.model_run_id ? "bg-muted" : ""}>
                      <TD>{m.name ?? m.model_family}<span className="block text-xs text-muted-foreground">{m.model_family} · {m.horizon_days} d</span></TD>
                      <TD>{formatPct(m.accuracy)}</TD>
                      <TD>{formatNumber(m.rmspe, 4)}</TD>
                      <TD>{formatNumber(m.rmsse, 4)}</TD>
                    </TR>
                  ))}
                </tbody>
              </Table>
            )}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader><CardTitle>Shift plan</CardTitle></CardHeader>
        <CardContent className="overflow-x-auto">
          <Table>
            <THead>
              <tr>
                <TH>Date</TH><TH>Revenue P50</TH><TH>Footfall</TH>
                {res.shift_names.map((n) => <TH key={n}>{n}</TH>)}
                <TH>Peak +</TH><TH>Hours</TH><TH>Net P50</TH><TH>Status</TH>
              </tr>
            </THead>
            <tbody>
              {res.days.map((d) => (
                <TR key={d.ds} className={d.is_open ? "" : "text-muted-foreground"}>
                  <TD className="whitespace-nowrap">{d.ds}{d.is_promo ? <Badge tone="info" className="ml-2">promo</Badge> : null}{d.event ? <span className="ml-2 text-xs">{d.event}</span> : null}</TD>
                  <TD className="tabular-nums">{formatCurrency(d.revenue_p50, cur)}</TD>
                  <TD className="tabular-nums">{formatNumber(d.footfall)}</TD>
                  {res.shift_names.map((n) => <TD key={n} className="tabular-nums">{d[`staff_${n}`] ?? 0}</TD>)}
                  <TD className="tabular-nums">{d.surge_staff}</TD>
                  <TD className="tabular-nums">{formatNumber(d.labor_hours)}</TD>
                  <TD className={`tabular-nums ${d.net_profit_p50 < 0 ? "text-destructive" : ""}`}>{formatCurrency(d.net_profit_p50, cur)}</TD>
                  <TD>
                    <Badge tone={d.staffing_status === "understaffed" ? "danger" : d.staffing_status === "overstaffed" ? "warning" : d.staffing_status === "closed" ? "neutral" : "success"}>
                      {d.staffing_status}
                    </Badge>
                  </TD>
                </TR>
              ))}
            </tbody>
          </Table>
        </CardContent>
      </Card>

      <Card>
        <CardHeader><CardTitle>Assumptions</CardTitle></CardHeader>
        <CardContent>
          <ul className="list-disc space-y-1 pl-5 text-sm text-muted-foreground">
            {res.assumptions.map((a) => <li key={a}>{a}</li>)}
          </ul>
        </CardContent>
      </Card>
    </div>
  );
}
