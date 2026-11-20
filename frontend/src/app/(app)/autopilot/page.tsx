"use client";

import Link from "next/link";
import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, Plus, RotateCcw, Rocket, SlidersHorizontal, Trash2 } from "lucide-react";
import { ApiError, api } from "@/lib/api";
import type {
  AutopilotRequest,
  AutopilotResult,
  AutopilotRun,
  AutopilotScenario,
  FinancialConfig,
  HealthReport,
  HorizonKind,
  LeverKind,
  ModelRun,
  ScenarioLever,
} from "@/lib/types";
import { useAutopilotRun, useAutopilotRuns, useDataset, useModelRuns, useStores } from "@/hooks/queries";
import { Button } from "@/components/ui/button";
import { Badge, Card, CardContent, CardDescription, CardHeader, CardTitle, Field, Input, Select, Skeleton, Table, TD, TH, THead, TR } from "@/components/ui/primitives";
import { EmptyState, ErrorNotice, JobProgress, PageHeader } from "@/components/shared";
import {
  AlertBanners,
  AssumptionsCard,
  ExportButtons,
  PlanKpis,
  PnlWaterfallCard,
  ProfitCard,
  RecommendationsFeed,
  RevenueCard,
  ShiftRoster,
  StaffingCard,
} from "@/components/autopilot";
import { cn, formatDateTime, formatNumber, formatPct } from "@/lib/utils";

const HORIZONS: { value: HorizonKind; label: string; days?: number }[] = [
  { value: "next_day", label: "Next day", days: 1 },
  { value: "week", label: "7 days", days: 7 },
  { value: "month", label: "30 days", days: 30 },
  { value: "custom", label: "Custom" },
];
const LEVER_KINDS: LeverKind[] = ["promo", "holiday", "closure", "local_event", "competitor", "price", "custom"];

type Tri = "" | "on" | "off";
const tri = (v: Tri): boolean | null => (v === "" ? null : v === "on");
const num = (v: string): number | null => (v.trim() === "" ? null : Number(v));

const addDays = (iso: string, n: number) => {
  const d = new Date(`${iso}T00:00:00Z`);
  d.setUTCDate(d.getUTCDate() + n);
  return d.toISOString().slice(0, 10);
};

interface Simulator {
  promo: Tri;
  schoolHoliday: Tri;
  stateHoliday: string;
  promo2: Tri;
  promo2Interval: string;
  demandPct: number;
  marginPct: string;
  hourlyWage: string;
  elasticity: string;
  priceChangePct: string;
  noPromoDefault: boolean;
}

const EMPTY_SIM: Simulator = {
  promo: "",
  schoolHoliday: "",
  stateHoliday: "",
  promo2: "",
  promo2Interval: "",
  demandPct: 0,
  marginPct: "",
  hourlyWage: "",
  elasticity: "",
  priceChangePct: "",
  noPromoDefault: false,
};

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

const emptyLever = (start = ""): LeverDraft => ({
  name: "",
  kind: "local_event",
  start_date: start,
  end_date: start,
  promo: "",
  school_holiday: "",
  open: "",
  state_holiday: "",
  demand_pct: "",
  price_change_pct: "",
});

function leverFromDraft(d: LeverDraft): ScenarioLever {
  const pct = (v: string) => (v.trim() === "" ? null : Number(v) / 100);
  return {
    name: d.name || d.kind.replace("_", " "),
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

/** Whole-window quick toggles + dated events -> the API scenario. */
function buildScenario(sim: Simulator, events: LeverDraft[], window: { start: string; end: string } | null): AutopilotScenario {
  const levers: ScenarioLever[] = [];
  const quick: Partial<ScenarioLever> = {
    promo: tri(sim.promo),
    school_holiday: tri(sim.schoolHoliday),
    state_holiday: sim.stateHoliday || null,
    demand_pct: sim.demandPct ? sim.demandPct / 100 : null,
    price_change_pct: num(sim.priceChangePct) !== null ? Number(sim.priceChangePct) / 100 : null,
  };
  if (window && Object.values(quick).some((v) => v !== null)) {
    const kind: LeverKind = quick.promo !== null ? "promo" : quick.price_change_pct !== null ? "price" : quick.demand_pct !== null ? "local_event" : "holiday";
    levers.push({ name: "Simulator (whole window)", kind, start_date: window.start, end_date: window.end, ...quick });
  }
  levers.push(...events.filter((e) => e.start_date).map(leverFromDraft));
  const margin = num(sim.marginPct);
  return {
    levers,
    promo2: tri(sim.promo2),
    promo2_interval: sim.promo2Interval || null,
    covariate_defaults: sim.noPromoDefault ? { Promo: 0 } : {},
    cogs_pct: margin === null ? null : 1 - margin / 100,
    hourly_wage: num(sim.hourlyWage),
    price_elasticity: num(sim.elasticity),
  };
}

export default function AutopilotDashboard() {
  const qc = useQueryClient();
  const stores = useStores();
  const models = useModelRuns();
  const [storeId, setStoreId] = useState("");
  const [modelId, setModelId] = useState("");
  const [horizon, setHorizon] = useState<HorizonKind>("week");
  const [custom, setCustom] = useState({ start: "", end: "" });
  const [sim, setSim] = useState<Simulator>(EMPTY_SIM);
  const [events, setEvents] = useState<LeverDraft[]>([]);
  const [selectedRun, setSelectedRun] = useState<string | null>(null);

  const sid = storeId || stores.data?.items.find((s) => s.has_financial_config)?.id || stores.data?.items[0]?.id || "";
  const store = stores.data?.items.find((s) => s.id === sid);
  const usable = (models.data?.items ?? []).filter((m) => m.status === "succeeded" && (!m.store_id || m.store_id === sid));
  const model: ModelRun | undefined = usable.find((m) => m.id === modelId) ?? usable[0];
  const dataset = useDataset(model?.dataset_id ?? "");
  const features = new Set(model?.feature_columns ?? []);
  const config = useQuery({
    queryKey: ["fin-config", sid],
    enabled: !!sid,
    queryFn: async () => {
      try {
        return await api.get<FinancialConfig>(`/stores/${sid}/financial-config`);
      } catch (e) {
        if (e instanceof ApiError && e.status === 404) return null;
        throw e;
      }
    },
  });
  const runs = useAutopilotRuns(sid || undefined);
  const runId = selectedRun ?? runs.data?.items[0]?.id ?? null;

  const dataEnd = dataset.data?.date_max ?? null;
  const first = dataEnd ? addDays(dataEnd, 1) : null;
  const span = HORIZONS.find((h) => h.value === horizon)?.days;
  const window =
    horizon === "custom"
      ? custom.start && custom.end
        ? custom
        : null
      : first && span
        ? { start: first, end: addDays(first, span - 1) }
        : null;
  const maxDate = dataEnd && model ? addDays(dataEnd, model.horizon_days) : undefined;

  const create = useMutation({
    mutationFn: (body: AutopilotRequest) => api.post<{ id: string }>("/autopilot/runs", body),
    onSuccess: (r) => {
      setSelectedRun(r.id);
      void qc.invalidateQueries({ queryKey: ["autopilot-runs"] });
    },
  });
  const run = () =>
    model &&
    create.mutate({
      store_id: sid,
      model_run_id: model.id,
      horizon,
      start_date: horizon === "custom" ? custom.start : null,
      end_date: horizon === "custom" ? custom.end : null,
      scenario: buildScenario(sim, events, window),
    });

  if (stores.isLoading || models.isLoading) {
    return (
      <>
        <PageHeader title="Store Autopilot" />
        <div className="grid gap-4 md:grid-cols-4">{[0, 1, 2, 3].map((i) => <Skeleton key={i} className="h-28" />)}</div>
      </>
    );
  }
  if (stores.error || models.error) return <ErrorNotice error={stores.error ?? models.error} />;
  if (!(models.data?.items ?? []).some((m) => m.status === "succeeded")) {
    return (
      <>
        <PageHeader title="Store Autopilot" />
        <EmptyState
          title="No trained models yet"
          description="Upload sales history, train a model in Model Lab, then plan stores here."
          action={<Button asChild size="sm"><Link href="/model-lab">Open Model Lab</Link></Button>}
        />
      </>
    );
  }

  const simChanges = [
    sim.promo && `promo ${sim.promo}`,
    sim.schoolHoliday && `school holiday ${sim.schoolHoliday}`,
    sim.stateHoliday && `state holiday ${sim.stateHoliday}`,
    sim.promo2 && `Promo2 ${sim.promo2}`,
    sim.demandPct && `demand ${sim.demandPct > 0 ? "+" : ""}${sim.demandPct}%`,
    sim.priceChangePct && `price ${sim.priceChangePct}%`,
    sim.marginPct && `margin ${sim.marginPct}%`,
    sim.hourlyWage && `wage ${sim.hourlyWage}`,
    sim.noPromoDefault && "no default promos",
    events.length && `${events.length} event(s)`,
  ].filter(Boolean) as string[];

  return (
    <>
      <PageHeader
        title="Store Autopilot"
        description="Executive plan for one store: demand with P10–P90 ranges, footfall, profit and staffing — and what-if simulation before you commit."
      />

      <Card className="mb-6">
        <CardContent className="flex flex-wrap items-end gap-4 p-5">
          <Field label="Store">
            <Select
              value={sid}
              onChange={(e) => {
                setStoreId(e.target.value);
                setSelectedRun(null);
              }}
              className="w-56"
            >
              {stores.data?.items.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.code} — {s.name}{s.has_financial_config ? "" : " (no config)"}
                </option>
              ))}
            </Select>
          </Field>
          <Field label="Model">
            <Select value={model?.id ?? ""} onChange={(e) => setModelId(e.target.value)} className="w-72">
              {usable.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name ?? m.model_family} · {m.model_family} · {m.horizon_days} d · acc {formatPct(m.metrics.wape !== undefined ? 1 - m.metrics.wape : null)}
                </option>
              ))}
            </Select>
          </Field>
          <div className="flex flex-col gap-1.5">
            <span className="text-xs font-medium text-muted-foreground">Horizon</span>
            <div role="tablist" className="inline-flex h-9 rounded-md border bg-muted p-0.5">
              {HORIZONS.map((h) => (
                <button
                  key={h.value}
                  role="tab"
                  aria-selected={horizon === h.value}
                  onClick={() => setHorizon(h.value)}
                  className={cn(
                    "rounded px-3 text-sm transition-colors",
                    horizon === h.value ? "bg-card font-medium shadow-xs" : "text-muted-foreground hover:text-foreground",
                  )}
                >
                  {h.label}
                </button>
              ))}
            </div>
          </div>
          {horizon === "custom" ? (
            <>
              <Field label="From"><Input type="date" min={first ?? undefined} max={maxDate} value={custom.start} onChange={(e) => setCustom({ ...custom, start: e.target.value })} /></Field>
              <Field label="To"><Input type="date" min={custom.start || first || undefined} max={maxDate} value={custom.end} onChange={(e) => setCustom({ ...custom, end: e.target.value })} /></Field>
            </>
          ) : null}
          <div className="ml-auto flex flex-col items-end gap-1">
            <Button onClick={run} disabled={create.isPending || !model || !store?.has_financial_config || (horizon === "custom" && !window)}>
              <Rocket /> {create.isPending ? "Queuing…" : simChanges.length ? "Simulate scenario" : "Run plan"}
            </Button>
            <span className="text-[11px] text-muted-foreground">
              {window ? `${window.start} → ${window.end}` : dataEnd ? "Pick dates" : "…"}
              {dataEnd ? ` · data through ${dataEnd}` : ""}
            </span>
          </div>
        </CardContent>
      </Card>
      {store && !store.has_financial_config ? (
        <p className="mb-6 flex items-start gap-2 rounded-md border border-warning/40 bg-warning/10 p-3 text-sm">
          <AlertTriangle className="mt-0.5 size-4 shrink-0" />
          This store has no financial & staffing configuration yet — set it on the&nbsp;<Link className="underline" href="/stores">Stores</Link>&nbsp;page.
        </p>
      ) : null}
      <ErrorNotice error={create.error} />

      <div className="mt-6 grid gap-6 xl:grid-cols-[22rem_1fr]">
        <div className="flex flex-col gap-6">
          <SimulatorCard
            sim={sim}
            setSim={setSim}
            events={events}
            setEvents={setEvents}
            features={features}
            config={config.data ?? null}
            defaultStart={window?.start ?? first ?? ""}
            changes={simChanges}
          />
          <RunList runs={runs.data?.items ?? []} loading={runs.isLoading} error={runs.error} active={runId} onSelect={setSelectedRun} />
        </div>
        <div className="min-w-0">
          {runId ? (
            <RunDetail id={runId} datasetId={model?.dataset_id} />
          ) : (
            <EmptyState title="No plans for this store yet" description="Pick a horizon, optionally simulate a scenario, then run the Autopilot." />
          )}
        </div>
      </div>
    </>
  );
}

function SimulatorCard({
  sim,
  setSim,
  events,
  setEvents,
  features,
  config,
  defaultStart,
  changes,
}: {
  sim: Simulator;
  setSim: (s: Simulator) => void;
  events: LeverDraft[];
  setEvents: (e: LeverDraft[] | ((e: LeverDraft[]) => LeverDraft[])) => void;
  features: Set<string>;
  config: FinancialConfig | null;
  defaultStart: string;
  changes: string[];
}) {
  const set = (patch: Partial<Simulator>) => setSim({ ...sim, ...patch });
  const update = (i: number, patch: Partial<LeverDraft>) => setEvents((ls) => ls.map((l, j) => (j === i ? { ...l, ...patch } : l)));
  const cfgMargin = config ? Math.round((1 - Number(config.cogs_pct)) * 1000) / 10 : null;
  const hasState = [...features].some((f) => f.startsWith("StateHoliday"));
  return (
    <Card>
      <CardHeader className="flex flex-row items-start justify-between gap-2">
        <div>
          <CardTitle className="flex items-center gap-2"><SlidersHorizontal className="size-4" /> What-if simulator</CardTitle>
          <CardDescription>Toggles apply to the whole window; add dated events below. Every simulation is compared with the baseline.</CardDescription>
        </div>
        {changes.length ? (
          <Button size="icon" variant="ghost" aria-label="Reset simulator" onClick={() => { setSim(EMPTY_SIM); setEvents([]); }}>
            <RotateCcw />
          </Button>
        ) : null}
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        {changes.length ? (
          <div className="flex flex-wrap gap-1.5">{changes.map((c) => <Badge key={c} tone="info">{c}</Badge>)}</div>
        ) : null}
        <div className="grid grid-cols-2 gap-3">
          {features.has("Promo") ? <TriSelect label="Promo" value={sim.promo} onChange={(v) => set({ promo: v })} /> : null}
          {features.has("SchoolHoliday") ? <TriSelect label="School holiday" value={sim.schoolHoliday} onChange={(v) => set({ schoolHoliday: v })} /> : null}
          {hasState ? (
            <Field label="State holiday">
              <Select value={sim.stateHoliday} onChange={(e) => set({ stateHoliday: e.target.value })}>
                <option value="">unchanged</option>
                <option value="0">none</option>
                <option value="a">public (a)</option>
                <option value="b">Easter (b)</option>
                <option value="c">Christmas (c)</option>
              </Select>
            </Field>
          ) : null}
          {features.has("st_promo2_active") ? <TriSelect label="Promo2" value={sim.promo2} onChange={(v) => set({ promo2: v })} /> : null}
        </div>
        {sim.promo2 === "on" ? (
          <Field label="Promo2 months" hint="Needed if the store has no Promo2 interval yet">
            <Input placeholder="Jan,Apr,Jul,Oct" value={sim.promo2Interval} onChange={(e) => set({ promo2Interval: e.target.value })} />
          </Field>
        ) : null}
        <Field label={`Demand uplift / shock: ${sim.demandPct > 0 ? "+" : ""}${sim.demandPct}%`} hint="Local event, competitor opening, weather…">
          <input
            type="range"
            min={-50}
            max={50}
            step={5}
            value={sim.demandPct}
            onChange={(e) => set({ demandPct: Number(e.target.value) })}
            className="accent-[var(--primary)]"
            aria-label="Demand uplift or shock percent"
          />
        </Field>
        <div className="grid grid-cols-2 gap-3">
          <Field label="Gross margin %" hint={cfgMargin !== null ? `config ${cfgMargin}%` : undefined}>
            <Input type="number" step="0.5" min={0} max={100} placeholder={cfgMargin !== null ? String(cfgMargin) : ""} value={sim.marginPct} onChange={(e) => set({ marginPct: e.target.value })} />
          </Field>
          <Field label="Hourly wage" hint={config ? `config ${config.hourly_wage}` : undefined}>
            <Input type="number" step="0.5" min={0} placeholder={config?.hourly_wage ?? ""} value={sim.hourlyWage} onChange={(e) => set({ hourlyWage: e.target.value })} />
          </Field>
          <Field label="Price change %">
            <Input type="number" step="0.5" value={sim.priceChangePct} onChange={(e) => set({ priceChangePct: e.target.value })} />
          </Field>
          <Field label="Price elasticity" hint="blank = volume unchanged">
            <Input type="number" step="0.1" max={0} placeholder="-1.2" value={sim.elasticity} onChange={(e) => set({ elasticity: e.target.value })} />
          </Field>
        </div>
        {features.has("Promo") ? (
          <label className="flex items-start gap-2 text-sm">
            <input type="checkbox" className="mt-1 accent-[var(--primary)]" checked={sim.noPromoDefault} onChange={(e) => set({ noPromoDefault: e.target.checked })} />
            <span>
              No promos unless a lever adds them
              <span className="block text-xs text-muted-foreground">Otherwise future promo days repeat the last observed weeks.</span>
            </span>
          </label>
        ) : null}

        <div className="border-t pt-3">
          <div className="mb-2 flex items-center justify-between">
            <p className="text-sm font-medium">Dated events</p>
            <Button size="sm" variant="outline" onClick={() => setEvents((l) => [...l, emptyLever(defaultStart)])}>
              <Plus /> Event
            </Button>
          </div>
          <div className="flex flex-col gap-3">
            {events.map((l, i) => (
              <div key={i} className="rounded-md border p-3">
                <div className="mb-2 flex gap-2">
                  <Input placeholder="Name" value={l.name} onChange={(e) => update(i, { name: e.target.value })} />
                  <Button size="icon" variant="ghost" aria-label="Remove event" onClick={() => setEvents((ls) => ls.filter((_, j) => j !== i))}>
                    <Trash2 />
                  </Button>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <Field label="Kind">
                    <Select value={l.kind} onChange={(e) => update(i, { kind: e.target.value as LeverKind })}>
                      {LEVER_KINDS.map((k) => <option key={k} value={k}>{k.replace("_", " ")}</option>)}
                    </Select>
                  </Field>
                  <TriSelect label="Open" value={l.open} onChange={(v) => update(i, { open: v })} onLabel="open" offLabel="closed" />
                  <Field label="From"><Input type="date" value={l.start_date} onChange={(e) => update(i, { start_date: e.target.value })} /></Field>
                  <Field label="To"><Input type="date" value={l.end_date} onChange={(e) => update(i, { end_date: e.target.value })} /></Field>
                  {features.has("Promo") ? <TriSelect label="Promo" value={l.promo} onChange={(v) => update(i, { promo: v })} /> : null}
                  {features.has("SchoolHoliday") ? <TriSelect label="School holiday" value={l.school_holiday} onChange={(v) => update(i, { school_holiday: v })} /> : null}
                  <Field label="Demand %"><Input type="number" step="1" value={l.demand_pct} onChange={(e) => update(i, { demand_pct: e.target.value })} /></Field>
                  <Field label="Price %"><Input type="number" step="0.5" value={l.price_change_pct} onChange={(e) => update(i, { price_change_pct: e.target.value })} /></Field>
                </div>
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function TriSelect({ label, value, onChange, onLabel = "on", offLabel = "off" }: { label: string; value: Tri; onChange: (v: Tri) => void; onLabel?: string; offLabel?: string }) {
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

function RunList({ runs, loading, error, active, onSelect }: { runs: AutopilotRun[]; loading: boolean; error: unknown; active: string | null; onSelect: (id: string) => void }) {
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
                  className={cn("flex w-full items-center justify-between gap-3 rounded-md px-3 py-2 text-left text-sm hover:bg-muted", active === r.id && "bg-muted")}
                >
                  <span className="min-w-0">
                    <span className="font-medium">{r.name ?? HORIZONS.find((h) => h.value === r.horizon)?.label}</span>
                    <span className="block truncate text-xs text-muted-foreground">
                      {r.summary.window_start ? `${r.summary.window_start} → ${r.summary.window_end}` : formatDateTime(r.created_at)}
                      {r.scenario.levers?.length || r.scenario.hourly_wage || r.scenario.cogs_pct ? " · scenario" : ""}
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

function RunDetail({ id, datasetId }: { id: string; datasetId?: string }) {
  const run = useAutopilotRun(id);
  const health = useQuery({
    queryKey: ["dataset-health", datasetId],
    enabled: !!datasetId,
    queryFn: () => api.get<{ report: HealthReport | null }>(`/datasets/${datasetId}/health`),
  });
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
        <CardHeader><CardTitle>Planning…</CardTitle><CardDescription>{r.status_message ?? "Forecasting baseline and scenario, then building the plan."}</CardDescription></CardHeader>
        <CardContent><JobProgress status={r.status} pct={r.progress_pct} stage={r.stage} /></CardContent>
      </Card>
    );
  }
  const res = r.result as AutopilotResult;
  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="text-sm font-medium">
            Store {res.store_code} · {res.window.start} → {res.window.end} ({res.window.days} days, {res.kpis.open_days} open)
          </p>
          <p className="text-xs text-muted-foreground">
            Data through {res.window.history_end}
            {res.model.model_family ? ` · ${res.model.model_family}` : ""}
            {res.model.metrics?.wape != null ? ` · validation accuracy ${formatPct(1 - (res.model.metrics.wape as number))}` : ""}
          </p>
        </div>
        <ExportButtons runId={id} res={res} />
      </div>
      <PlanKpis res={res} />
      <AlertBanners res={res} health={health.data?.report} />
      <RecommendationsFeed res={res} />
      <RevenueCard res={res} />
      <div className="grid gap-6 lg:grid-cols-2">
        <ProfitCard res={res} />
        <StaffingCard res={res} />
      </div>
      <PnlWaterfallCard res={res} />
      <ShiftRoster res={res} />
      <ModelComparison res={res} />
      <AssumptionsCard items={res.assumptions} />
    </div>
  );
}

function ModelComparison({ res }: { res: AutopilotResult }) {
  if (!res.model_comparison.length) return null;
  return (
    <Card>
      <CardHeader><CardTitle>Model comparison</CardTitle><CardDescription>Validation metrics of finished runs on this dataset; highlighted = used for this plan</CardDescription></CardHeader>
      <CardContent className="overflow-x-auto">
        <Table>
          <THead><tr><TH>Model</TH><TH className="text-right">Horizon</TH><TH className="text-right">Accuracy</TH><TH className="text-right">WAPE</TH><TH className="text-right">RMSPE</TH><TH className="text-right">RMSSE</TH><TH className="text-right">R²</TH></tr></THead>
          <tbody>
            {res.model_comparison.slice(0, 10).map((m) => (
              <TR key={m.model_run_id} className={m.model_run_id === res.model.model_run_id ? "bg-muted font-medium" : ""}>
                <TD><Link className="hover:underline" href={`/model-lab/${m.model_run_id}`}>{m.name ?? m.model_family}</Link><span className="block text-xs text-muted-foreground">{m.model_family} · {m.run_type}</span></TD>
                <TD className="text-right">{m.horizon_days} d</TD>
                <TD className="text-right tabular-nums">{formatPct(m.accuracy, 2)}</TD>
                <TD className="text-right tabular-nums">{formatNumber(m.wape, 4)}</TD>
                <TD className="text-right tabular-nums">{formatNumber(m.rmspe, 4)}</TD>
                <TD className="text-right tabular-nums">{formatNumber(m.rmsse, 4)}</TD>
                <TD className="text-right tabular-nums">{formatNumber(m.r2, 3)}</TD>
              </TR>
            ))}
          </tbody>
        </Table>
      </CardContent>
    </Card>
  );
}
