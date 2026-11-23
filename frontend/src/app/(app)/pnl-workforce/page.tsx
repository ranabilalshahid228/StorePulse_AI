"use client";

import Link from "next/link";
import { useState } from "react";
import type { AutopilotResult } from "@/lib/types";
import { useAutopilotRun, useAutopilotRuns, useStores } from "@/hooks/queries";
import { Button } from "@/components/ui/button";
import { Card, CardContent, Field, Select, Skeleton } from "@/components/ui/primitives";
import { EmptyState, ErrorNotice, JobProgress, PageHeader } from "@/components/shared";
import { hasPlanInterval, AssumptionsCard, ExportButtons, PnlWaterfallCard, ProfitCard, ShiftRoster, StaffingCard } from "@/components/autopilot";
import { ForecastPnl } from "@/components/forecast-pnl";
import { cn, formatCurrency, formatNumber } from "@/lib/utils";

type Tab = "autopilot" | "forecast";

export default function PnlWorkforcePage() {
  const [tab, setTab] = useState<Tab>("autopilot");
  return (
    <>
      <PageHeader
        title="P&L & Workforce"
        description="Profit bridge, daily P&L and the shift-by-shift roster for a store plan — with one-click PDF, Excel and CSV exports."
        actions={
          <div role="tablist" className="inline-flex h-9 rounded-md border bg-muted p-0.5">
            {(
              [
                ["autopilot", "Autopilot plans"],
                ["forecast", "Forecast runs"],
              ] as [Tab, string][]
            ).map(([k, label]) => (
              <button
                key={k}
                role="tab"
                aria-selected={tab === k}
                onClick={() => setTab(k)}
                className={cn("rounded px-3 text-sm", tab === k ? "bg-card font-medium shadow-xs" : "text-muted-foreground hover:text-foreground")}
              >
                {label}
              </button>
            ))}
          </div>
        }
      />
      {tab === "autopilot" ? <AutopilotPnl /> : <ForecastPnl />}
    </>
  );
}

function AutopilotPnl() {
  const stores = useStores();
  const [storeId, setStoreId] = useState("");
  const sid = storeId || stores.data?.items.find((s) => s.has_financial_config)?.id || stores.data?.items[0]?.id || "";
  const runs = useAutopilotRuns(sid || undefined);
  const done = (runs.data?.items ?? []).filter((r) => r.status === "succeeded");
  const [runId, setRunId] = useState("");
  const rid = runId && done.some((r) => r.id === runId) ? runId : done[0]?.id ?? "";
  const run = useAutopilotRun(rid || null);
  const active = (runs.data?.items ?? []).filter((r) => ["queued", "running"].includes(r.status));

  if (stores.isLoading || runs.isLoading) return <Skeleton className="h-64" />;
  return (
    <>
      <Card className="mb-6">
        <CardContent className="flex flex-wrap items-end gap-4 p-5">
          <Field label="Store">
            <Select value={sid} onChange={(e) => { setStoreId(e.target.value); setRunId(""); }} className="w-56">
              {stores.data?.items.map((s) => <option key={s.id} value={s.id}>{s.code} — {s.name}</option>)}
            </Select>
          </Field>
          <Field label="Plan">
            <Select value={rid} onChange={(e) => setRunId(e.target.value)} className="w-80" disabled={!done.length}>
              {done.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.summary.window_start} → {r.summary.window_end} · {r.horizon.replace("_", " ")}
                  {r.scenario.levers?.length || r.scenario.hourly_wage || r.scenario.cogs_pct ? " · scenario" : ""}
                </option>
              ))}
            </Select>
          </Field>
          {active.length ? <JobProgress status={active[0]!.status} pct={active[0]!.progress_pct} stage={active[0]!.stage} /> : null}
          {run.data && "kpis" in run.data.result ? (
            <div className="ml-auto"><ExportButtons runId={rid} res={run.data.result as AutopilotResult} /></div>
          ) : null}
        </CardContent>
      </Card>
      <ErrorNotice error={runs.error ?? run.error} />
      {!done.length ? (
        <EmptyState
          title="No finished plans for this store"
          description="Run the Store Autopilot to get a P&L, staffing roster and reports."
          action={<Button asChild size="sm"><Link href="/autopilot">Open Store Autopilot</Link></Button>}
        />
      ) : run.isLoading || !run.data ? (
        <Skeleton className="h-96" />
      ) : "kpis" in run.data.result ? (
        <PlanView res={run.data.result as AutopilotResult} />
      ) : null}
    </>
  );
}

function PlanView({ res }: { res: AutopilotResult }) {
  const k = res.kpis;
  const cur = res.currency;
  return (
    <div className="flex flex-col gap-6">
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="Net revenue P50" value={formatCurrency(k.revenue_p50, cur)} sub={hasPlanInterval(res) ? `P10 ${formatCurrency(k.revenue_p10, cur)} · P90 ${formatCurrency(k.revenue_p90, cur)}` : "Forecast interval unavailable"} />
        <Stat label="Net profit P50" value={formatCurrency(k.net_profit_p50, cur)} sub={hasPlanInterval(res) ? `P10 ${formatCurrency(k.net_profit_p10, cur)} · P90 ${formatCurrency(k.net_profit_p90, cur)}` : "Forecast interval unavailable"} loss={k.net_profit_p50 < 0} />
        <Stat label="Labour" value={formatCurrency(k.labor_cost, cur)} sub={`${formatNumber(k.labor_hours)} scheduled hours`} />
        <Stat label="Staffing flags" value={`${k.understaffed_shifts} under / ${k.overstaffed_shifts} over`} sub={`peak ${k.peak_staff_per_shift} staff per shift`} />
      </div>
      <PnlWaterfallCard res={res} />
      <div className="grid gap-6 lg:grid-cols-2">
        <ProfitCard res={res} />
        <StaffingCard res={res} />
      </div>
      <ShiftRoster res={res} />
      <AssumptionsCard items={res.assumptions} />
    </div>
  );
}

function Stat({ label, value, sub, loss }: { label: string; value: string; sub: string; loss?: boolean }) {
  return (
    <Card>
      <CardContent className="p-5">
        <p className="text-xs font-medium text-muted-foreground">{label}</p>
        <p className={cn("mt-2 text-2xl font-semibold tracking-tight tabular-nums", loss && "text-destructive")}>{value}</p>
        <p className="mt-1 text-xs text-muted-foreground">{sub}</p>
      </CardContent>
    </Card>
  );
}
