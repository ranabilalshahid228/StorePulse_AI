"use client";

import { useState, type FormEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ApiError, api } from "@/lib/api";
import type { AutopilotSettings, FinancialConfig, Store } from "@/lib/types";
import { useStores } from "@/hooks/queries";
import { Button } from "@/components/ui/button";
import { Badge, Card, CardContent, CardDescription, CardHeader, CardTitle, Field, Input, Table, TD, TH, THead, TR } from "@/components/ui/primitives";
import { EmptyState, ErrorNotice, PageHeader } from "@/components/shared";
import { formatNumber } from "@/lib/utils";

const FIELDS: { key: Exclude<keyof FinancialConfig, "autopilot_settings">; label: string; hint?: string; step?: string }[] = [
  { key: "gross_margin_pct", label: "Planned gross margin", hint: "Fraction, e.g. 0.42", step: "0.0001" },
  { key: "cogs_pct", label: "COGS share", hint: "Fraction of list-price revenue", step: "0.0001" },
  { key: "promo_discount_pct", label: "Promo discount", hint: "Fraction off list price", step: "0.0001" },
  { key: "fixed_daily_overhead", label: "Fixed daily overhead", step: "0.01" },
  { key: "hourly_wage", label: "Hourly wage", step: "0.01" },
  { key: "customers_per_staff_hour", label: "Customers per staff-hour", step: "0.01" },
  { key: "min_staff_per_shift", label: "Min staff per shift", step: "1" },
  { key: "shift_hours", label: "Shift length (h)", step: "0.25" },
  { key: "operating_hours_per_day", label: "Operating hours / day", step: "0.25" },
  { key: "avg_transaction_value", label: "Avg transaction value", hint: "Optional if dataset has customers", step: "0.01" },
  { key: "avg_units_per_transaction", label: "Avg units / transaction", hint: "Needed for unit targets", step: "0.001" },
];

/** Mirrors AutopilotSettings defaults (backend app/schemas/autopilot.py) for stores without a config yet. */
const DEFAULT_AUTOPILOT: AutopilotSettings = {
  shifts: [
    { name: "Morning", traffic_share: 0.3 },
    { name: "Midday Peak", traffic_share: 0.45 },
    { name: "Evening", traffic_share: 0.25 },
  ],
  peak_hour_share: 0.14,
  max_staff_per_shift: null,
  overstaff_utilization: 0.55,
  surge_block_hours: 2,
  cold_start_min_days: 90,
  cohort_size: 10,
  basket_lookback_days: 182,
  anomaly_threshold: 0.35,
};

export default function StoresPage() {
  const stores = useStores();
  const [selected, setSelected] = useState<Store | null>(null);

  return (
    <>
      <PageHeader title="Stores" description="Store metadata and the unit economics used by P&L and workforce planning. Stores are created automatically from uploaded data." />
      <div className="grid gap-6 lg:grid-cols-5">
        <Card className="lg:col-span-2">
          <CardHeader><CardTitle>All stores</CardTitle><CardDescription>{stores.data?.total ?? 0} stores</CardDescription></CardHeader>
          <CardContent>
            {stores.data?.items.length ? (
              <Table>
                <THead><tr><TH>Code</TH><TH>Type</TH><TH>Competition</TH><TH>Config</TH></tr></THead>
                <tbody>
                  {stores.data.items.map((s) => (
                    <TR key={s.id} onClick={() => setSelected(s)} className={`cursor-pointer ${selected?.id === s.id ? "bg-muted" : ""}`}>
                      <TD className="font-medium">{s.code}</TD>
                      <TD>{[s.store_type, s.assortment].filter(Boolean).join(" / ") || "—"}</TD>
                      <TD>{s.competition_distance !== null ? `${formatNumber(s.competition_distance)} m` : "—"}</TD>
                      <TD>{s.has_financial_config ? <Badge tone="success">set</Badge> : <Badge tone="warning">missing</Badge>}</TD>
                    </TR>
                  ))}
                </tbody>
              </Table>
            ) : (
              <EmptyState title="No stores yet" description="Stores appear after a dataset with a store column is processed." />
            )}
          </CardContent>
        </Card>
        <div className="lg:col-span-3">
          {selected ? <ConfigForm key={selected.id} store={selected} /> : <EmptyState title="Select a store" description="Pick a store to edit its financial & staffing configuration." />}
        </div>
      </div>
    </>
  );
}

function ConfigForm({ store }: { store: Store }) {
  const qc = useQueryClient();
  const cfg = useQuery({
    queryKey: ["fin-config", store.id],
    queryFn: async () => {
      try {
        return await api.get<FinancialConfig>(`/stores/${store.id}/financial-config`);
      } catch (e) {
        if (e instanceof ApiError && e.status === 404) return null;
        throw e;
      }
    },
  });
  const save = useMutation({
    mutationFn: (body: Record<string, unknown>) => api.put(`/stores/${store.id}/financial-config`, body),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ["fin-config", store.id] });
      void qc.invalidateQueries({ queryKey: ["stores"] });
    },
  });

  function onSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const f = new FormData(e.currentTarget);
    const body: Record<string, unknown> = {};
    for (const { key } of FIELDS) {
      const v = String(f.get(key) ?? "").trim();
      body[key] = v === "" ? null : key === "min_staff_per_shift" ? Number(v) : v;
    }
    // Round-trip the Autopilot rules so a save here never resets them to defaults.
    const ap = cfg.data?.autopilot_settings ?? DEFAULT_AUTOPILOT;
    const maxStaff = String(f.get("ap_max_staff") ?? "").trim();
    body.autopilot_settings = {
      ...ap,
      max_staff_per_shift: maxStaff === "" ? null : Number(maxStaff),
      peak_hour_share: Number(f.get("ap_peak_share")) / 100,
      shifts: ap.shifts.map((s) => ({ ...s, traffic_share: Number(f.get(`ap_share_${s.name}`)) / 100 })),
    };
    save.mutate(body);
  }

  if (cfg.isLoading) return null;
  const current = cfg.data;
  return (
    <Card>
      <CardHeader>
        <CardTitle>Store {store.code} — financial & staffing</CardTitle>
        <CardDescription>Target: {store.target_type} · Currency: {store.currency}</CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={onSubmit} className="grid gap-4 md:grid-cols-2">
          {FIELDS.map((fd) => (
            <Field key={fd.key} label={fd.label} hint={fd.hint}>
              <Input
                name={fd.key}
                type="number"
                step={fd.step}
                min={0}
                defaultValue={current?.[fd.key] ?? ""}
                required={!["avg_transaction_value", "avg_units_per_transaction"].includes(fd.key)}
              />
            </Field>
          ))}
          <div className="md:col-span-2 border-t pt-4">
            <p className="text-sm font-medium">Store Autopilot rules</p>
            <p className="text-xs text-muted-foreground">Shift traffic split, peak hour and capacity used for shift plans and staffing warnings.</p>
          </div>
          {(current?.autopilot_settings ?? DEFAULT_AUTOPILOT).shifts.map((s) => (
            <Field key={s.name} label={`${s.name} share of footfall (%)`}>
              <Input name={`ap_share_${s.name}`} type="number" step="1" min={1} max={100} required defaultValue={Math.round(s.traffic_share * 100)} />
            </Field>
          ))}
          <Field label="Peak-hour share of daily footfall (%)">
            <Input name="ap_peak_share" type="number" step="1" min={1} max={100} required defaultValue={Math.round((current?.autopilot_settings ?? DEFAULT_AUTOPILOT).peak_hour_share * 100)} />
          </Field>
          <Field label="Max staff per shift" hint="Store capacity; blank = no limit">
            <Input name="ap_max_staff" type="number" step="1" min={1} defaultValue={(current?.autopilot_settings ?? DEFAULT_AUTOPILOT).max_staff_per_shift ?? ""} />
          </Field>
          <div className="md:col-span-2"><ErrorNotice error={save.error ?? cfg.error} /></div>
          <div className="md:col-span-2 flex items-center gap-3">
            <Button type="submit" disabled={save.isPending}>Save configuration</Button>
            {save.isSuccess ? <span className="text-xs text-success">Saved</span> : null}
          </div>
        </form>
      </CardContent>
    </Card>
  );
}
