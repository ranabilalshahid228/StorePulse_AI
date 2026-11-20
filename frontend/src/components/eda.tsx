"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { DatasetEda, EdaStoreRow } from "@/lib/types";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, Field, Input, Skeleton, Table, TD, TH, THead, TR } from "@/components/ui/primitives";
import { ErrorNotice, Kpi } from "@/components/shared";
import { HorizontalBarChart, SimpleBarChart, TimeSeriesChart } from "@/components/charts";
import { cn, formatCompact, formatNumber, formatPct } from "@/lib/utils";

const lift = (v: number | null | undefined) => (v === null || v === undefined ? "—" : `${v >= 0 ? "+" : ""}${formatPct(v)}`);

export function EdaStudio({ datasetId }: { datasetId: string }) {
  const eda = useQuery({ queryKey: ["dataset-eda", datasetId], queryFn: () => api.get<DatasetEda>(`/datasets/${datasetId}/eda`), staleTime: 5 * 60_000 });
  const [store, setStore] = useState<string>("");

  if (eda.isLoading) {
    return (
      <div className="mb-6 grid gap-6 lg:grid-cols-2">
        <Skeleton className="h-72" />
        <Skeleton className="h-72" />
      </div>
    );
  }
  if (eda.error) return <div className="mb-6"><ErrorNotice error={eda.error} /></div>;
  const e = eda.data;
  if (!e) return null;
  const allStores = [...e.stores.top, ...e.stores.bottom];
  const drill = store || e.stores.top[0]?.store_code || "";

  return (
    <section className="mb-6 flex flex-col gap-6">
      <div>
        <h2 className="text-base font-semibold">Exploratory analysis</h2>
        <p className="text-sm text-muted-foreground">
          Store-day totals{e.columns.open ? ", open days only" : ""}. Lifts compare mean daily target with vs without the effect.
        </p>
      </div>
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Kpi label="Stores" value={formatNumber(e.store_count)} sub={`mean ${formatCompact(e.open_day_mean)} per open store-day`} />
        <Kpi label="Promo lift" value={lift(e.promo?.lift)} sub={e.promo ? `${formatPct(e.promo.promo_day_share, 0)} of open days on promo` : "no promo column mapped"} />
        <Kpi label="Footfall lift on promo" value={lift(e.promo?.footfall_lift)} sub={e.promo?.basket_lift != null ? `basket ${lift(e.promo.basket_lift)}` : "needs a customers column"} />
        <Kpi
          label="Store spread (P10 → P90)"
          value={`${formatCompact(e.stores.distribution.p10)} → ${formatCompact(e.stores.distribution.p90)}`}
          sub={`median ${formatCompact(e.stores.distribution.p50)} per open day`}
        />
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader><CardTitle>Monthly seasonality</CardTitle><CardDescription>Index vs the average open store-day (1.0 = average)</CardDescription></CardHeader>
          <CardContent><SimpleBarChart data={e.monthly} xKey="month" series={[{ key: "index", label: "Seasonality index" }]} /></CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Weekday × promo</CardTitle>
            <CardDescription>Mean daily target{e.promo ? " with and without promo" : ""}; open share shown in the table below</CardDescription>
          </CardHeader>
          <CardContent>
            <SimpleBarChart
              data={e.weekday}
              xKey="dow"
              series={e.promo ? [{ key: "promo", label: "Promo" }, { key: "no_promo", label: "No promo", color: "var(--chart-5)" }] : [{ key: "mean", label: "Mean" }]}
            />
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader><CardTitle>Sales trend</CardTitle><CardDescription>Monthly total across all stores</CardDescription></CardHeader>
        <CardContent><TimeSeriesChart data={e.trend} series={[{ key: "y", label: "Monthly total" }]} height={240} /></CardContent>
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader><CardTitle>Weekday profile</CardTitle></CardHeader>
          <CardContent>
            <Table>
              <THead><tr><TH>Day</TH><TH className="text-right">Open share</TH><TH className="text-right">Mean</TH>{e.promo ? <TH className="text-right">Promo lift</TH> : null}</tr></THead>
              <tbody>
                {e.weekday.map((w) => (
                  <TR key={w.dow}>
                    <TD>{w.dow}</TD>
                    <TD className="text-right tabular-nums">{formatPct(w.open_share, 0)}</TD>
                    <TD className="text-right tabular-nums">{formatCompact(w.mean)}</TD>
                    {e.promo ? <TD className="text-right tabular-nums">{lift(w.promo_lift)}</TD> : null}
                  </TR>
                ))}
              </tbody>
            </Table>
          </CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle>Holiday effects</CardTitle><CardDescription>Open-day mean vs normal days</CardDescription></CardHeader>
          <CardContent>
            {e.holidays.length ? (
              <Table>
                <THead><tr><TH>Holiday</TH><TH className="text-right">Days</TH><TH className="text-right">Open share</TH><TH className="text-right">Mean</TH><TH className="text-right">Lift</TH></tr></THead>
                <tbody>
                  {e.holidays.map((h) => (
                    <TR key={h.holiday}>
                      <TD>{h.holiday}</TD>
                      <TD className="text-right tabular-nums">{formatNumber(h.days)}</TD>
                      <TD className="text-right tabular-nums">{h.open_share !== undefined ? formatPct(h.open_share, 0) : "—"}</TD>
                      <TD className="text-right tabular-nums">{formatCompact(h.mean)}</TD>
                      <TD className={cn("text-right tabular-nums", (h.lift ?? 0) < 0 && "text-destructive")}>{lift(h.lift)}</TD>
                    </TR>
                  ))}
                </tbody>
              </Table>
            ) : (
              <p className="text-sm text-muted-foreground">No holiday columns (SchoolHoliday / StateHoliday) in this dataset.</p>
            )}
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-6 lg:grid-cols-5">
        <Card className="lg:col-span-2">
          <CardHeader><CardTitle>Top stores</CardTitle><CardDescription>Mean target per open day</CardDescription></CardHeader>
          <CardContent>
            <HorizontalBarChart data={e.stores.top.map((s) => ({ ...s, label: `Store ${s.store_code}` }))} labelKey="label" valueKey="mean" />
          </CardContent>
        </Card>
        <Card className="lg:col-span-3">
          <CardHeader><CardTitle>Store comparison</CardTitle><CardDescription>Click a store to drill into its history</CardDescription></CardHeader>
          <CardContent className="max-h-[420px] overflow-auto">
            <StoreTable rows={e.stores.top} active={drill} onPick={setStore} />
            {e.stores.bottom.length ? (
              <>
                <p className="mt-4 mb-1 text-xs font-medium text-muted-foreground">Lowest stores</p>
                <StoreTable rows={e.stores.bottom} active={drill} onPick={setStore} />
              </>
            ) : null}
          </CardContent>
        </Card>
      </div>

      {e.segments.length ? (
        <Card>
          <CardHeader><CardTitle>Segments</CardTitle><CardDescription>Store type × assortment from imported store metadata</CardDescription></CardHeader>
          <CardContent>
            <Table>
              <THead><tr><TH>Segment</TH><TH className="text-right">Stores</TH><TH className="text-right">Mean / open day</TH><TH className="text-right">Basket</TH><TH className="text-right">Median promo lift</TH></tr></THead>
              <tbody>
                {e.segments.map((s) => (
                  <TR key={s.segment}>
                    <TD>{s.segment}</TD>
                    <TD className="text-right tabular-nums">{s.stores}</TD>
                    <TD className="text-right tabular-nums">{formatCompact(s.mean)}</TD>
                    <TD className="text-right tabular-nums">{formatNumber(s.basket, 2)}</TD>
                    <TD className="text-right tabular-nums">{lift(s.promo_lift)}</TD>
                  </TR>
                ))}
              </tbody>
            </Table>
          </CardContent>
        </Card>
      ) : null}

      <StoreDrilldown datasetId={datasetId} storeCode={drill} onChange={setStore} known={allStores.map((s) => s.store_code)} />
    </section>
  );
}

function StoreTable({ rows, active, onPick }: { rows: EdaStoreRow[]; active: string; onPick: (code: string) => void }) {
  return (
    <Table>
      <THead><tr><TH>Store</TH><TH>Type</TH><TH className="text-right">Mean</TH><TH className="text-right">Footfall</TH><TH className="text-right">Basket</TH><TH className="text-right">Promo lift</TH><TH className="text-right">Open days</TH></tr></THead>
      <tbody>
        {rows.map((s) => (
          <TR key={s.store_code} onClick={() => onPick(s.store_code)} className={cn("cursor-pointer", active === s.store_code && "bg-muted")}>
            <TD className="font-medium">{s.store_code}</TD>
            <TD className="text-muted-foreground">{[s.store_type, s.assortment].filter(Boolean).join(" / ") || "—"}</TD>
            <TD className="text-right tabular-nums">{formatCompact(s.mean)}</TD>
            <TD className="text-right tabular-nums">{formatNumber(s.footfall)}</TD>
            <TD className="text-right tabular-nums">{formatNumber(s.basket, 2)}</TD>
            <TD className="text-right tabular-nums">{lift(s.promo_lift)}</TD>
            <TD className="text-right tabular-nums">{formatNumber(s.open_days)}</TD>
          </TR>
        ))}
      </tbody>
    </Table>
  );
}

function StoreDrilldown({ datasetId, storeCode, onChange, known }: { datasetId: string; storeCode: string; onChange: (c: string) => void; known: string[] }) {
  const series = useQuery({
    queryKey: ["dataset-series", datasetId, storeCode],
    enabled: !!storeCode,
    queryFn: () => api.get<{ points: { ds: string; y: number; customers?: number }[] }>(`/datasets/${datasetId}/series?store_code=${encodeURIComponent(storeCode)}`),
  });
  const hasCustomers = series.data?.points.some((p) => p.customers !== undefined && p.customers !== null);
  return (
    <Card>
      <CardHeader className="flex flex-row flex-wrap items-end justify-between gap-4">
        <div>
          <CardTitle>Store history</CardTitle>
          <CardDescription>Daily target{hasCustomers ? " and customers" : ""} for one store</CardDescription>
        </div>
        <Field label="Store code">
          <Input list="eda-stores" className="w-40" value={storeCode} onChange={(e) => onChange(e.target.value.trim())} />
        </Field>
        <datalist id="eda-stores">{known.map((c) => <option key={c} value={c} />)}</datalist>
      </CardHeader>
      <CardContent>
        <ErrorNotice error={series.error} />
        {series.isLoading ? (
          <Skeleton className="h-64" />
        ) : series.data?.points.length ? (
          <TimeSeriesChart
            data={series.data.points}
            series={[{ key: "y", label: "Target" }, ...(hasCustomers ? [{ key: "customers", label: "Customers", color: "var(--chart-2)" }] : [])]}
          />
        ) : (
          <p className="text-sm text-muted-foreground">{storeCode ? "No rows for this store." : "Pick a store."}</p>
        )}
      </CardContent>
    </Card>
  );
}
