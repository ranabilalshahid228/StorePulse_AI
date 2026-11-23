"use client";

import Link from "next/link";
import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowUpRight, CalendarDays, ChartNoAxesCombined, Download, Package, TrendingUp } from "lucide-react";

import { ForecastBandChart } from "@/components/charts";
import { EmptyState, ErrorNotice, JobProgress } from "@/components/shared";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, Field, Select } from "@/components/ui/primitives";
import { useDataset, useModelRuns, useStores } from "@/hooks/queries";
import { api } from "@/lib/api";
import type { ForecastPoint, ForecastRun } from "@/lib/types";
import { formatNumber, formatPct } from "@/lib/utils";

const HORIZONS = [
  { days: 1, label: "Next day" },
  { days: 7, label: "Next week" },
  { days: 30, label: "Next 30 days" },
];

interface ProductRanking {
  product_count: number;
  products: { item_id: string; forecast: number }[];
}

export default function ForecastsPage() {
  const [selectedModel, setSelectedModel] = useState("");
  const [selectedRun, setSelectedRun] = useState("");
  const [requestedDays, setRequestedDays] = useState<number>();
  const qc = useQueryClient();
  const models = useModelRuns();
  const stores = useStores();
  const available = models.data?.items.filter((run) => run.status === "succeeded") ?? [];
  const model = available.find((run) => run.id === selectedModel) ?? available[0];
  const dataset = useDataset(model?.dataset_id ?? "");
  const store = stores.data?.items.find((item) => item.id === model?.store_id);
  const target = dataset.data?.schema_mapping?.target_column ?? "mapped target";
  const create = useMutation({
    mutationFn: ({ days, modelId }: { days: number; modelId: string }) => api.post<{ id: string }>("/forecasts", {
      model_run_id: modelId,
      horizon_days: days,
    }),
    onSuccess: ({ id }) => {
      setSelectedRun(id);
      void qc.invalidateQueries({ queryKey: ["forecasts"] });
    },
  });
  const run = useQuery({
    queryKey: ["forecast-overview", selectedRun],
    queryFn: () => api.get<ForecastRun>(`/forecasts/${selectedRun}`),
    enabled: Boolean(selectedRun),
    refetchInterval: ({ state }) => {
      const status = state.data?.status;
      return status === "queued" || status === "running" ? 1500 : false;
    },
  });

  return (
    <div className="space-y-6">
      <section className="forecast-hero relative isolate overflow-hidden rounded-2xl bg-hero px-6 py-10 text-hero-foreground sm:px-9 sm:py-14">
        <div className="relative max-w-xl motion-safe:animate-[forecast-enter_450ms_ease-out_both]">
          <ChartNoAxesCombined aria-hidden className="mb-5 size-7 text-hero-accent" />
          <h1 className="text-3xl font-semibold tracking-tight sm:text-5xl">Sales forecasts</h1>
          <p className="mt-3 max-w-md text-sm leading-relaxed text-hero-foreground/85 sm:text-base">
            See expected demand and the products forecast to lead over the next day, week, or month.
          </p>
        </div>
      </section>

      {models.error ? (
        <div className="space-y-3">
          <ErrorNotice error={models.error} />
          <Button variant="outline" disabled={models.isFetching} onClick={() => void models.refetch()}>
            Retry loading models
          </Button>
        </div>
      ) : models.isLoading ? (
        <p role="status" className="text-sm text-muted-foreground">Loading trained models…</p>
      ) : !model ? (
        <EmptyState title="Train a model to get your first forecast"
          description="Upload daily sales history, then train a model in Model Lab."
          action={<Button asChild><Link href="/model-lab">Open Model Lab <ArrowUpRight /></Link></Button>} />
      ) : (
        <>
          <Card>
            <CardContent className="space-y-4 pt-5">
              <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_auto] lg:items-end">
                <Field label="Forecast model">
                  <Select value={model.id} onChange={(event) => {
                    setSelectedModel(event.target.value);
                    setSelectedRun("");
                    setRequestedDays(undefined);
                    create.reset();
                  }} disabled={create.isPending}>
                    {available.map((item) => (
                      <option key={item.id} value={item.id}>
                        {item.name || item.model_family} · {item.horizon_days} days · {item.created_at.slice(0, 10)}
                      </option>
                    ))}
                  </Select>
                </Field>
                <div className="flex flex-wrap gap-2">
                  {HORIZONS.map(({ days, label }) => (
                    <Button key={days} type="button" variant={requestedDays === days ? "default" : "outline"}
                      disabled={create.isPending || days > model.horizon_days}
                      onClick={() => {
                        setRequestedDays(days);
                        setSelectedRun("");
                        create.mutate({ days, modelId: model.id });
                      }}>
                      <CalendarDays />{label}
                    </Button>
                  ))}
                </div>
              </div>
              <p className="text-sm text-muted-foreground">
                {dataset.data?.name ?? "Training dataset"} · {store?.name ?? (model.store_id ? "Single-store model" : "All stores in this model")}
                {dataset.data?.date_max ? ` · History through ${dataset.data.date_max}` : ""}
                {model.metrics.wape != null ? ` · Validation WAPE ${formatPct(model.metrics.wape, 2)}` : ""}
              </p>
              {model.horizon_days < 30 && (
                <p className="text-xs text-muted-foreground">This model supports up to {model.horizon_days} days. Train a 30-day model to enable the monthly forecast.</p>
              )}
            </CardContent>
          </Card>
          <ErrorNotice error={create.error ?? run.error} />
          {(create.isPending || run.isLoading) && <p role="status" className="text-sm text-muted-foreground">Starting forecast…</p>}
          {run.data && run.data.status !== "succeeded" && (
            <Card><CardContent className="space-y-2 pt-5">
              <JobProgress status={run.data.status} pct={run.data.progress_pct} stage={run.data.stage} />
              {run.data.error && <ErrorNotice error={new Error(run.data.error)} />}
            </CardContent></Card>
          )}
          {run.data?.status === "succeeded" && <ForecastResults key={run.data.id} forecast={run.data} target={target} />}
        </>
      )}
    </div>
  );
}

function ForecastResults({ forecast, target }: { forecast: ForecastRun; target: string }) {
  const points = useQuery({
    queryKey: ["forecast-series", forecast.id],
    queryFn: () => api.get<{ points: ForecastPoint[] }>(`/forecasts/${forecast.id}/series`),
  });
  const products = useQuery({
    queryKey: ["forecast-products", forecast.id],
    queryFn: () => api.get<ProductRanking>(`/forecasts/${forecast.id}/products`),
  });
  const download = useMutation({
    mutationFn: () => api.download(`/reports/forecasts/${forecast.id}/forecast.csv`, "storepulse-forecast.csv"),
  });
  const total = forecast.summary.horizon_totals?.[`h${forecast.horizon_days}`]?.yhat;
  const daily = (points.data?.points ?? []).map((point) => ({
    ...point,
    band: point.yhat_lower != null && point.yhat_upper != null ? [point.yhat_lower, point.yhat_upper] : null,
  }));
  const ranking = products.data?.products ?? [];
  const topDemand = ranking[0]?.forecast ?? 0;
  const singleDay = daily.length === 1 ? daily[0] : undefined;
  const peak = daily.reduce<(typeof daily)[number] | undefined>(
    (best, day) => (!best || day.yhat > best.yhat ? day : best),
    undefined,
  );

  return (
    <div className="grid gap-5 motion-safe:animate-[forecast-enter_350ms_ease-out_both] xl:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2 text-primary"><TrendingUp aria-hidden className="size-4" /> <CardTitle>Expected demand</CardTitle></div>
          <CardDescription>{forecast.summary.start_date} → {forecast.summary.end_date}</CardDescription>
          <p className="mt-2 text-3xl font-semibold tabular-nums">{formatNumber(total, 2)} <span className="text-sm font-normal text-muted-foreground">{target}</span></p>
        </CardHeader>
        <CardContent>
          <ErrorNotice error={points.error ?? download.error} />
          {points.isLoading && <p role="status" className="text-sm text-muted-foreground">Loading daily figures…</p>}
          {daily.length > 1 && <ForecastBandChart data={daily} />}
          {daily.length > 1 && peak && (
            <p className="mt-2 text-sm text-muted-foreground">
              Highest forecast day: {peak.ds} · {formatNumber(peak.yhat, 2)} {target}
            </p>
          )}
          {singleDay?.band && (
            <p className="text-sm text-muted-foreground">Forecast range {singleDay.band.map((value) => formatNumber(value, 2)).join("–")}</p>
          )}
          <Button className="mt-4" variant="outline" size="sm" disabled={download.isPending} onClick={() => download.mutate()}>
            <Download /> Download CSV
          </Button>
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2 text-primary"><Package aria-hidden className="size-4" /> <CardTitle>Top products</CardTitle></div>
          <CardDescription>Ranked by predicted {target} across the forecast window.</CardDescription>
        </CardHeader>
        <CardContent>
          <ErrorNotice error={products.error} />
          {products.isLoading && <p role="status" className="text-sm text-muted-foreground">Ranking products…</p>}
          {products.data && !ranking.length && (
            <p className="text-sm leading-relaxed text-muted-foreground">
              This forecast has store totals. To rank products, train on sales history with a product ID or SKU
              mapped to the item column. Use units sold as the target to rank by volume.
            </p>
          )}
          {ranking.length > 0 && (
            <ol className="space-y-4" aria-label="Products ranked by predicted demand">
              {ranking.map((item) => (
                <li key={item.item_id} className="space-y-1.5">
                  <div className="flex justify-between gap-3 text-sm">
                    <span className="break-all font-medium">{item.item_id}</span>
                    <span className="whitespace-nowrap tabular-nums">{formatNumber(item.forecast, 2)}</span>
                  </div>
                  <div className="h-1.5 overflow-hidden rounded-full bg-muted">
                    <div className="h-full origin-left rounded-full bg-primary motion-safe:animate-[forecast-bar_650ms_ease-out_both]"
                      style={{ width: `${topDemand > 0 ? item.forecast / topDemand * 100 : 0}%` }} />
                  </div>
                </li>
              ))}
            </ol>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
