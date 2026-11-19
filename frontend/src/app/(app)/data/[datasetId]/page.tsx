"use client";

import { use, useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { HealthReport, SchemaMapping } from "@/lib/types";
import { useDataset, useDatasets } from "@/hooks/queries";
import { JoinsEditor } from "@/components/joins-editor";
import { Button } from "@/components/ui/button";
import { Badge, Card, CardContent, CardDescription, CardHeader, CardTitle, Field, Input, Select, Skeleton, Table, TD, TH, THead, TR } from "@/components/ui/primitives";
import { ErrorNotice, JobProgress, Kpi, PageHeader, StatusBadge } from "@/components/shared";
import { SimpleBarChart, TimeSeriesChart } from "@/components/charts";
import { formatCompact, formatPct } from "@/lib/utils";

const DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const SEVERITY_TONE = { critical: "danger", warning: "warning", info: "info" } as const;

export default function DatasetPage({ params }: { params: Promise<{ datasetId: string }> }) {
  const { datasetId } = use(params);
  const { data: ds, error } = useDataset(datasetId);

  if (error) return <ErrorNotice error={error} />;
  if (!ds) return <Skeleton className="h-64" />;

  return (
    <>
      <PageHeader
        title={ds.name}
        description={`${ds.source_filename} · ${(ds.size_bytes / 1e6).toFixed(1)} MB`}
        actions={<JobProgress status={ds.status} pct={ds.progress_pct} stage={ds.stage} />}
      />
      {ds.error ? <div className="mb-6"><ErrorNotice error={new Error(ds.error)} /></div> : null}
      {ds.role === "auxiliary" ? (
        <p className="mb-6 text-sm text-muted-foreground">Side table — join it into a sales history from that dataset&apos;s mapping form.</p>
      ) : null}
      {ds.role === "primary" && (ds.status === "mapping_required" || ds.status === "failed") ? (
        <MappingForm datasetId={ds.id} columns={(ds.column_profile.columns ?? []).map((c) => c.name)} suggested={ds.schema_mapping ?? ds.suggested_mapping} />
      ) : null}
      {ds.role === "primary" && ds.status === "ready" ? <HealthAndEda datasetId={ds.id} /> : null}
      <ColumnProfileCard columns={ds.column_profile.columns ?? []} />
    </>
  );
}

function MappingForm({ datasetId, columns, suggested }: { datasetId: string; columns: string[]; suggested: Partial<SchemaMapping> }) {
  const qc = useQueryClient();
  const datasets = useDatasets();
  const auxiliary = (datasets.data?.items ?? []).filter((d) => d.role === "auxiliary");
  const [m, setM] = useState<SchemaMapping>({
    layout: "long",
    known_future_columns: [],
    categorical_columns: [],
    joins: [],
    helper_columns: [],
    ...suggested,
  } as SchemaMapping);
  useEffect(() => setM((prev) => ({ ...prev, ...suggested }) as SchemaMapping), [suggested]);

  const save = useMutation({
    mutationFn: () => api.put(`/datasets/${datasetId}/mapping`, m),
    onSuccess: () => void qc.invalidateQueries({ queryKey: ["dataset", datasetId] }),
  });

  const joined = (m.joins ?? []).flatMap((j) => j.columns);
  // Wide files only carry id columns; covariates there must come from joins.
  const covariatePool = m.layout === "long" ? [...columns, ...joined] : joined;
  const categoricalPool = m.layout === "long" ? [...columns, ...joined] : [...(m.id_columns ?? []), ...joined];

  const single = (key: keyof SchemaMapping, label: string, required = false) => (
    <Field label={label}>
      <Select value={(m[key] as string | null) ?? ""} onChange={(e) => setM({ ...m, [key]: e.target.value || null })} required={required}>
        <option value="">— none —</option>
        {columns.map((c) => <option key={c} value={c}>{c}</option>)}
      </Select>
    </Field>
  );
  const multi = (
    key: "known_future_columns" | "categorical_columns" | "helper_columns" | "id_columns",
    label: string,
    hint: string,
    pool: string[],
  ) => (
    <Field label={label} hint={hint}>
      <div className="flex flex-wrap gap-1.5">
        {pool.length ? null : <span className="text-xs text-muted-foreground">No columns available yet.</span>}
        {[...new Set(pool)].map((c) => {
          const current = m[key] ?? [];
          const on = current.includes(c);
          return (
            <button
              type="button"
              key={c}
              onClick={() => setM({ ...m, [key]: on ? current.filter((x) => x !== c) : [...current, c] })}
              className={`rounded-full border px-2.5 py-0.5 text-xs ${on ? "border-primary bg-primary text-primary-foreground" : "hover:bg-muted"}`}
            >
              {c}
            </button>
          );
        })}
      </div>
    </Field>
  );
  const role = (key: "promo_column" | "open_column" | "price_column", label: string) => (
    <Field label={label}>
      <Select value={m[key] ?? ""} onChange={(e) => setM({ ...m, [key]: e.target.value || null })}>
        <option value="">— none —</option>
        {m.known_future_columns.map((c) => <option key={c}>{c}</option>)}
      </Select>
    </Field>
  );

  return (
    <Card className="mb-6">
      <CardHeader>
        <CardTitle>Confirm schema mapping</CardTitle>
        <CardDescription>We pre-filled this from the file&apos;s columns. Confirm to convert to Parquet and run diagnostics.</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-5">
        <Field label="Layout">
          <Select value={m.layout} onChange={(e) => setM({ ...m, layout: e.target.value as "long" | "wide" })}>
            <option value="long">Long (one row per series per day)</option>
            <option value="wide">Wide (one column per day, e.g. M5 d_1…d_N)</option>
          </Select>
        </Field>
        {m.layout === "long" ? (
          <div className="grid gap-4 md:grid-cols-3">
            {single("date_column", "Date column", true)}
            {single("target_column", "Target column", true)}
            {single("store_column", "Store column")}
            {single("item_column", "Item / SKU column")}
            {single("customers_column", "Customers column")}
          </div>
        ) : (
          <>
            <div className="grid gap-4 md:grid-cols-4">
              {single("store_column", "Store column")}
              {single("item_column", "Item column")}
              <Field label="Day-column prefix"><Input value={m.wide_value_prefix ?? ""} onChange={(e) => setM({ ...m, wide_value_prefix: e.target.value })} /></Field>
              <Field label="Date of day 1" hint="M5: 2011-01-29"><Input type="date" value={m.wide_start_date ?? ""} onChange={(e) => setM({ ...m, wide_start_date: e.target.value })} /></Field>
            </div>
            {multi("id_columns", "Id columns to keep", "Hierarchy columns such as dept / category / state.", columns)}
          </>
        )}
        <Field label="Only these stores" hint="Comma-separated store codes; leave empty for all. Useful for very large files.">
          <Input
            defaultValue={(m.store_filter ?? []).join(", ")}
            placeholder="e.g. CA_1, TX_2"
            onBlur={(e) => {
              const v = e.target.value.split(",").map((s) => s.trim()).filter(Boolean);
              setM({ ...m, store_filter: v.length ? v : null });
            }}
          />
        </Field>
        <div className="flex flex-col gap-2">
          <p className="text-sm font-medium">Side-table joins</p>
          <JoinsEditor joins={m.joins ?? []} auxiliary={auxiliary} onChange={(joins) => setM({ ...m, joins })} />
        </div>
        {multi("known_future_columns", "Known-future covariates", "Numeric/flag columns known ahead of time (promo, open, holidays, price).", covariatePool)}
        {multi("categorical_columns", "Categorical covariates", "Categories such as holiday type, department or event name.", categoricalPool)}
        {joined.length ? multi("helper_columns", "Join-only helper columns", "Used only as keys for later joins, then dropped (e.g. M5 wm_yr_wk).", joined) : null}
        <div className="grid gap-4 md:grid-cols-3">
          {role("promo_column", "Promo flag (for P&L)")}
          {role("open_column", "Open flag (for staffing)")}
          {role("price_column", "Unit price (values units forecasts)")}
        </div>
        <ErrorNotice error={save.error} />
        <div><Button onClick={() => save.mutate()} disabled={save.isPending}>Confirm & process</Button></div>
      </CardContent>
    </Card>
  );
}

function HealthAndEda({ datasetId }: { datasetId: string }) {
  const health = useQuery({
    queryKey: ["dataset-health", datasetId],
    queryFn: () => api.get<{ report: HealthReport | null }>(`/datasets/${datasetId}/health`),
  });
  const r = health.data?.report;
  if (!r) return <Skeleton className="mb-6 h-48" />;
  const weekday = Object.entries(r.weekday_profile).map(([k, v]) => ({ dow: DOW[Number(k)] ?? k, mean: v }));

  return (
    <>
      <div className="mb-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Kpi label="Health score" value={`${r.score}/100`} sub={`${r.issues.length} findings`} />
        <Kpi label="Rows" value={formatCompact(r.row_count)} sub={`${r.series_count} series · ${r.store_count} stores`} />
        <Kpi label="Missing series-days" value={formatCompact(r.missing_series_days)} sub={`${r.date_min} → ${r.date_max}`} />
        <Kpi label="Zero share" value={formatPct(r.zero_fraction)} sub={`${formatCompact(r.outlier_count)} robust outliers`} />
      </div>
      <div className="mb-6 grid gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader><CardTitle>Total target over time</CardTitle></CardHeader>
          <CardContent><TimeSeriesChart data={r.daily_total} series={[{ key: "y", label: "Target" }]} /></CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle>Weekday profile</CardTitle><CardDescription>Mean target per weekday</CardDescription></CardHeader>
          <CardContent><SimpleBarChart data={weekday} xKey="dow" series={[{ key: "mean", label: "Mean" }]} /></CardContent>
        </Card>
      </div>
      <Card className="mb-6">
        <CardHeader><CardTitle>Diagnostics</CardTitle></CardHeader>
        <CardContent>
          {r.issues.length ? (
            <ul className="flex flex-col gap-2">
              {r.issues.map((i) => (
                <li key={i.code} className="flex items-start gap-3 text-sm">
                  <Badge tone={SEVERITY_TONE[i.severity]}>{i.severity}</Badge>
                  <span>{i.message}</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-muted-foreground">No issues detected.</p>
          )}
        </CardContent>
      </Card>
    </>
  );
}

function ColumnProfileCard({ columns }: { columns: { name: string; inferred_type: string; null_fraction: number; sample_values: unknown[] }[] }) {
  if (!columns.length) return null;
  return (
    <Card>
      <CardHeader><CardTitle>Column profile</CardTitle><CardDescription>Inferred from a sample of the raw upload</CardDescription></CardHeader>
      <CardContent>
        <Table>
          <THead><tr><TH>Column</TH><TH>Type</TH><TH>Nulls</TH><TH>Samples</TH></tr></THead>
          <tbody>
            {columns.map((c) => (
              <TR key={c.name}>
                <TD className="font-medium">{c.name}</TD>
                <TD><StatusBadge status={c.inferred_type} /></TD>
                <TD>{formatPct(c.null_fraction)}</TD>
                <TD className="max-w-md truncate text-muted-foreground">{c.sample_values.map(String).join(", ")}</TD>
              </TR>
            ))}
          </tbody>
        </Table>
      </CardContent>
    </Card>
  );
}
