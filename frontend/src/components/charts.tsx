"use client";

import {
  Area,
  Bar,
  Cell,
  BarChart,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ReferenceLine,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { formatCompact } from "@/lib/utils";

const axis = { fontSize: 11, fill: "var(--muted-foreground)" };
const grid = <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" vertical={false} />;
const tooltipStyle = {
  contentStyle: {
    background: "var(--card)",
    border: "1px solid var(--border)",
    borderRadius: 8,
    fontSize: 12,
  },
};

export interface SeriesDef {
  key: string;
  label: string;
  color?: string;
  dashed?: boolean;
}

export function TimeSeriesChart({ data, series, height = 280 }: { data: object[]; series: SeriesDef[]; height?: number }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
        {grid}
        <XAxis dataKey="ds" tick={axis} tickLine={false} axisLine={false} minTickGap={32} />
        <YAxis tick={axis} tickLine={false} axisLine={false} tickFormatter={formatCompact} width={48} />
        <Tooltip {...tooltipStyle} />
        {series.length > 1 ? <Legend wrapperStyle={{ fontSize: 12 }} /> : null}
        {series.map((s, i) => (
          <Line
            key={s.key}
            dataKey={s.key}
            name={s.label}
            stroke={s.color ?? `var(--chart-${(i % 7) + 1})`}
            strokeWidth={1.8}
            strokeDasharray={s.dashed ? "4 3" : undefined}
            dot={false}
            connectNulls
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

/** Forecast with an uncertainty band (yhat_lower..yhat_upper). */
export function ForecastBandChart({ data, height = 300 }: { data: object[]; height?: number }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
        {grid}
        <XAxis dataKey="ds" tick={axis} tickLine={false} axisLine={false} minTickGap={32} />
        <YAxis tick={axis} tickLine={false} axisLine={false} tickFormatter={formatCompact} width={48} />
        <Tooltip {...tooltipStyle} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        <Area dataKey="band" name="P10–P90 band" stroke="none" fill="var(--chart-1)" fillOpacity={0.15} />
        <Line dataKey="yhat" name="Forecast" stroke="var(--chart-1)" strokeWidth={2} dot={false} />
        <Line dataKey="baseline_yhat" name="Baseline" stroke="var(--chart-5)" strokeDasharray="4 3" dot={false} connectNulls />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

/** Autopilot revenue: P10–P90 band, P50, scenario baseline and break-even revenue. */
export function RevenuePlanChart({ data, height = 300, showBand }: { data: object[]; height?: number; showBand: boolean }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
        {grid}
        <XAxis dataKey="ds" tick={axis} tickLine={false} axisLine={false} minTickGap={24} />
        <YAxis tick={axis} tickLine={false} axisLine={false} tickFormatter={formatCompact} width={48} />
        <Tooltip {...tooltipStyle} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        {showBand && <Area dataKey="band" name="P10–P90" stroke="none" fill="var(--chart-1)" fillOpacity={0.15} />}
        <Line dataKey="revenue_p50" name="Revenue P50" stroke="var(--chart-1)" strokeWidth={2} dot={false} />
        <Line dataKey="baseline_revenue_p50" name="Baseline" stroke="var(--chart-5)" strokeDasharray="4 3" dot={false} connectNulls />
        <Line dataKey="break_even_revenue" name="Break-even" stroke="var(--chart-4)" strokeDasharray="2 3" strokeWidth={1.4} dot={false} />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

export function SimpleBarChart({
  data,
  xKey,
  series,
  height = 240,
  stacked = false,
}: {
  data: object[];
  xKey: string;
  series: SeriesDef[];
  height?: number;
  stacked?: boolean;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
        {grid}
        <XAxis dataKey={xKey} tick={axis} tickLine={false} axisLine={false} minTickGap={16} />
        <YAxis tick={axis} tickLine={false} axisLine={false} tickFormatter={formatCompact} width={48} />
        <Tooltip {...tooltipStyle} cursor={{ fill: "var(--muted)" }} />
        {series.length > 1 ? <Legend wrapperStyle={{ fontSize: 12 }} /> : null}
        {series.map((s, i) => (
          <Bar
            key={s.key}
            dataKey={s.key}
            name={s.label}
            stackId={stacked ? "stack" : undefined}
            fill={s.color ?? `var(--chart-${(i % 7) + 1})`}
            radius={stacked && i < series.length - 1 ? [0, 0, 0, 0] : [3, 3, 0, 0]}
          />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Ranked horizontal bars (e.g. feature importance). */
export function HorizontalBarChart({
  data,
  labelKey,
  valueKey,
  height,
  valueFormatter = formatCompact,
}: {
  data: object[];
  labelKey: string;
  valueKey: string;
  height?: number;
  valueFormatter?: (v: number) => string;
}) {
  const h = height ?? Math.max(160, data.length * 22 + 24);
  return (
    <ResponsiveContainer width="100%" height={h}>
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 16, left: 8, bottom: 0 }}>
        <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" horizontal={false} />
        <XAxis type="number" tick={axis} tickLine={false} axisLine={false} tickFormatter={valueFormatter} />
        <YAxis type="category" dataKey={labelKey} tick={axis} tickLine={false} axisLine={false} width={150} interval={0} />
        <Tooltip {...tooltipStyle} cursor={{ fill: "var(--muted)" }} formatter={(v) => valueFormatter(Number(v))} />
        <Bar dataKey={valueKey} fill="var(--chart-1)" radius={[0, 3, 3, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export interface WaterfallStep {
  label: string;
  value: number;
  /** Totals are drawn from zero; other steps float from the running total. */
  total?: boolean;
}

/** P&L bridge: revenue → deductions → profit. Positive steps use --chart-1, negative --chart-4. */
export function WaterfallChart({
  steps,
  height = 300,
  valueFormatter = formatCompact,
}: {
  steps: WaterfallStep[];
  height?: number;
  valueFormatter?: (v: number) => string;
}) {
  let running = 0;
  // Floating range bars [low, high] draw correctly even when the running total crosses zero (losses).
  const data = steps.map((s) => {
    const start = s.total ? 0 : running;
    const end = s.total ? s.value : running + s.value;
    running = end;
    const kind = s.total ? "total" : s.value >= 0 ? "up" : "down";
    return { label: s.label, range: [Math.min(start, end), Math.max(start, end)], value: s.value, kind };
  });
  const fill = (k: string) => (k === "total" ? "var(--chart-2)" : k === "up" ? "var(--chart-1)" : "var(--chart-4)");
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
        {grid}
        <XAxis dataKey="label" tick={axis} tickLine={false} axisLine={false} interval={0} />
        <YAxis tick={axis} tickLine={false} axisLine={false} tickFormatter={valueFormatter} width={56} />
        <ReferenceLine y={0} stroke="var(--border)" />
        <Tooltip
          {...tooltipStyle}
          cursor={{ fill: "var(--muted)" }}
          formatter={(_v, _n, item) => [valueFormatter((item.payload as { value: number }).value), "Amount"]}
        />
        <Bar dataKey="range" radius={[3, 3, 3, 3]} isAnimationActive={false}>
          {data.map((d) => (
            <Cell key={d.label} fill={fill(d.kind)} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
