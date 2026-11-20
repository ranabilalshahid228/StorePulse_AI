"use client";

import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
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
            stroke={s.color ?? `var(--chart-${(i % 5) + 1})`}
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
export function RevenuePlanChart({ data, height = 300 }: { data: object[]; height?: number }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
        {grid}
        <XAxis dataKey="ds" tick={axis} tickLine={false} axisLine={false} minTickGap={24} />
        <YAxis tick={axis} tickLine={false} axisLine={false} tickFormatter={formatCompact} width={48} />
        <Tooltip {...tooltipStyle} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        <Area dataKey="band" name="P10–P90" stroke="none" fill="var(--chart-1)" fillOpacity={0.15} />
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
            fill={s.color ?? `var(--chart-${(i % 5) + 1})`}
            radius={stacked && i < series.length - 1 ? [0, 0, 0, 0] : [3, 3, 0, 0]}
          />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}
