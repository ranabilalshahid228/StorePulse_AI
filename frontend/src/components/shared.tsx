"use client";

import type { ReactNode } from "react";
import { AlertTriangle, Inbox } from "lucide-react";
import { ApiError } from "@/lib/api";
import { Badge, Card, CardContent, Progress } from "@/components/ui/primitives";

export function PageHeader({ title, description, actions }: { title: string; description?: string; actions?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">{title}</h1>
        {description ? <p className="mt-1 max-w-2xl text-sm text-muted-foreground">{description}</p> : null}
      </div>
      {actions ? <div className="flex items-center gap-2">{actions}</div> : null}
    </div>
  );
}

export function EmptyState({ title, description, action }: { title: string; description?: string; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 rounded-lg border border-dashed px-6 py-12 text-center">
      <Inbox className="size-6 text-muted-foreground" />
      <p className="text-sm font-medium">{title}</p>
      {description ? <p className="max-w-sm text-xs text-muted-foreground">{description}</p> : null}
      {action ? <div className="mt-2">{action}</div> : null}
    </div>
  );
}

export function ErrorNotice({ error }: { error: unknown }) {
  if (!error) return null;
  const msg = error instanceof ApiError ? error.message : error instanceof Error ? error.message : String(error);
  const code = error instanceof ApiError ? error.code : null;
  return (
    <div className="flex items-start gap-2 rounded-md border border-destructive/30 bg-destructive/5 p-3 text-sm text-destructive">
      <AlertTriangle className="mt-0.5 size-4 shrink-0" />
      <div>
        <p>{msg}</p>
        {code ? <p className="text-xs opacity-70">{code}</p> : null}
      </div>
    </div>
  );
}

const TONE: Record<string, "success" | "warning" | "danger" | "info" | "neutral"> = {
  succeeded: "success",
  ready: "success",
  running: "info",
  processing: "info",
  queued: "neutral",
  pending: "neutral",
  mapping_required: "warning",
  uploaded: "neutral",
  failed: "danger",
  cancelled: "neutral",
};

export function StatusBadge({ status }: { status: string }) {
  return <Badge tone={TONE[status] ?? "neutral"}>{status.replace("_", " ")}</Badge>;
}

export function JobProgress({ status, pct, stage }: { status: string; pct: number; stage: string | null }) {
  const active = ["queued", "running", "processing", "pending"].includes(status);
  if (!active) return <StatusBadge status={status} />;
  return (
    <div className="flex min-w-40 flex-col gap-1">
      <div className="flex justify-between text-[11px] text-muted-foreground">
        <span>{stage ?? status}</span>
        <span>{pct.toFixed(0)}%</span>
      </div>
      <Progress value={pct} />
    </div>
  );
}

export function Kpi({ label, value, sub }: { label: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <Card>
      <CardContent className="p-5">
        <p className="text-xs font-medium text-muted-foreground">{label}</p>
        <p className="mt-2 text-2xl font-semibold tracking-tight">{value}</p>
        {sub ? <p className="mt-1 text-xs text-muted-foreground">{sub}</p> : null}
      </CardContent>
    </Card>
  );
}
