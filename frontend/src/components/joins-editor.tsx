"use client";

import { Plus, Trash2 } from "lucide-react";
import type { Dataset, JoinSpec } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Field, Input, Select } from "@/components/ui/primitives";

const CANONICAL = ["ds", "store_code", "item_id"];
const csv = (v: string) => v.split(",").map((s) => s.trim()).filter(Boolean);

/**
 * Edits SchemaMapping.joins. Each join left-joins columns from an auxiliary dataset; later joins may
 * key on columns added by earlier ones (e.g. M5: calendar adds wm_yr_wk, prices join on it).
 */
export function JoinsEditor({
  joins,
  onChange,
  auxiliary,
}: {
  joins: JoinSpec[];
  onChange: (joins: JoinSpec[]) => void;
  auxiliary: Dataset[];
}) {
  const update = (i: number, patch: Partial<JoinSpec>) => onChange(joins.map((j, k) => (k === i ? { ...j, ...patch } : j)));

  if (!auxiliary.length) {
    return (
      <p className="text-xs text-muted-foreground">
        Upload a side table (role &ldquo;Side table&rdquo;) such as a calendar or price list to join it here.
      </p>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      {joins.map((j, i) => {
        const aux = auxiliary.find((a) => a.id === j.dataset_id);
        const auxCols = (aux?.column_profile.columns ?? []).map((c) => c.name);
        const available = [...CANONICAL, ...joins.slice(0, i).flatMap((p) => p.columns)];
        return (
          <div key={i} className="grid gap-3 rounded-md border p-3 md:grid-cols-[1fr_1fr_1fr_auto]">
            <Field label={`Join ${i + 1}: side table`}>
              <Select value={j.dataset_id} onChange={(e) => update(i, { dataset_id: e.target.value, right_on: [], columns: [] })}>
                {auxiliary.map((a) => <option key={a.id} value={a.id}>{a.name}</option>)}
              </Select>
            </Field>
            <Field label="Match keys (this data → side table)" hint={`Available: ${available.join(", ")}`}>
              <Input
                placeholder="ds=date, store_code=store_id"
                defaultValue={j.left_on.map((l, k) => `${l}=${j.right_on[k] ?? ""}`).join(", ")}
                onBlur={(e) => {
                  const pairs = csv(e.target.value).map((p) => p.split("=").map((s) => s.trim()));
                  update(i, { left_on: pairs.map((p) => p[0] ?? ""), right_on: pairs.map((p) => p[1] ?? p[0] ?? "") });
                }}
              />
            </Field>
            <Field label="Columns to add">
              <div className="flex flex-wrap gap-1.5">
                {auxCols.filter((c) => !j.right_on.includes(c)).map((c) => {
                  const on = j.columns.includes(c);
                  return (
                    <button
                      type="button"
                      key={c}
                      onClick={() => update(i, { columns: on ? j.columns.filter((x) => x !== c) : [...j.columns, c] })}
                      className={`rounded-full border px-2.5 py-0.5 text-xs ${on ? "border-primary bg-primary text-primary-foreground" : "hover:bg-muted"}`}
                    >
                      {c}
                    </button>
                  );
                })}
              </div>
            </Field>
            <Button type="button" variant="ghost" size="icon" aria-label="Remove join" onClick={() => onChange(joins.filter((_, k) => k !== i))}>
              <Trash2 />
            </Button>
          </div>
        );
      })}
      <div>
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={() => onChange([...joins, { dataset_id: auxiliary[0]!.id, left_on: [], right_on: [], columns: [] }])}
        >
          <Plus /> Add join
        </Button>
      </div>
    </div>
  );
}
