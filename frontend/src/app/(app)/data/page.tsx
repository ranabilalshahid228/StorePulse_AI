"use client";

import Link from "next/link";
import { useState, type FormEvent } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Upload } from "lucide-react";
import { api } from "@/lib/api";
import type { Dataset } from "@/lib/types";
import { useDatasets } from "@/hooks/queries";
import { Button } from "@/components/ui/button";
import { Badge, Card, CardContent, CardDescription, CardHeader, CardTitle, Field, Input, Select, Table, TD, TH, THead, TR } from "@/components/ui/primitives";
import { EmptyState, ErrorNotice, JobProgress, PageHeader } from "@/components/shared";
import { formatCompact, formatDateTime } from "@/lib/utils";

export default function DataPage() {
  const qc = useQueryClient();
  const datasets = useDatasets();
  const [formKey, setFormKey] = useState(0);
  const upload = useMutation({
    mutationFn: (fd: FormData) => api.post<Dataset>("/datasets", fd),
    onSuccess: () => {
      setFormKey((k) => k + 1);
      void qc.invalidateQueries({ queryKey: ["datasets"] });
    },
  });

  function onSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    upload.mutate(new FormData(e.currentTarget));
  }

  return (
    <>
      <PageHeader title="Data & EDA" description="Upload sales history, confirm the schema mapping, and review data health before training." />
      <Card className="mb-6">
        <CardHeader>
          <CardTitle>Upload dataset</CardTitle>
          <CardDescription>CSV or Parquet. Sales history in long or M5-style wide format; side tables (calendars, price lists) are joined in when you map the sales history.</CardDescription>
        </CardHeader>
        <CardContent>
          <form key={formKey} onSubmit={onSubmit} className="grid items-end gap-4 md:grid-cols-[1fr_12rem_1fr_auto]">
            <Field label="Name">
              <Input name="name" required placeholder="e.g. Rossmann daily sales 2013–2015" />
            </Field>
            <Field label="Role">
              <Select name="role" defaultValue="primary">
                <option value="primary">Sales history</option>
                <option value="auxiliary">Side table (calendar, prices…)</option>
              </Select>
            </Field>
            <Field label="File">
              <Input name="file" type="file" accept=".csv,.parquet" required />
            </Field>
            <Button type="submit" disabled={upload.isPending}>
              <Upload /> {upload.isPending ? "Uploading…" : "Upload"}
            </Button>
          </form>
          <div className="mt-3"><ErrorNotice error={upload.error} /></div>
        </CardContent>
      </Card>

      <StoreMetadataCard />

      <Card>
        <CardHeader><CardTitle>Datasets</CardTitle></CardHeader>
        <CardContent>
          {datasets.data?.items.length ? (
            <Table>
              <THead>
                <tr><TH>Name</TH><TH>Status</TH><TH>Rows</TH><TH>Series</TH><TH>Date range</TH><TH>Uploaded</TH></tr>
              </THead>
              <tbody>
                {datasets.data.items.map((d) => (
                  <TR key={d.id}>
                    <TD>
                      <Link href={`/data/${d.id}`} className="font-medium hover:underline">{d.name}</Link>
                      <p className="text-xs text-muted-foreground">
                        {d.role === "auxiliary" ? <Badge className="mr-1.5">side table</Badge> : null}
                        {d.source_filename}
                      </p>
                    </TD>
                    <TD><JobProgress status={d.status} pct={d.progress_pct} stage={d.stage} /></TD>
                    <TD>{formatCompact(d.row_count)}</TD>
                    <TD>{formatCompact(d.series_count)}</TD>
                    <TD className="text-muted-foreground">{d.date_min ? `${d.date_min} → ${d.date_max}` : "—"}</TD>
                    <TD className="text-muted-foreground">{formatDateTime(d.created_at)}</TD>
                  </TR>
                ))}
              </tbody>
            </Table>
          ) : (
            <EmptyState title="No datasets yet" description="Upload your first sales history file above." />
          )}
        </CardContent>
      </Card>
    </>
  );
}

/** Store attributes (Rossmann store.csv): type, assortment, competition, Promo2 - used as model
 * features, for cold-start cohorts and EDA segments. */
function StoreMetadataCard() {
  const qc = useQueryClient();
  const [formKey, setFormKey] = useState(0);
  const imp = useMutation({
    mutationFn: (fd: FormData) => api.post<unknown[]>("/stores/import", fd),
    onSuccess: () => {
      setFormKey((k) => k + 1);
      void qc.invalidateQueries({ queryKey: ["stores"] });
    },
  });
  return (
    <Card className="mb-6">
      <CardHeader>
        <CardTitle>Store metadata</CardTitle>
        <CardDescription>
          Optional CSV with one row per store (Rossmann store.csv columns or Store field names). Adds store type, assortment,
          competition and Promo2 features, enables cold-start cohorts and EDA segments. Re-train models afterwards.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form
          key={formKey}
          onSubmit={(e: FormEvent<HTMLFormElement>) => {
            e.preventDefault();
            imp.mutate(new FormData(e.currentTarget));
          }}
          className="grid items-end gap-4 md:grid-cols-[1fr_8rem_10rem_auto]"
        >
          <Field label="File"><Input name="file" type="file" accept=".csv" required /></Field>
          <Field label="Currency"><Input name="currency" defaultValue="EUR" maxLength={3} /></Field>
          <Field label="Target type">
            <Select name="target_type" defaultValue="sales">
              <option value="sales">Sales (revenue)</option>
              <option value="units">Units</option>
              <option value="customers">Customers</option>
            </Select>
          </Field>
          <Button type="submit" variant="secondary" disabled={imp.isPending}>
            <Upload /> {imp.isPending ? "Importing…" : "Import stores"}
          </Button>
        </form>
        <div className="mt-3"><ErrorNotice error={imp.error} /></div>
        {imp.isSuccess ? <p className="mt-2 text-xs text-success">Imported {imp.data.length} store(s).</p> : null}
      </CardContent>
    </Card>
  );
}
