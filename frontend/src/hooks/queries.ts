"use client";

import { useQuery } from "@tanstack/react-query";
import { api, isActive } from "@/lib/api";
import type { AutopilotRun, AutopilotRunDetail, Dataset, ForecastRun, Me, ModelRun, Page, Store } from "@/lib/types";

/** Poll every 2s while any item is still running (progress lives in PostgreSQL). */
const pollWhileActive =
  <T extends { status: string }>(pick: (d: unknown) => T[]) =>
  (q: { state: { data: unknown } }) =>
    q.state.data && pick(q.state.data).some((x) => isActive(x.status)) ? 2000 : false;

export const useMe = () => useQuery({ queryKey: ["me"], queryFn: () => api.get<Me>("/auth/me") });

export const useStores = () =>
  useQuery({ queryKey: ["stores"], queryFn: () => api.get<Page<Store>>("/stores?limit=500") });

export const useDatasets = () =>
  useQuery({
    queryKey: ["datasets"],
    queryFn: () => api.get<Page<Dataset>>("/datasets?limit=100"),
    refetchInterval: pollWhileActive<Dataset>((d) => (d as Page<Dataset>).items),
  });

export const useDataset = (id: string) =>
  useQuery({
    queryKey: ["dataset", id],
    queryFn: () => api.get<Dataset>(`/datasets/${id}`),
    refetchInterval: pollWhileActive<Dataset>((d) => [d as Dataset]),
  });

export const useModelRuns = (datasetId?: string) =>
  useQuery({
    queryKey: ["model-runs", datasetId ?? "all"],
    queryFn: () => api.get<Page<ModelRun>>(`/model-runs?limit=100${datasetId ? `&dataset_id=${datasetId}` : ""}`),
    refetchInterval: pollWhileActive<ModelRun>((d) => (d as Page<ModelRun>).items),
  });

export const useForecasts = () =>
  useQuery({
    queryKey: ["forecasts"],
    queryFn: () => api.get<Page<ForecastRun>>("/forecasts?limit=100"),
    refetchInterval: pollWhileActive<ForecastRun>((d) => (d as Page<ForecastRun>).items),
  });

export const useAutopilotRuns = (storeId?: string) =>
  useQuery({
    queryKey: ["autopilot-runs", storeId ?? "all"],
    queryFn: () => api.get<Page<AutopilotRun>>(`/autopilot/runs?limit=50${storeId ? `&store_id=${storeId}` : ""}`),
    refetchInterval: pollWhileActive<AutopilotRun>((d) => (d as Page<AutopilotRun>).items),
  });

export const useAutopilotRun = (id: string | null) =>
  useQuery({
    queryKey: ["autopilot-run", id],
    enabled: !!id,
    queryFn: () => api.get<AutopilotRunDetail>(`/autopilot/runs/${id}`),
    refetchInterval: pollWhileActive<AutopilotRunDetail>((d) => [d as AutopilotRunDetail]),
  });
