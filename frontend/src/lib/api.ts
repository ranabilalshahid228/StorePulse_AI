"use client";

import type { ApiErrorBody, TokenPair } from "./types";

export const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000/api/v1";

const ACCESS = "sp.access";
const REFRESH = "sp.refresh";

export const tokens = {
  get access() {
    return typeof window === "undefined" ? null : window.localStorage.getItem(ACCESS);
  },
  get refresh() {
    return typeof window === "undefined" ? null : window.localStorage.getItem(REFRESH);
  },
  set(pair: TokenPair) {
    window.localStorage.setItem(ACCESS, pair.access_token);
    window.localStorage.setItem(REFRESH, pair.refresh_token);
  },
  clear() {
    window.localStorage.removeItem(ACCESS);
    window.localStorage.removeItem(REFRESH);
  },
};

export class ApiError extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
    public details?: unknown,
    public requestId?: string,
  ) {
    super(message);
  }
}

async function tryRefresh(): Promise<boolean> {
  const refresh = tokens.refresh;
  if (!refresh) return false;
  const res = await fetch(`${API_BASE}/auth/refresh`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ refresh_token: refresh }),
  });
  if (!res.ok) return false;
  tokens.set((await res.json()) as TokenPair);
  return true;
}

async function raw(path: string, init: RequestInit = {}, retry = true): Promise<Response> {
  const headers = new Headers(init.headers);
  const access = tokens.access;
  if (access) headers.set("Authorization", `Bearer ${access}`);
  if (init.body && !(init.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const res = await fetch(`${API_BASE}${path}`, { ...init, headers });
  if (res.status === 401 && retry && (await tryRefresh())) return raw(path, init, false);
  if (!res.ok) {
    let body: ApiErrorBody | null = null;
    try {
      body = (await res.json()) as ApiErrorBody;
    } catch {
      /* non-JSON error */
    }
    if (res.status === 401 && typeof window !== "undefined" && !path.startsWith("/auth/")) {
      tokens.clear();
      window.location.href = "/login";
    }
    throw new ApiError(
      res.status,
      body?.error.code ?? "http_error",
      body?.error.message ?? res.statusText,
      body?.error.details,
      body?.error.request_id,
    );
  }
  return res;
}

export const api = {
  async get<T>(path: string): Promise<T> {
    return (await raw(path)).json() as Promise<T>;
  },
  async post<T>(path: string, body?: unknown): Promise<T> {
    const init: RequestInit = { method: "POST" };
    if (body instanceof FormData) init.body = body;
    else if (body !== undefined) init.body = JSON.stringify(body);
    return (await raw(path, init)).json() as Promise<T>;
  },
  async put<T>(path: string, body: unknown): Promise<T> {
    return (await raw(path, { method: "PUT", body: JSON.stringify(body) })).json() as Promise<T>;
  },
  async blobUrl(path: string): Promise<string> {
    return URL.createObjectURL(await (await raw(path)).blob());
  },
  async download(path: string, filename: string): Promise<void> {
    const url = await api.blobUrl(path);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  },
};

export const isActive = (status: string) => ["pending", "queued", "running", "processing"].includes(status);
