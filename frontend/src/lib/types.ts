// Mirrors backend Pydantic schemas (app/schemas/*). Keep in sync when the API changes.

export type UUID = string;
export type JobStatus = "pending" | "queued" | "running" | "succeeded" | "failed" | "cancelled";
export type DatasetStatus = "uploaded" | "mapping_required" | "queued" | "processing" | "ready" | "failed";
export type ModelFamily = "seasonal_naive" | "lightgbm" | "torch_mlp";
export type TargetType = "sales" | "units" | "customers";

export interface Page<T> {
  items: T[];
  total: number;
  limit: number;
  offset: number;
}

export interface ApiErrorBody {
  error: { code: string; message: string; details?: unknown; request_id?: string };
}

export interface TokenPair {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
}

export interface Me {
  user: { id: UUID; email: string; full_name: string | null; role: string; organization_id: UUID };
  organization: { id: UUID; name: string; slug: string; plan: string; default_currency: string };
}

export interface Store {
  id: UUID;
  code: string;
  name: string;
  store_type: string | null;
  assortment: string | null;
  competition_distance: number | null;
  promo2: boolean;
  promo_interval: string | null;
  target_type: TargetType;
  currency: string;
  has_financial_config: boolean;
}

export interface FinancialConfig {
  gross_margin_pct: string;
  cogs_pct: string;
  fixed_daily_overhead: string;
  promo_discount_pct: string;
  hourly_wage: string;
  customers_per_staff_hour: string;
  min_staff_per_shift: number;
  shift_hours: string;
  operating_hours_per_day: string;
  avg_transaction_value: string | null;
  avg_units_per_transaction: string | null;
}

export interface ColumnProfile {
  name: string;
  inferred_type: string;
  null_fraction: number;
  sample_values: unknown[];
}

export interface SchemaMapping {
  layout: "long" | "wide";
  date_column?: string | null;
  target_column?: string | null;
  store_column?: string | null;
  item_column?: string | null;
  customers_column?: string | null;
  known_future_columns: string[];
  categorical_columns: string[];
  promo_column?: string | null;
  open_column?: string | null;
  frequency?: "D" | "W";
  wide_value_prefix?: string | null;
  wide_start_date?: string | null;
  id_columns?: string[];
}

export interface Dataset {
  id: UUID;
  name: string;
  source_filename: string;
  file_format: string;
  size_bytes: number;
  status: DatasetStatus;
  progress_pct: number;
  stage: string | null;
  status_message: string | null;
  error: string | null;
  row_count: number | null;
  series_count: number | null;
  date_min: string | null;
  date_max: string | null;
  schema_mapping: SchemaMapping | null;
  suggested_mapping: Partial<SchemaMapping>;
  column_profile: { columns?: ColumnProfile[] };
  created_at: string;
}

export interface HealthIssue {
  severity: "critical" | "warning" | "info";
  code: string;
  message: string;
  details: Record<string, unknown>;
}

export interface HealthReport {
  score: number;
  row_count: number;
  series_count: number;
  store_count: number;
  date_min: string;
  date_max: string;
  zero_fraction: number;
  missing_series_days: number;
  outlier_count: number;
  weekday_profile: Record<string, number>;
  daily_total: { ds: string; y: number }[];
  issues: HealthIssue[];
}

export interface ModelRun {
  id: UUID;
  dataset_id: UUID;
  store_id: UUID | null;
  name: string | null;
  model_family: ModelFamily;
  run_type: "train" | "tune";
  horizon_days: number;
  validation_days: number;
  hyperparameters: Record<string, unknown>;
  n_trials: number | null;
  metrics: Record<string, unknown> & { wape?: number; rmse?: number; mae?: number; rmspe?: number };
  inference_ms: number | null;
  device: string | null;
  status: JobStatus;
  progress_pct: number;
  stage: string | null;
  status_message: string | null;
  error: string | null;
  plot_paths: string[];
  created_at: string;
  finished_at: string | null;
}

export interface ModelRunDetail extends ModelRun {
  epoch_logs: Record<string, number>[];
  trials: { number: number; value: number; params: Record<string, unknown> }[];
}

export interface ModelFamilyInfo {
  family: ModelFamily;
  description: string;
  supports_gpu: boolean;
  available: boolean;
}

export interface LeaderboardEntry {
  rank: number;
  model_run_id: UUID;
  name: string | null;
  model_family: ModelFamily;
  run_type: string;
  metric_value: number;
  metrics: Record<string, number>;
  inference_ms: number | null;
}

export interface ForecastRun {
  id: UUID;
  model_run_id: UUID;
  dataset_id: UUID;
  store_id: UUID | null;
  kind: "multi_horizon" | "event";
  horizon_days: number;
  status: JobStatus;
  progress_pct: number;
  stage: string | null;
  error: string | null;
  summary: {
    start_date?: string;
    end_date?: string;
    store_count?: number;
    series_count?: number;
    horizon_totals?: Record<string, { yhat: number; yhat_lower: number | null; yhat_upper: number | null }>;
    top_stores?: { store_code: string; yhat: number }[];
    event_impact?: Record<string, { yhat: number; baseline_yhat: number; uplift: number; uplift_pct: number | null }>;
  };
  created_at: string;
}

export interface ForecastPoint {
  ds: string;
  store_code: string;
  yhat: number;
  yhat_lower: number | null;
  yhat_upper: number | null;
  baseline_yhat: number | null;
}

export interface PnLReport {
  store_code: string;
  currency: string;
  days: {
    ds: string;
    net_revenue: number;
    gross_profit: number;
    labor_cost: number;
    operating_profit: number;
    is_promo: boolean;
  }[];
  totals: {
    net_revenue: number;
    gross_profit: number;
    labor_cost: number;
    operating_profit: number;
    realized_gross_margin_pct: number | null;
    operating_margin_pct: number | null;
    promo_discount: number;
  };
  assumptions: string[];
}

export interface WorkforcePlan {
  currency: string;
  days: { ds: string; staff_per_shift: number; scheduled_staff_hours: number; expected_customers: number }[];
  total_scheduled_hours: number;
  total_labor_cost: number;
  peak_staff_per_shift: number;
  assumptions: string[];
}
