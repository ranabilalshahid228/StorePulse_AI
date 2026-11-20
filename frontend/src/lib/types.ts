// Mirrors backend Pydantic schemas (app/schemas/*). Keep in sync when the API changes.

export type UUID = string;
export type JobStatus = "pending" | "queued" | "running" | "succeeded" | "failed" | "cancelled";
export type DatasetStatus = "uploaded" | "mapping_required" | "queued" | "processing" | "ready" | "failed";
export type ModelFamily = "seasonal_naive" | "lightgbm" | "torch_mlp" | "torch_gru" | "ensemble";
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
  autopilot_settings?: AutopilotSettings;
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
  price_column?: string | null;
  frequency?: "D" | "W";
  wide_value_prefix?: string | null;
  wide_start_date?: string | null;
  id_columns?: string[];
  joins?: JoinSpec[];
  helper_columns?: string[];
  store_filter?: string[] | null;
}

export interface JoinSpec {
  dataset_id: UUID;
  left_on: string[];
  right_on: string[];
  columns: string[];
}

export interface Dataset {
  id: UUID;
  name: string;
  source_filename: string;
  file_format: string;
  size_bytes: number;
  role: "primary" | "auxiliary";
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
  training_config: { history_days?: number | null; use_store_features?: boolean };
  feature_columns: string[];
  metrics: Record<string, unknown> & {
    wape?: number;
    rmse?: number;
    mae?: number;
    rmspe?: number;
    rmsse?: number | null;
    r2?: number | null;
    interval?: IntervalMetrics;
    ensemble?: EnsembleComparison;
  };
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

export interface IntervalMetrics {
  method: "quantile_model" | "empirical_relative_residual";
  coverage?: number;
  /** Empirical band only: calibrated on the first half of validation, measured on the second. */
  coverage_holdout?: number | null;
  mean_width?: number;
  levels?: number[];
  pinball_p10?: number;
  pinball_p50?: number;
  pinball_p90?: number;
}

export interface EnsembleComparison {
  weights: Record<string, number>;
  weight_metric: string;
  weights_fitted_on: string;
  compared_on: string;
  holdout_rows: number;
  models: Record<string, { wape: number | null; rmse: number | null; mae: number | null; bias_pct: number | null; r2: number | null }>;
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
  yhat_p50?: number | null;
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

/* ---- Store Autopilot (mirrors app/schemas/autopilot.py) ------------------------------------ */

export interface ShiftDefinition {
  name: string;
  traffic_share: number;
  hours?: number | null;
}

export interface AutopilotSettings {
  shifts: ShiftDefinition[];
  peak_hour_share: number;
  max_staff_per_shift: number | null;
  overstaff_utilization: number;
  surge_block_hours: number;
  cold_start_min_days: number;
  cohort_size: number;
  basket_lookback_days: number;
  anomaly_threshold: number;
}

export type HorizonKind = "next_day" | "week" | "month" | "custom";
export type LeverKind = "promo" | "holiday" | "closure" | "local_event" | "competitor" | "price" | "custom";

export interface ScenarioLever {
  name: string;
  kind: LeverKind;
  start_date: string;
  end_date: string;
  promo?: boolean | null;
  school_holiday?: boolean | null;
  state_holiday?: string | null;
  open?: boolean | null;
  demand_pct?: number | null;
  price_change_pct?: number | null;
  covariates?: Record<string, number | string>;
}

export interface AutopilotScenario {
  levers: ScenarioLever[];
  promo2?: boolean | null;
  promo2_interval?: string | null;
  covariate_defaults?: Record<string, number>;
  price_elasticity?: number | null;
  cogs_pct?: number | null;
  promo_discount_pct?: number | null;
}

export interface AutopilotRequest {
  store_id: UUID;
  model_run_id: UUID;
  horizon: HorizonKind;
  start_date?: string | null;
  end_date?: string | null;
  scenario: AutopilotScenario;
  name?: string | null;
}

export type Priority = "critical" | "high" | "medium" | "low";

export interface Recommendation {
  priority: Priority;
  category: string;
  title: string;
  detail: string;
  dates: string[];
  impact: number | null;
}

export interface AutopilotDay {
  ds: string;
  is_open: boolean;
  is_promo: boolean;
  event: string | null;
  revenue_p10: number;
  revenue_p50: number;
  revenue_p90: number;
  baseline_revenue_p50?: number | null;
  baseline_net_profit_p50?: number | null;
  footfall: number;
  basket_size: number | null;
  gross_revenue: number;
  promo_discount: number;
  cogs: number;
  gross_profit: number;
  labor_hours: number;
  labor_cost: number;
  fixed_overhead: number;
  net_profit_p10: number;
  net_profit_p50: number;
  net_profit_p90: number;
  break_even_revenue: number | null;
  surge_staff: number;
  peak_staff_needed: number;
  utilization: number | null;
  staffing_status: "ok" | "understaffed" | "overstaffed" | "closed";
  [staffColumn: `staff_${string}`]: number;
}

export interface AutopilotShift {
  ds: string;
  shift: string;
  hours: number;
  expected_customers: number;
  required_hours: number;
  headcount: number;
  utilization: number | null;
  status: "ok" | "understaffed" | "overstaffed" | "closed";
}

export interface AutopilotTotals {
  gross_revenue: number;
  promo_discount: number;
  revenue_p10: number;
  revenue_p50: number;
  revenue_p90: number;
  cogs: number;
  gross_profit: number;
  labor_cost: number;
  fixed_overhead: number;
  net_profit_p10: number;
  net_profit_p50: number;
  net_profit_p90: number;
  gross_margin_pct: number | null;
  net_margin_pct: number | null;
  break_even_revenue: number | null;
  margin_of_safety_pct: number | null;
}

export interface AutopilotKpis {
  window_start: string;
  window_end: string;
  days: number;
  open_days: number;
  revenue_p10: number;
  revenue_p50: number;
  revenue_p90: number;
  net_profit_p10: number;
  net_profit_p50: number;
  net_profit_p90: number;
  gross_margin_pct: number | null;
  net_margin_pct: number | null;
  break_even_revenue: number | null;
  margin_of_safety_pct: number | null;
  footfall: number;
  labor_hours: number;
  labor_cost: number;
  peak_staff_per_shift: number;
  understaffed_shifts: number;
  overstaffed_shifts: number;
  cold_start: boolean;
  recommendations: Record<Priority, number>;
  scenario_net_profit_delta?: number;
}

export interface ModelComparisonRow {
  model_run_id: UUID;
  name: string | null;
  model_family: string;
  run_type: string;
  horizon_days: number;
  wape: number | null;
  rmspe: number | null;
  rmsse: number | null;
  r2: number | null;
  accuracy: number | null;
  interval_coverage: number | null;
}

export interface AutopilotResult {
  window: { kind: HorizonKind; start: string; end: string; history_end: string; days: number };
  store_code: string;
  currency: string;
  target_type: TargetType;
  kpis: AutopilotKpis;
  totals: AutopilotTotals;
  scenario_impact: { revenue_p50: number; net_profit_p50: number; labor_cost: number } | null;
  days: AutopilotDay[];
  shifts: AutopilotShift[];
  shift_names: string[];
  recommendations: Recommendation[];
  anomalies: { kind: string; ds: string | null; detail: string }[];
  cold_start: { codes: string[]; matched_on: string[]; scale: number; note: string; open_days: number } | null;
  model: { model_run_id?: UUID; model_family?: string; metrics?: Record<string, number | null> };
  model_comparison: ModelComparisonRow[];
  scenario: AutopilotScenario;
  assumptions: string[];
}

export interface AutopilotRun {
  id: UUID;
  store_id: UUID;
  model_run_id: UUID;
  dataset_id: UUID;
  name: string | null;
  horizon: HorizonKind;
  start_date: string | null;
  end_date: string | null;
  scenario: AutopilotScenario;
  summary: Partial<AutopilotKpis>;
  status: string;
  progress_pct: number;
  stage: string | null;
  status_message: string | null;
  error: string | null;
  created_at: string;
  finished_at: string | null;
}

export interface AutopilotRunDetail extends AutopilotRun {
  result: AutopilotResult | Record<string, never>;
}
