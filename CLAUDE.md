# StorePulse AI — engineering guide

Multi-tenant SaaS for retail demand forecasting and store autopilot: upload sales history → map schema →
data-health diagnostics → train / Optuna-tune models → multi-horizon & event forecasts → P&L and
workforce plans → CSV/PDF reports.

```
Organization ─┬─ User (owner | admin | analyst | viewer)
              └─ Store ─── FinancialStaffingConfig (1:1)
              └─ Dataset (store_id optional: multi-store files map rows via store column)
                    └─ ModelRun (train | tune) ─── ForecastRun (multi_horizon | event)
```

## Repository layout

```
.
├── CLAUDE.md                 ← this file
├── docker-compose.yml        ← postgres, redis, backend, celery-worker, frontend (+ gpu / s3 profiles)
├── .env.example              ← every environment variable, documented
├── data/raw/                 ← local sample data (Rossmann, M5). Git-ignored, never baked into images.
├── backend/
│   ├── pyproject.toml        ← deps; extras: [torch], [dev]
│   ├── Dockerfile            ← API + CPU worker image (runs `alembic upgrade head` on API start)
│   ├── Dockerfile.gpu        ← CUDA worker for the `training` queue
│   ├── alembic.ini, alembic/ ← migrations (versions/YYYYMMDD_<rev>_<slug>.py)
│   ├── app/
│   │   ├── main.py           ← FastAPI factory: request-id middleware, CORS, error handlers, /api/v1
│   │   ├── core/             ← config (pydantic-settings), security (JWT, bcrypt), exceptions, logging
│   │   ├── db/               ← Declarative Base + mixins (UUID pk, timestamps, JobStateMixin), engines
│   │   ├── models/           ← SQLAlchemy 2.0 models: tenancy.py, dataset.py, runs.py, enums.py
│   │   ├── schemas/          ← Pydantic v2 request/response models (one module per domain)
│   │   ├── api/
│   │   │   ├── deps.py       ← DbSession, CurrentPrincipal, AnalystPrincipal, AdminPrincipal
│   │   │   └── v1/           ← routers: auth, stores, datasets, model_runs, forecasts, finance, reports, health
│   │   ├── services/         ← storage (local/S3), ingestion, diagnostics, finance, reports, tenancy helpers
│   │   ├── ml/               ← features, model families, metrics, training/tuning, forecasting, plots
│   │   └── workers/          ← celery_app, tasks, progress reporter, dispatch (enqueue by task name)
│   └── tests/                ← pytest: API envelope/auth, finance math, end-to-end pipeline on Rossmann
└── frontend/                 ← Next.js 15 App Router + TypeScript + Tailwind v4 + shadcn/ui-style + Recharts
    └── src/
        ├── app/(app)/        ← authenticated shell: / (Command Center), /data, /data/[id], /model-lab,
        │                        /model-lab/[id], /pnl-workforce, /reports, /stores
        ├── app/login/        ← sign in / create workspace
        ├── components/ui/    ← shadcn-style primitives (button, card, badge, input, select, table, progress)
        ├── components/       ← app-sidebar, shared (PageHeader, JobProgress, Kpi…), charts (Recharts)
        ├── hooks/queries.ts  ← TanStack Query hooks; poll every 2 s while a job is active
        └── lib/              ← api.ts (fetch + JWT refresh + error envelope), types.ts (mirrors schemas)
```

## Artifact storage layout

All keys are tenant-prefixed (`app/services/storage.py::ArtifactKeys`). Same layout on local disk
(`STORAGE_LOCAL_ROOT`) and S3-compatible buckets (`S3_BUCKET`):

```
orgs/{org_id}/datasets/{dataset_id}/raw/{original_filename}
orgs/{org_id}/datasets/{dataset_id}/canonical.parquet          # ds, store_code, item_id, y, customers?, covariates
orgs/{org_id}/models/{model_run_id}/model.pkl | model.pt       # joblib bundle (spec+model) | torch state + prep
orgs/{org_id}/models/{model_run_id}/plots/{name}.png           # validation_actual_vs_forecast, learning_curve, feature_importance
orgs/{org_id}/forecasts/{forecast_run_id}/forecast.parquet     # ds, store_code, item_id, yhat, yhat_lower/upper, baseline_yhat, is_promo, is_open, event
orgs/{org_id}/reports/{forecast_run_id}/{name}                 # reserved for persisted reports (currently streamed)
```

Workers needing a filesystem path call `storage.local_path(key)` — on S3 this downloads into
`STORAGE_CACHE_DIR`. Writers create a temp file and `storage.put_file(key, path)`.

## Environment variables

Full list with defaults in `.env.example`; source of truth is `backend/app/core/config.py`.

| Variable | Default | Notes |
|---|---|---|
| `JWT_SECRET_KEY` | — (**required**, ≥32 chars) | HS256 signing key |
| `ACCESS_TOKEN_EXPIRE_MINUTES` / `REFRESH_TOKEN_EXPIRE_MINUTES` | 60 / 10080 | |
| `POSTGRES_HOST/PORT/USER/PASSWORD/DB` | localhost/5432/storepulse… | API uses asyncpg, workers & Alembic use psycopg 3 |
| `REDIS_URL` | redis://localhost:6379/0 | broker + result backend unless `CELERY_*` set |
| `CELERY_TASK_ALWAYS_EAGER` | false | run tasks inline (debugging only) |
| `STORAGE_BACKEND` | local | `local` or `s3` |
| `STORAGE_LOCAL_ROOT`, `STORAGE_CACHE_DIR` | ./storage, ./storage/.cache | |
| `S3_BUCKET`, `S3_ENDPOINT_URL`, `S3_REGION`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY` | | MinIO: `S3_ENDPOINT_URL=http://minio:9000` |
| `MAX_UPLOAD_MB` | 2048 | |
| `INGESTION_CHUNK_ROWS` | 500000 | CSV → Parquet streaming chunk size |
| `TRAINING_DEVICE` | auto | `auto` picks CUDA when available for GPU families; `cuda` fails fast if missing |
| `OPTUNA_DEFAULT_TRIALS` | 30 | |
| `OPTUNA_STORAGE_URL` | unset | set to a DB URL to persist/resume studies |
| `CORS_ORIGINS` | http://localhost:3000 | comma-separated |
| `NEXT_PUBLIC_API_URL` | http://localhost:8000/api/v1 | inlined into the frontend at build time |

## Local development

Prereqs: Python 3.11+, Node 22, Docker (for Postgres/Redis or the full stack).

Full stack in Docker:
```powershell
Copy-Item .env.example .env   # then set JWT_SECRET_KEY
docker compose up --build
# UI http://localhost:3000 · API docs http://localhost:8000/api/v1/docs
```

Backend natively (Postgres + Redis from compose):
```powershell
docker compose up -d postgres redis
cd backend
python -m venv .venv; .venv\Scripts\Activate.ps1      # source .venv/bin/activate on *nix
pip install -e ".[dev]"
Copy-Item ..\.env.example .env
alembic upgrade head
uvicorn app.main:app --reload --port 8000
# second terminal (Windows needs the solo pool):
celery -A app.workers.celery_app worker -Q ingestion,training,forecasting -P solo --loglevel=INFO
```

Frontend:
```powershell
cd frontend; npm install; npm run dev     # http://localhost:3000
npm run typecheck; npm run build
```

Tests & lint (backend):
```powershell
cd backend
pytest                 # tests/test_pipeline.py uses data/raw/rossman dataset/train.csv; skips if absent
ruff check app tests; ruff format app tests
```

Migrations:
```powershell
alembic revision --autogenerate -m "add xyz"   # review the file! name constraints explicitly
alembic upgrade head
alembic downgrade -1
```

### Typical end-to-end flow (API)
1. `POST /auth/register` → tokens. 2. `POST /stores/import` with Rossmann `store.csv` (optional metadata).
3. `POST /datasets` (multipart `file`, `name`) → profile + `suggested_mapping`.
4. `PUT /datasets/{id}/mapping` → enqueues ingestion; poll `GET /datasets/{id}` or SSE `/datasets/{id}/events`.
5. `PUT /stores/{id}/financial-config`.
6. `POST /model-runs/train` or `/model-runs/tune`; `GET /model-runs/leaderboard?dataset_id=…`.
7. `POST /forecasts` or `/forecasts/event`; then `/forecasts/{id}/pnl|workforce?store_id=…`.
8. `GET /reports/forecasts/{id}/forecast.csv | store-plan.csv | store-plan.pdf`.

### Multi-file datasets (auxiliary tables + joins)

Upload side tables with `role=auxiliary` (`POST /datasets`, form field `role`). They are profiled and
immediately `ready`, never trained on. A primary dataset's mapping lists `joins` (applied in order, at
ingestion **and again for future dates at forecast time**, so calendar events / planned prices use real
future values). `left_on` may use canonical keys (`ds, store_code, item_id`), categorical id columns, or
columns from earlier joins; every joined column must be listed in `known_future_columns`,
`categorical_columns` or `helper_columns` (join-only keys, dropped after). Implementation:
`services/joins.py` (shared), `services/ingestion.py::convert_to_parquet`, `workers/tasks.py::load_mapping_joins`.

**Rossmann** (long): `date_column=Date, target_column=Sales, store_column=Store, customers_column=Customers,
known_future_columns=[Open, Promo, SchoolHoliday], categorical_columns=[StateHoliday], promo_column=Promo,
open_column=Open`. Import `store.csv` via `POST /stores/import` → store features are added automatically
(`use_store_features`, default on): store type, assortment, competition distance & months open,
Promo2 active flag (from since-week/year + interval months) and weeks since Promo2 start.

**M5** (wide, 3 files): upload `calendar.csv` and `sell_prices.csv` as auxiliary, then map
`sales_train_evaluation.csv`:
```json
{"layout": "wide", "wide_value_prefix": "d_", "wide_start_date": "2011-01-29",
 "store_column": "store_id", "item_column": "item_id", "id_columns": ["dept_id", "cat_id", "state_id"],
 "store_filter": ["CA_1"],
 "joins": [
   {"dataset_id": "<calendar>", "left_on": ["ds"], "right_on": ["date"],
    "columns": ["wm_yr_wk", "event_name_1", "event_type_1", "snap_CA", "snap_TX", "snap_WI"]},
   {"dataset_id": "<sell_prices>", "left_on": ["store_code", "item_id", "wm_yr_wk"],
    "right_on": ["store_id", "item_id", "wm_yr_wk"], "columns": ["sell_price"]}],
 "helper_columns": ["wm_yr_wk"],
 "categorical_columns": ["dept_id", "cat_id", "state_id", "event_name_1", "event_type_1"],
 "known_future_columns": ["snap_CA", "snap_TX", "snap_WI", "sell_price"], "price_column": "sell_price"}
```
Set the store's `target_type=units`; with `price_column` the P&L values the forecast as Σ units × price.
Full M5 is ~59M rows long — use `store_filter` and/or `history_days` (e.g. 730) per training run.

### Local pipeline runner (no DB / queue)
`scripts/run_pipeline.py` runs the production ingestion → diagnostics → features → train → forecast code
on the bundled datasets and prints validation metrics:
```powershell
cd backend
python -m scripts.run_pipeline rossmann                          # all 1,115 stores
python -m scripts.run_pipeline m5 --stores CA_1 --history-days 730 --out m5.json
python -m scripts.run_pipeline rossmann --stores 1 2 3 --families lightgbm --no-store-features
python -m scripts.run_pipeline rossmann --families lightgbm --lgb-rounds 4000 --tune 25   # honest Optuna
```
Other flags: `--horizon`, `--validation-days` (compare horizons on the same window), `--lgb-objective`,
`--quantiles`, `--lgb-param KEY=VALUE` (JSON value, repeatable). Each line prints `ACC` = 1 − WAPE.

Reference results (Rossmann, all 1,115 stores, horizon 42, last 42 days held out, LightGBM):
default params **91.52 %** (WAPE 0.0848, RMSPE 0.1245, R² 0.954); 25-trial tuning **91.77 %**
(WAPE 0.0823). Tried and rejected: log1p target (91.36 %), L1 objective (91.29 %), shorter horizons
(14 d: 90.54 % on the same window). M5 CA_1 item-level is a different problem (WAPE ≈ 0.73).

Neural nets and ensemble (same data, last 42 days held out for every horizon, CPU, `--torch-epochs 20`):

| Model / variant | 14 d | 28 d | 42 d | Notes (42 d) |
|---|---|---|---|---|
| LightGBM (default) | 90.54 % | 91.61 % | 91.52 % | ~1 min |
| `torch_mlp` (log1p target) | **91.77 %** | 91.11 % | 91.35 % | 5 min, RMSPE 0.1255 |
| `torch_mlp` `target_transform=standard` | | | 91.46 % | |
| `torch_mlp` quantile heads | | | 91.40 % | learned band covers 78.6 % |
| `torch_mlp` `feature_log=false` | | | 90.94 % | log-compressing inputs matters |
| `torch_mlp` 6-trial honest tuning | | | 91.17 % | inner 92.07 % did not carry over |
| `torch_gru` | 91.57 % | 91.30 % | 91.37 % | 52 min, RMSPE 0.1215 |
| `ensemble` (LightGBM + GRU, 0.45/0.55) | | | **91.84 %** | RMSPE 0.1186, RMSSE 0.158 |

The ensemble's weights are fitted on the first half of the validation window; on the untouched second half
the fitted blend scores WAPE 0.0779 vs LightGBM 0.0801 and GRU 0.0840. Empirical P10–P90 bands hold
78–85 % coverage out-of-sample (`coverage_holdout`).

## Background jobs & progress

- Queues: `ingestion` (upload → Parquet + diagnostics), `training` (train + Optuna), `forecasting`.
- Tasks receive **only the row id**; all inputs live in Postgres → idempotent and re-runnable
  (`acks_late`, `reject_on_worker_lost`, `prefetch_multiplier=1`).
- `JobStateMixin` columns (`status, progress_pct, stage, status_message, started_at, finished_at, error,
  celery_task_id`) on `datasets`, `model_runs`, `forecast_runs`. Workers write through
  `workers/progress.py::ProgressReporter` (throttled to ≤2 writes/s; `.sub(lo, hi)` maps sub-stages
  into a slice of the bar). The API exposes them via GET + SSE (`/…/{id}/events`); the UI polls every 2 s.
- The API never imports ML code for dispatch: `workers/dispatch.py` enqueues by task **name**.

## ML design

- Canonical long format; `ml/features.py::complete_calendar` reindexes each series to a contiguous
  daily calendar before lags. All features are vectorized across series (integer group ids +
  groupby/rolling, float32) — an M5 store (3,049 items) featurizes in seconds.
- **Direct multi-horizon, leakage-free:** every target-derived feature reads the target shifted by
  `lag_offset = ceil(horizon/7)*7`, so any day ≤ horizon ahead is predictable from history alone. A
  model trained for horizon H can forecast ≤ H days (enforced by the API).
  `tests/test_models_metrics.py::test_target_features_are_leak_free` scrambles all targets after a
  cutoff and asserts no target-derived feature changes for rows whose lag origin is ≤ cutoff —
  extend its prefix list whenever you add a target-derived feature.
- Features (`build_features`):
  - calendar; weekday-aligned lags (offset+0/7/14/21/49); rolling mean 7/28, std 28, zero share 28;
    same-weekday 4-week mean
  - intermittent demand: non-zero rate 28/91, Croston-style mean non-zero size 91, days since last sale
  - price (`price_column`): vs trailing 28-day mean, vs trailing 364-day max, 1- and 7-day change
  - promo / closures (known-future flags, backward-looking): promo streak, days since promo, days since
    closed, closure streak ending yesterday
  - group aggregates (`spec.agg_groups`, chosen at fit): 28-day rolling mean of the group's daily mean
    target at the lag origin + the series' ratio to it. Groups: store and store×hierarchy (M5
    `dept_id`, `cat_id` via `id_columns`) when stores hold many items; store type / assortment across
    stores when each store is one series (Rossmann). Forecasts load the model's training scope so
    aggregates match training, then filter output to the requested store.
  - store metadata (Rossmann `store.csv`), known-future covariates, categorical ids
  - seasonal / conditional (`_seasonal_and_conditional`): same-weekday 13-week mean, `lag_364`,
    15-day mean around last year's date, rolling mean over open days only, and 91-day mean target on
    promo vs non-promo days + their lift
  - `seq_0..seq_{L-1}`: target window ending at the lag origin, only when a sequence model needs it
- **Closed-day rule** (`spec.zero_when_closed`, default on, needs `open_column`): closed days are dropped
  from training and every family's `predict`/`predict_quantiles` returns exactly 0 where Open == 0
  (`_with_closed_rule`, applied via `ForecastModel.__init_subclass__`).
- Training options per run (`training_config`): `history_days`, `use_store_features`.
- **Honest tuning** (`training.tune`): Optuna trials are scored on `inner_split` — the last
  `validation_days` of the *training* period — never on the reported validation window. For GBDT the
  best trial's round count is scaled to the full training set and fixed (`early_stopping_rounds=0`);
  for neural nets the best epoch count is fixed (`patience=0`). The final validation metrics stay
  out-of-sample; `metrics.tuning` records the scheme. LightGBM also
  accepts `target_transform="log1p"` and `early_stopping_rounds<=0` (disable).
- Validation = last `validation_days` (default = horizon). Metrics (`ml/metrics.py`): MAE, RMSE, WAPE,
  RMSSE (M5 definition: per-series RMSE scaled by the training-period one-step naive error, averaged
  over series), R², MAPE, sMAPE, RMSPE (Rossmann), bias. Tune on any of wape/rmse/mae/rmspe/rmsse.
- Intervals: quantile models give P10/P50/P90 directly (pinball loss + coverage reported); otherwise an
  empirical P10–P90 band from relative validation residuals (in-sample coverage, plus
  `coverage_holdout`: calibrated on the first half of validation, measured on the second; pinball
  losses treat the band as P10/P90 and the point forecast as P50). Aggregated
  forecast bands are sums of per-series bands — conservative for the total.
- Families (`ml/families.py`; add one: subclass `ForecastModel`, register in `FAMILIES`, `ModelFamily`
  enum, frontend `types.ts`; tabular models read `spec.feature_columns`, sequence models
  `spec.model_columns` and declare `required_seq_len`):
  - `seasonal_naive` — baseline.
  - `lightgbm` — `objective=auto` → Tweedie (`tweedie_variance_power`, tunable 1.05–1.9) when the target
    is zero-heavy (>30 % zeros, non-negative), else L2; `quantiles=auto` trains P10/P50/P90 boosters
    for such targets (sorted to remove crossing).
  - `torch_mlp` — embeddings + MLP over the same tabular features as LightGBM (wide-range numeric
    inputs get signed log1p, `feature_log`). `loss=auto` → Poisson (log-rate output) for zero-heavy
    targets, else MSE in z-scored `target_transform` space (`log1p` default, or `standard` = raw
    units); decoding clips to [0, 5× training max] so expm1 can't overflow. `quantiles` (auto/true/false)
    adds P10/P50/P90 heads trained with pinball loss in the same space (monotone inverse → valid
    quantiles, sorted, ≥ 0). `patience<=0` trains a fixed `max_epochs`; `best_epoch` is recorded.
    Batch capped for ≥50 steps/epoch.
  - `torch_gru` — GRU over the `seq_len`-day sales window (scaled by its open-day mean) plus open /
    promo history channels (`seqx_{col}_{k}`, from `spec.seq_covariates`), fused with the tabular path.
  - `ensemble` — trains `members` (default LightGBM + GRU) and blends them. Weights (grid search for
    2 members, NNLS otherwise) are fitted on the **first half** of the validation window; members,
    equal-weight and fitted blends are all scored on the **second half** (`metrics.ensemble`). Torch
    members pickle via `__getstate__` (state-dict blob), so ensembles are single `.pkl` artifacts.
- Event forecasts override known-future covariates on date ranges and diff against a baseline run.
  Future covariates resolve as: auxiliary joins (real future values) → `covariate_defaults` → last
  observed same-weekday value. Id categoricals carry forward before joins run.

## GPU training workflow

1. Host: NVIDIA driver + NVIDIA Container Toolkit (`docker run --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`).
2. `docker compose --profile gpu up --build` starts `celery-worker-gpu` (Dockerfile.gpu, CUDA 12.4 torch wheel,
   `TRAINING_DEVICE=cuda`, concurrency 1, listens on `training`). To keep training off the CPU worker,
   set its `-Q` to `ingestion,forecasting`.
3. Submit `model_family=torch_mlp`, `torch_gru` or `ensemble`; the run's `device` column records `cuda`/`cpu`, `epoch_logs` carry
   per-epoch loss/WAPE, checkpoints save as `model.pt` (state_dict + preprocessing stats, loaded with `map_location`).
4. Native (no Docker): `pip install torch --index-url https://download.pytorch.org/whl/cu124`, then run a
   worker with `-Q training -c 1`. Verify with `python -c "import torch; print(torch.cuda.is_available())"`.
5. LightGBM runs on CPU (pip wheel); tune `num_threads` via hyperparameters if sharing a host.

## Coding conventions

**Tenancy & security (non-negotiable)**
- Never fetch a tenant-owned row by id alone — use `services/tenancy.get_owned(session, Model, id, org_id)`
  or `scoped(Model, org_id)`. Foreign rows return 404, not 403.
- Every new tenant table gets `organization_id` (FK, indexed, `ondelete=CASCADE`).
- Role gates via `AnalystPrincipal` / `AdminPrincipal` deps (rank: viewer < analyst < admin < owner).
- Storage keys always come from `ArtifactKeys`; never build paths from user input.

**Errors** — raise `AppError` subclasses (`NotFoundError`, `ConflictError`, `DomainValidationError`,
`PreconditionFailedError`, `PermissionDeniedError`, `AuthenticationError`). Never return ad-hoc error
dicts. Everything leaves as `{"error": {code, message, details, request_id}}`.

**No mock data in production paths** — finance, diagnostics, training and forecasting compute only from
stored datasets, forecasts and configs. If an input is missing (e.g. `avg_transaction_value` for a sales
target without a customers column), raise `DomainValidationError` explaining what to configure; never
substitute a guess. Fixtures/synthetic data belong in `tests/` only.

**Backend style** — Python 3.11+, full type hints, `from __future__ import annotations`; ruff (line 110);
Pydantic v2 (`model_validate`, `model_dump`, `ConfigDict(from_attributes=True)`); SQLAlchemy 2.0 typed
`Mapped[...]`. Async in the API; blocking work (pandas, file I/O) via `run_in_threadpool` or a Celery task.
Enums persist as VARCHAR + CHECK constraint (no native PG enums). Percentages are fractions in [0, 1].
Money in DB is `Numeric`; computed reports round to 2 dp.

**Migrations** — one Alembic revision per schema change; wrap explicit constraint names in `op.f()`;
keep `models/` and migrations in sync (`alembic check` against a dev DB).

**Frontend** — App Router, client components for data views; TanStack Query for all server state;
`lib/api.ts` is the only fetch path (adds JWT, refreshes once on 401, maps the error envelope to
`ApiError`). Keep `lib/types.ts` in sync with `app/schemas`. Colors only via CSS tokens in `globals.css`
(charts use `--chart-1..5`). Every data view needs loading, empty and error states.

**Tests** — add a pytest for every service function with business math (see `tests/test_finance.py`)
and extend `tests/test_pipeline.py` when touching features/families/forecasting.
