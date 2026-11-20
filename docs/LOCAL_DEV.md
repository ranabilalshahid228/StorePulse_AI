# Lean local development (Windows, no Docker)

Run and verify the whole app — API, database migrations, background jobs and the Next.js UI — from
`D:\StorePulse_AI` without Docker images, Redis, or caches piling up on `C:`.

| Piece | How it runs locally | Where it writes |
|---|---|---|
| PostgreSQL 16 | binaries from the `pgserver` wheel inside `backend\.venv` (≈ 41 MB), driven by `scripts/local_stack.py` on `127.0.0.1:55432` | `.local\pgdata` |
| Celery broker | the same PostgreSQL (kombu SQLAlchemy transport, `CELERY_BROKER_URL=sqla+postgresql+psycopg://…`) — no Redis | tables `kombu_*` in the local DB |
| Worker | real `celery worker -P solo` process consuming `ingestion,training,forecasting` | — |
| API | `uvicorn app.main:app` on `:8000` | — |
| UI | `next dev` on `:3000` | `frontend\.next` |
| Uploads, models, reports | `STORAGE_LOCAL_ROOT` | `.local\storage` |
| Temp files, pip / npm / matplotlib caches | `TEMP`, `PIP_CACHE_DIR`, `npm_config_cache`, `MPLCONFIGDIR` | `.local\tmp`, `.local\pip-cache`, `.local\npm-cache`, `.local\mpl` |
| Logs | one file per process | `.local\logs` |

`.local\` and `outputs\` are git-ignored. Production still uses Redis + Docker (`docker-compose.yml`);
the Postgres broker is only meant for a single local user.

## One-time setup

Prereqs: Python 3.11+, Node 22, Git (all already installed; nothing else goes to `C:`).

```powershell
cd D:\StorePulse_AI
$env:PIP_CACHE_DIR = 'D:\StorePulse_AI\.local\pip-cache'      # keep pip's cache off C:
$env:npm_config_cache = 'D:\StorePulse_AI\.local\npm-cache'   # keep npm's cache off C:
cd backend
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev,local]"         # local = pgserver (PostgreSQL binaries)
# optional, for torch_mlp / torch_gru / ensemble on CPU:
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
cd ..\frontend
npm ci
```

## Run the stack

```powershell
cd D:\StorePulse_AI\backend
.venv\Scripts\python -m scripts.local_stack up
```

This initializes the database on first use, starts PostgreSQL, runs `alembic upgrade head`, then starts
the API, worker and UI with all paths pointed at `.local\`. Open http://localhost:3000 (API docs at
http://localhost:8000/api/v1/docs). `Ctrl+C` stops everything including PostgreSQL (`--keep-db` leaves it
running). Variants:

```powershell
.venv\Scripts\python -m scripts.local_stack up --no-ui        # backend only
.venv\Scripts\python -m scripts.local_stack db status         # or: db start | db stop | db psql
.venv\Scripts\python -m scripts.local_stack migrate           # start db + alembic upgrade head
.venv\Scripts\python -m scripts.local_stack env               # PowerShell $env: lines to run pieces by hand
.venv\Scripts\python -m scripts.local_stack clean             # delete temp files and logs
.venv\Scripts\python -m scripts.local_stack clean --all       # also the database, uploads and caches
```

Running a piece by hand (e.g. to debug the worker) — paste the output of `local_stack env` into the
terminal first, then:

```powershell
.venv\Scripts\python -m uvicorn app.main:app --reload --port 8000
.venv\Scripts\python -m celery -A app.workers.celery_app worker -Q ingestion,training,forecasting -P solo --loglevel=INFO
```

## Verify end to end

With the stack running:

```powershell
cd D:\StorePulse_AI\backend
.venv\Scripts\python -m scripts.smoke_local --stores 1 2 3 4 5
```

It registers a throwaway workspace, imports Rossmann `store.csv`, uploads and maps a 5-store slice of
`train.csv`, trains LightGBM on the worker, configures store 1, runs a 7-day Store Autopilot plan and
downloads the PDF / Excel / CSV. The login it created is written to `.local\smoke_login.txt` so you can
look around the UI with the same data.

`GET /api/v1/health/ready` should report `{"database": "ok", "broker": "ok"}`.

## Tests and checks (no stack needed)

```powershell
cd D:\StorePulse_AI\backend
.venv\Scripts\python -m pytest                    # Rossmann tests skip if data\raw is absent
.venv\Scripts\ruff check app tests scripts alembic
.venv\Scripts\ruff format --check app tests scripts alembic
cd ..\frontend
npm run typecheck
npm run build
```

With the local database running, `alembic check` confirms models and migrations are in sync
(load the env first: `.venv\Scripts\python -m scripts.local_stack env | Invoke-Expression`).

## Disk footprint

Everything above lives under `D:\StorePulse_AI` (`backend\.venv`, `frontend\node_modules`, `.local\`,
`outputs\`). `local_stack clean --all` plus deleting `backend\.venv`, `frontend\node_modules` and
`frontend\.next` returns the repo to a fresh clone. The only things on `C:` are the Python / Node / Git
installs you already had.
