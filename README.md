# StorePulse AI

Multi-tenant retail forecasting & store autopilot (FastAPI · Celery · PostgreSQL · Next.js).

```powershell
Copy-Item .env.example .env   # set JWT_SECRET_KEY
docker compose up --build
```

UI: http://localhost:3000 · API docs: http://localhost:8000/api/v1/docs

- No Docker locally? [docs/LOCAL_DEV.md](docs/LOCAL_DEV.md) — `python -m scripts.local_stack up` (PostgreSQL in the venv, no Redis, everything under `.local/`).
- Train on a university GPU: [docs/GPU_TRAINING.md](docs/GPU_TRAINING.md) — `python -m scripts.train_cli train … --device cuda`.

See [CLAUDE.md](CLAUDE.md) for architecture, environment variables, dev commands, GPU workflow and conventions.
