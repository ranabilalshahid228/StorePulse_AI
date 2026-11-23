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

## Forecast benchmarks

In Model Lab, train **ARIMA**, **SARIMA**, or the weekday baseline on the same dataset, stores,
forecast horizon, and validation window as your other models. Compare their WAPE and RMSSE on the
leaderboard. ARIMA defaults to (1,1,1), and SARIMA adds a weekly seasonal AR term. Both run on CPU
and fit one model per store/item series. Start with one store for item-level data.

The benchmarks use historical demand and the configured closure rule. They do not estimate
promotion or price effects. Short or non-convergent series use a weekday-average fallback, with
fit and fallback counts recorded in the training log.

From `backend/`, you can also run:
```bash
python -m scripts.run_pipeline rossmann --stores 1 2 3 --families seasonal_naive arima sarima lightgbm
```

### Rossmann backtest

Each model was trained on the history preceding each of three 28-day holdouts from May through
July 2015. The sample included 24 stores, with six from each store type. LightGBM used available
promotion, calendar, and store features.

| Model | Mean WAPE | Mean R² |
| --- | ---: | ---: |
| LightGBM | 8.93% | 0.9549 |
| Weekday baseline | 15.24% | 0.9011 |
| SARIMA | 18.82% | 0.8566 |
| ARIMA | 19.58% | 0.8275 |

## Try it with your own store

Upload daily store-level sales, map the columns, and configure your store's costs and staffing
assumptions. See [store onboarding and pilot deployment](docs/PUBLIC_PILOT.md) for the expected
CSV columns and a synthetic demo that runs without Rossmann data.
