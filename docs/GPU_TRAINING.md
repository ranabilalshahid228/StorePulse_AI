# Training on a university GPU

Train, tune, evaluate and forecast on a remote Linux GPU machine with `backend/scripts/train_cli.py` —
no web server, database, Redis or Docker there — then copy the run folder back and (optionally) import
the model into your local StorePulse app.

```
laptop (D:\StorePulse_AI)                 GPU host (~/StorePulse_AI)
  git push  ───────────────────────────▶  git pull
  scp data\raw\…  ─────────────────────▶  data/raw/…            (data is never in git)
                                           setup_env.sh  (once)
                                           train_cli train … --device cuda
  scp outputs/<run>  ◀──────────────────  outputs/<run>/          (git-ignored)
  train_cli list / evaluate / predict
  import_model → Store Autopilot
```

## 1. Code: sync through GitHub

On the laptop, commit and `git push` (repository `ranabilalshahid228/StorePulse_AI`). On the GPU host:

```bash
git clone git@github.com:ranabilalshahid228/StorePulse_AI.git ~/StorePulse_AI   # first time
cd ~/StorePulse_AI && git pull                                                   # afterwards
```

If the repository is private, add an SSH key from the GPU host to your GitHub account
(`ssh-keygen -t ed25519`, then GitHub → Settings → SSH keys) or use a fine-grained token over HTTPS.
Put the code on the filesystem your cluster recommends for jobs (often `$WORK` or `$SCRATCH`, not `$HOME`).

## 2. SSH access

A `~/.ssh/config` entry on Windows (`C:\Users\<you>\.ssh\config`) keeps commands short. Many
universities require the VPN or a gateway host first:

```
Host gpu
    HostName gpu-node.example-university.de
    User your-login
    # ProxyJump login-gateway.example-university.de    # if the GPU node is behind a gateway
```

Then `ssh gpu`, `scp … gpu:…` work from PowerShell (OpenSSH ships with Windows).

## 3. Data: copy once

The datasets are git-ignored. Copy only what you train on:

```powershell
ssh gpu "mkdir -p ~/StorePulse_AI/data/raw"
scp -r "D:\StorePulse_AI\data\raw\rossman dataset" gpu:~/StorePulse_AI/data/raw/
scp -r "D:\StorePulse_AI\data\raw\m5" gpu:~/StorePulse_AI/data/raw/          # only if you train M5
```

(Or download from Kaggle on the host.) `--data-dir` points elsewhere if the cluster keeps data in a
shared project folder. Expected layout: `<data-dir>/rossman dataset/{train,store}.csv` and
`<data-dir>/m5/{sales_train_evaluation,calendar,sell_prices}.csv`.

## 4. Environment: once per host

```bash
cd ~/StorePulse_AI
bash backend/scripts/gpu/setup_env.sh                     # venv at .venv-gpu, torch for CUDA 12.4
# CUDA_TAG=cu121 bash backend/scripts/gpu/setup_env.sh    # match the driver (nvidia-smi → CUDA Version)
# VENV_DIR=$SCRATCH/sp-venv bash backend/scripts/gpu/setup_env.sh   # if $HOME quota is small
```

It installs PyTorch with CUDA and the backend package (`pip install -e backend`), keeps the pip cache next
to the venv, and prints whether a GPU is visible. On SLURM clusters you may need `module load python/3.11`
first, and GPUs are only visible inside a job.

## 5. Train

All commands from `backend/` with the venv active (`source ../.venv-gpu/bin/activate`).

```bash
# Rossmann, every family, 42-day horizon, one GPU
python -m scripts.train_cli train --dataset rossmann --families lightgbm torch_mlp torch_gru ensemble \
    --horizon 42 --epochs 30 --device cuda --output-dir ../outputs --run-name rossmann_h42

# 30-day horizon (Autopilot "month" plans need horizon >= 30), no promos assumed in the future forecast
python -m scripts.train_cli train --dataset rossmann --families ensemble --horizon 30 --device cuda \
    --covariate-default Promo=0 --run-name rossmann_h30

# Honest Optuna tuning (trials scored on an inner split of the training period)
python -m scripts.train_cli train --dataset rossmann --families torch_gru --tune 20 --epochs 30 --device cuda

# M5, one store, last two years
python -m scripts.train_cli train --dataset m5 --stores CA_1 --history-days 730 --families lightgbm torch_gru

# Your own data: a SchemaMapping JSON (same body as PUT /datasets/{id}/mapping)
python -m scripts.train_cli train --dataset custom --train-file /data/sales.csv --mapping-file mapping.json \
    --store-file /data/stores.csv --families lightgbm
```

| Flag | Meaning |
|---|---|
| `--dataset rossmann\|m5\|custom`, `--data-dir`, `--train-file`, `--mapping-file`, `--store-file`, `--aux ID=FILE` | data source |
| `--stores`, `--history-days`, `--no-store-features` | scope |
| `--families …` | `seasonal_naive lightgbm torch_mlp torch_gru ensemble` (`--ensemble-members` to change the blend) |
| `--horizon`, `--validation-days` | forecast horizon (Rossmann default 42) and hold-out window (default = horizon) |
| `--epochs`, `--patience`, `--lgb-rounds`, `--tune N` | training budget / Optuna trials |
| `--lgb-param K=V`, `--torch-param K=V` | any family hyperparameter, e.g. `--torch-param batch_size=8192` |
| `--device auto\|cuda\|cpu` | `cuda` fails immediately if no GPU is visible |
| `--output-dir`, `--run-name`, `--overwrite`, `--no-forecast`, `--forecast-format` | outputs |

Long runs: on a plain SSH box use `bash backend/scripts/gpu/run_detached.sh <flags>` (tmux/nohup, survives
disconnects); on SLURM submit `sbatch backend/scripts/gpu/train.sbatch` from the repo root (edit partition /
account / time; `FAMILIES`, `HORIZON`, `EPOCHS`, `RUN_NAME` env vars or extra flags override defaults).
Watch progress with `tail -f outputs/<run>/train.log` and `nvidia-smi`.

## 6. What a run produces

```
outputs/<run>/
  run.json             args, git commit, host, GPU names, torch/CUDA versions, timings, leaderboard
  leaderboard.csv      accuracy (1 - WAPE), WAPE, RMSPE, RMSSE, RMSE, MAE, R², interval coverage, fit time
  train.log
  <family>/model.pt | model.pkl, config.json, metrics.json, training_log.json,
           validation.parquet, forecast.parquet (+ forecast_summary.json)
outputs/_cache/        canonical Parquet per dataset (re-used by later runs; don't copy it back)
```

Checkpoints load on CPU or GPU (`map_location` is handled), so a GPU-trained model works on the laptop.

## 7. Copy results back

From PowerShell on the laptop (copy just the run, not `_cache`):

```powershell
scp -r gpu:~/StorePulse_AI/outputs/rossmann_h42 D:\StorePulse_AI\outputs\
# many runs, resumable (WSL or Git Bash with rsync):  rsync -av --exclude _cache gpu:~/StorePulse_AI/outputs/ /d/StorePulse_AI/outputs/
```

On a SLURM cluster you can also `tar czf rossmann_h42.tgz -C outputs rossmann_h42` first and copy one file.

## 8. Use the results locally

```powershell
cd D:\StorePulse_AI\backend
.venv\Scripts\python -m scripts.train_cli list --output-dir ..\outputs                       # compare runs
.venv\Scripts\python -m scripts.train_cli evaluate --model-dir ..\outputs\rossmann_h42\ensemble  # re-score here
.venv\Scripts\python -m scripts.train_cli predict  --model-dir ..\outputs\rossmann_h42\ensemble --format csv
```

`evaluate` / `predict` rebuild the canonical Parquet from your local `data\raw` when the cached one is not
there. To plan stores with a GPU model in the app, start the local stack (see `LOCAL_DEV.md`), upload the
same data in the UI, then register the checkpoint as a finished model run of that dataset:

```powershell
.venv\Scripts\python -m scripts.import_model --model-dir ..\outputs\rossmann_h42\ensemble `
    --dataset-id <id from the /data/<id> URL> --name "GPU ensemble h42"
```

It appears in Model Lab and in the Store Autopilot model picker; plans run on your laptop's worker.

## Troubleshooting

- `--device cuda but CUDA is not available` — wrong `CUDA_TAG` for the driver, not inside a GPU job, or
  `CUDA_VISIBLE_DEVICES` empty. Check `python -c "import torch; print(torch.cuda.is_available())"`.
- Out of GPU memory — lower the batch: `--torch-param batch_size=2048`, or `--history-days 730`.
- `module not found: app` — run from `backend/` (the CLI is `python -m scripts.train_cli`).
- LightGBM always trains on CPU (pip wheel); give it cores with `--cpus-per-task` / `--lgb-param num_threads=16`.
