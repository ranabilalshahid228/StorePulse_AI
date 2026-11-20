#!/usr/bin/env bash
# SLURM template - submit from the repo root:   sbatch backend/scripts/gpu/train.sbatch [extra train_cli flags]
# Adjust partition / account / time to your cluster (sinfo, sacctmgr show assoc user=$USER).
#SBATCH --job-name=storepulse-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/slurm-%j.log
##SBATCH --partition=gpu
##SBATCH --account=your-project
set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
VENV="${VENV_DIR:-$REPO/.venv-gpu}"
mkdir -p "$REPO/outputs"
source "$VENV/bin/activate"
export MPLCONFIGDIR="$REPO/outputs/.mpl" PYTHONUNBUFFERED=1
nvidia-smi || true

cd "$REPO/backend"
python -m scripts.train_cli train \
  --dataset "${DATASET:-rossmann}" --data-dir "${DATA_DIR:-$REPO/data/raw}" \
  --families ${FAMILIES:-lightgbm torch_mlp torch_gru ensemble} \
  --horizon "${HORIZON:-42}" --epochs "${EPOCHS:-30}" --device cuda \
  --output-dir "$REPO/outputs" --run-name "${RUN_NAME:-${DATASET:-rossmann}_h${HORIZON:-42}_${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}}" \
  "$@"
