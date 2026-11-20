#!/usr/bin/env bash
# One-time environment for standalone training on a Linux GPU host (no Docker, no servers).
#
#   bash backend/scripts/gpu/setup_env.sh                 # CUDA 12.4 wheels, venv at <repo>/.venv-gpu
#   CUDA_TAG=cu121 VENV_DIR=$SCRATCH/sp-venv bash backend/scripts/gpu/setup_env.sh
#
# CUDA_TAG must match what the driver supports (nvidia-smi shows "CUDA Version: 12.x"):
# cu118 | cu121 | cu124 | cu126.  Caches go next to the venv, not into $HOME, to respect quotas.
set -euo pipefail

CUDA_TAG="${CUDA_TAG:-cu124}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENV="${VENV_DIR:-$REPO/.venv-gpu}"
PYTHON="${PYTHON:-python3}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$(dirname "$VENV")/.pip-cache}"

# Needs a *final* Python >= 3.11: Ubuntu 22.04's python3.11 is 3.11.0rc1, which recent PyTorch rejects
# (missing sys.get_int_max_str_digits). If the interpreter is unsuitable and uv exists, use a uv-managed one.
py_ok() { "$1" -c 'import sys; v = sys.version_info; sys.exit(0 if v >= (3, 11) and v.releaselevel == "final" else 1)' 2>/dev/null; }
if ! py_ok "$PYTHON"; then
  if command -v uv >/dev/null; then
    echo "$PYTHON is not a final Python >= 3.11 - installing one with uv (user-local, no root)"
    uv python install 3.11
    PYTHON="$(uv python find 3.11)"
  else
    echo "Python 3.11+ (final release) required: module load python/3.11, or install uv (pip install --user uv)" >&2
    exit 1
  fi
fi
"$PYTHON" --version
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || \
  echo "warning: nvidia-smi not found - on SLURM clusters GPUs are only visible inside a GPU job"

"$PYTHON" -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip wheel
"$VENV/bin/pip" install torch --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
"$VENV/bin/pip" install -e "$REPO/backend"
"$VENV/bin/python" - <<'PY'
import torch
print("torch", torch.__version__, "| CUDA build", torch.version.cuda, "| GPU visible:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
PY
echo "done: source $VENV/bin/activate"
