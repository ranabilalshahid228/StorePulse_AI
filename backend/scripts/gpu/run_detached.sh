#!/usr/bin/env bash
# For a plain SSH GPU box (no SLURM): start training detached so it survives a dropped connection.
#   bash backend/scripts/gpu/run_detached.sh --families torch_gru ensemble --horizon 42 --epochs 30
# Follow with:  tail -f outputs/<run>/train.log      (or: tmux attach -t storepulse)
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENV="${VENV_DIR:-$REPO/.venv-gpu}"
RUN_NAME="${RUN_NAME:-rossmann_$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$REPO/outputs"
CMD="source '$VENV/bin/activate' && cd '$REPO/backend' && MPLCONFIGDIR='$REPO/outputs/.mpl' \
python -m scripts.train_cli train --device cuda --output-dir '$REPO/outputs' --run-name '$RUN_NAME' $*"
if command -v tmux >/dev/null; then
  tmux new-session -d -s storepulse "bash -lc \"$CMD; exec bash\""
  echo "started in tmux session 'storepulse' -> outputs/$RUN_NAME"
else
  nohup bash -lc "$CMD" > "$REPO/outputs/$RUN_NAME.nohup.log" 2>&1 &
  echo "started with nohup (pid $!) -> outputs/$RUN_NAME, log outputs/$RUN_NAME.nohup.log"
fi
