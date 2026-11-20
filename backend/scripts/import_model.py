"""Register a checkpoint trained with scripts/train_cli.py (e.g. on a university GPU) as a finished
model run in a StorePulse workspace, so the web app (Model Lab, forecasts, Store Autopilot) can use it.

The checkpoint must come from the same data as the target dataset (same mapping and store codes);
the script checks the store codes and warns when the data's last date differs.

    # against the lean local stack (picks up its DB settings automatically)
    .venv\\Scripts\\python -m scripts.import_model --model-dir ..\\outputs\\rossmann_h42\\ensemble \\
        --dataset-id <dataset uuid from the Data page URL> --name "GPU ensemble h42"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True, help="outputs/<run>/<family> written by train_cli")
    ap.add_argument("--dataset-id", required=True, help="target dataset (its workspace owns the new run)")
    ap.add_argument("--name", help="run name shown in Model Lab")
    ap.add_argument("--no-local-stack", action="store_true", help="use POSTGRES_* / STORAGE_* from the env")
    args = ap.parse_args()

    if not args.no_local_stack and "POSTGRES_PORT" not in os.environ:
        from scripts.local_stack import lean_env

        os.environ.update(
            {
                k: v
                for k, v in lean_env().items()
                if k not in os.environ or k.startswith(("POSTGRES", "STORAGE"))
            }
        )

    import pyarrow.compute as pc
    import pyarrow.dataset as pads

    from app.db.session import sync_session
    from app.ml.families import ForecastModel
    from app.models import Dataset, ModelRun
    from app.models.enums import JobStatus
    from app.services.storage import ArtifactKeys, get_storage

    model_dir = Path(args.model_dir).resolve()
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    logs = json.loads((model_dir / "training_log.json").read_text(encoding="utf-8"))
    ckpt = model_dir / cfg["checkpoint"]
    model = ForecastModel.load(ckpt)  # proves the checkpoint loads before touching the DB

    with sync_session() as s:
        ds = s.get(Dataset, uuid.UUID(args.dataset_id))
        if ds is None or ds.status != "ready" or not ds.parquet_path:
            raise SystemExit("dataset not found or not ready")
        org_id = ds.organization_id
        table = pads.dataset(get_storage().local_path(ds.parquet_path), format="parquet").to_table(
            columns=["ds", "store_code"]
        )
        codes = set(pc.unique(table["store_code"]).to_pylist())
        trained = set(model.spec.store_static) or None
        if trained and not codes & trained:
            raise SystemExit("none of the dataset's store codes appear in the checkpoint's training data")
        data_end = str(pc.max(table["ds"]).as_py())[:10]
        ckpt_end = (
            str(pc.max(pads.dataset(cfg["canonical"]).to_table(columns=["ds"])["ds"]).as_py())[:10]
            if Path(cfg["canonical"]).exists()
            else None
        )
        if ckpt_end and ckpt_end != data_end:
            print(f"warning: checkpoint data ends {ckpt_end}, dataset ends {data_end}")

        run_id = uuid.uuid4()
        key = ArtifactKeys.model_artifact(org_id, run_id, model.artifact_ext)
        get_storage().put_file(key, ckpt)
        s.add(
            ModelRun(
                id=run_id,
                organization_id=org_id,
                dataset_id=ds.id,
                name=args.name or f"{cfg['family']} (imported from {model_dir.parent.name})",
                model_family=cfg["family"],
                run_type="train",
                horizon_days=cfg["horizon"],
                validation_days=cfg["validation_days"],
                hyperparameters=cfg["params"],
                metrics=metrics,
                epoch_logs=logs,
                feature_columns=model.spec.feature_columns,
                training_config={
                    "history_days": cfg.get("history_days"),
                    "use_store_features": cfg.get("use_store_features", True),
                    "imported_from": str(model_dir),
                },
                artifact_path=key,
                device=cfg.get("device"),
                status=JobStatus.SUCCEEDED.value,
                progress_pct=100.0,
                stage="done",
                status_message="imported checkpoint",
            )
        )
    print(f"imported {cfg['family']} as model run {run_id} (accuracy {1 - metrics['wape']:.2%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
