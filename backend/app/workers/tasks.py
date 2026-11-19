"""Celery tasks. Each task loads its row, reports progress to PostgreSQL, and records results.

Tasks take the row id only - all inputs live in the database so tasks are idempotent and
re-runnable (acks_late + reject_on_worker_lost).
"""

from __future__ import annotations

import logging
import tempfile
import traceback
import uuid
from pathlib import Path

from sqlalchemy import select

from app.core.config import get_settings
from app.db.session import sync_session
from app.models import Dataset, ForecastRun, ModelRun, Store
from app.models.enums import DatasetStatus, JobStatus, ModelFamily
from app.schemas.dataset import SchemaMapping
from app.services.storage import ArtifactKeys, get_storage
from app.workers.celery_app import celery_app
from app.workers.progress import ProgressReporter

logger = logging.getLogger(__name__)


def _err(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


# ----------------------------------------------------------------------------------------
# Ingestion
# ----------------------------------------------------------------------------------------


@celery_app.task(name="storepulse.ingest_dataset", bind=True)
def ingest_dataset(self, dataset_id: str) -> dict:  # noqa: ANN001
    from app.services.diagnostics import compute_health_report
    from app.services.ingestion import convert_to_parquet, infer_frequency

    did = uuid.UUID(dataset_id)
    progress = ProgressReporter(Dataset, did)
    progress.start("starting", running_status=DatasetStatus.PROCESSING.value)
    try:
        with sync_session() as s:
            ds = s.get(Dataset, did)
            if ds is None:
                raise LookupError(f"dataset {did} not found")
            org_id, raw_key, fmt = ds.organization_id, ds.raw_path, ds.file_format
            mapping = SchemaMapping.model_validate(ds.schema_mapping)
            single_store_id = ds.store_id

        storage = get_storage()
        raw_local = storage.local_path(raw_key)
        out_key = ArtifactKeys.dataset_parquet(org_id, did)
        progress.update(0.02, "loading_auxiliary_tables", force=True)
        joins = load_mapping_joins(org_id, mapping)
        with tempfile.TemporaryDirectory() as tmp:
            out_local = Path(tmp) / "canonical.parquet"
            stats = convert_to_parquet(
                raw_local,
                fmt,
                mapping,
                out_local,
                get_settings().ingestion_chunk_rows,
                lambda f, stage: progress.update(0.05 + 0.65 * f, stage),
                joins=joins,
            )
            progress.update(0.72, "uploading_parquet", force=True)
            storage.put_file(out_key, out_local)

            progress.update(0.75, "registering_stores", force=True)
            _ensure_stores(org_id, stats["store_codes"], single_store_id)

            progress.update(0.8, "diagnostics", force=True)
            report = compute_health_report(out_local)

        freq = infer_frequency(
            stats["date_min"], stats["date_max"], stats["row_count"] / max(stats["series_count"], 1)
        )
        progress.succeed(
            status=DatasetStatus.READY.value,
            parquet_path=out_key,
            row_count=stats["row_count"],
            series_count=stats["series_count"],
            date_min=stats["date_min"],
            date_max=stats["date_max"],
            frequency=freq,
            health_report=report,
            status_message=f"{stats['row_count']:,} rows, {stats['dropped_rows']:,} dropped",
        )
        return {"rows": stats["row_count"]}
    except Exception as exc:
        logger.exception("Ingestion failed for %s", did)
        progress.fail(_err(exc), status=DatasetStatus.FAILED.value)
        raise


def load_mapping_joins(org_id: uuid.UUID, mapping: SchemaMapping) -> list:
    """Load every auxiliary table referenced by `mapping.joins` (tenant-scoped)."""
    from app.services.joins import load_join

    prepared = []
    for spec in mapping.joins:
        with sync_session() as s:
            aux = s.get(Dataset, spec.dataset_id)
            if aux is None or aux.organization_id != org_id or aux.role != "auxiliary":
                raise LookupError(f"auxiliary dataset {spec.dataset_id} not found")
            key, fmt = aux.raw_path, aux.file_format
        prepared.append(load_join(spec, get_storage().local_path(key), fmt))
    return prepared


def _ensure_stores(org_id: uuid.UUID, codes: list[str], single_store_id: uuid.UUID | None) -> None:
    """Create Store rows for codes seen in the data that the org hasn't registered yet."""
    if single_store_id is not None:
        return
    with sync_session() as s:
        existing = set(s.scalars(select(Store.code).where(Store.organization_id == org_id)))
        for code in codes:
            if code not in existing:
                s.add(
                    Store(
                        organization_id=org_id,
                        code=code,
                        name=f"Store {code}",
                        promo2=False,
                        target_type="sales",
                        currency="EUR",
                        timezone="Europe/Berlin",
                        attributes={},
                    )
                )


@celery_app.task(name="storepulse.compute_dataset_health")
def compute_dataset_health(dataset_id: str) -> dict:
    from app.services.diagnostics import compute_health_report

    did = uuid.UUID(dataset_id)
    with sync_session() as s:
        ds = s.get(Dataset, did)
        if ds is None or not ds.parquet_path:
            raise LookupError(f"dataset {did} not ready")
        key = ds.parquet_path
    report = compute_health_report(get_storage().local_path(key))
    with sync_session() as s:
        s.get(Dataset, did).health_report = report  # type: ignore[union-attr]
    return {"score": report["score"]}


# ----------------------------------------------------------------------------------------
# Training & tuning
# ----------------------------------------------------------------------------------------


def _load_run(run_id: uuid.UUID) -> dict:
    with sync_session() as s:
        run = s.get(ModelRun, run_id)
        if run is None:
            raise LookupError(f"model run {run_id} not found")
        if run.status == JobStatus.CANCELLED:
            raise RuntimeError("run was cancelled")
        ds = s.get(Dataset, run.dataset_id)
        store = s.get(Store, run.store_id) if run.store_id else None
        if ds is None or not ds.parquet_path:
            raise RuntimeError("dataset is not ready")
        return {
            "org_id": run.organization_id,
            "family": ModelFamily(run.model_family),
            "params": dict(run.hyperparameters),
            "search_space": run.search_space,
            "n_trials": run.n_trials,
            "horizon": run.horizon_days,
            "validation_days": run.validation_days,
            "store_code": store.code if store else (ds.store.code if ds.store_id and ds.store else None),
            "parquet_key": ds.parquet_path,
            "mapping": dict(ds.schema_mapping or {}),
            "history_days": (run.training_config or {}).get("history_days"),
            "store_static": (
                _store_static(s, run.organization_id)
                if (run.training_config or {}).get("use_store_features", True)
                else {}
            ),
        }


STORE_STATIC_FIELDS = (
    "store_type",
    "assortment",
    "competition_distance",
    "competition_open_since_month",
    "competition_open_since_year",
    "promo2",
    "promo2_since_week",
    "promo2_since_year",
    "promo_interval",
)


def _store_static(s, org_id: uuid.UUID) -> dict[str, dict]:  # noqa: ANN001
    """Store metadata (e.g. imported from Rossmann store.csv) keyed by store code."""
    stores = s.scalars(select(Store).where(Store.organization_id == org_id))
    return {st.code: {f: getattr(st, f) for f in STORE_STATIC_FIELDS} for st in stores}


@celery_app.task(name="storepulse.train_model")
def train_model(run_id: str) -> dict:
    from app.ml import training

    rid = uuid.UUID(run_id)
    progress = ProgressReporter(ModelRun, rid)
    progress.start()
    try:
        ctx = _load_run(rid)
        result = training.train(
            org_id=ctx["org_id"],
            run_id=rid,
            parquet_path=get_storage().local_path(ctx["parquet_key"]),
            mapping=ctx["mapping"],
            family=ctx["family"],
            params=ctx["params"],
            horizon=ctx["horizon"],
            validation_days=ctx["validation_days"],
            store_code=ctx["store_code"],
            progress=progress,
            history_days=ctx["history_days"],
            store_static=ctx["store_static"],
        )
        progress.succeed(**result)
        return {"metrics": result["metrics"]}
    except Exception as exc:
        logger.error("Training failed for %s\n%s", rid, traceback.format_exc())
        progress.fail(_err(exc))
        raise


@celery_app.task(name="storepulse.tune_model")
def tune_model(run_id: str, metric: str = "wape", timeout_s: int | None = None) -> dict:
    from app.ml import training

    rid = uuid.UUID(run_id)
    progress = ProgressReporter(ModelRun, rid)
    progress.start()
    try:
        ctx = _load_run(rid)
        result = training.tune(
            org_id=ctx["org_id"],
            run_id=rid,
            parquet_path=get_storage().local_path(ctx["parquet_key"]),
            mapping=ctx["mapping"],
            family=ctx["family"],
            base_params=ctx["params"],
            search_space=ctx["search_space"] or {},
            n_trials=ctx["n_trials"] or get_settings().optuna_default_trials,
            timeout_s=timeout_s,
            metric=metric,
            horizon=ctx["horizon"],
            validation_days=ctx["validation_days"],
            store_code=ctx["store_code"],
            progress=progress,
            history_days=ctx["history_days"],
            store_static=ctx["store_static"],
        )
        progress.succeed(**result)
        return {"best": result["metrics"].get("best_trial")}
    except Exception as exc:
        logger.error("Tuning failed for %s\n%s", rid, traceback.format_exc())
        progress.fail(_err(exc))
        raise


# ----------------------------------------------------------------------------------------
# Forecasting
# ----------------------------------------------------------------------------------------


@celery_app.task(name="storepulse.run_forecast")
def run_forecast(forecast_id: str) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from app.ml import forecasting
    from app.ml.families import ForecastModel

    fid = uuid.UUID(forecast_id)
    progress = ProgressReporter(ForecastRun, fid)
    progress.start()
    try:
        with sync_session() as s:
            fr = s.get(ForecastRun, fid)
            if fr is None:
                raise LookupError(f"forecast {fid} not found")
            run = s.get(ModelRun, fr.model_run_id)
            ds = s.get(Dataset, fr.dataset_id)
            store = s.get(Store, fr.store_id) if fr.store_id else None
            assert run is not None and ds is not None and run.artifact_path and ds.parquet_path
            scope = s.get(Store, run.store_id) if run.store_id else None
            ctx = {
                "org_id": fr.organization_id,
                "artifact": run.artifact_path,
                "metrics": dict(run.metrics),
                "parquet": ds.parquet_path,
                "store_code": store.code if store else None,
                "scope_store_code": scope.code if scope else None,
                "horizon": fr.horizon_days,
                "scenario": dict(fr.scenario),
                "mapping": dict(ds.schema_mapping or {}),
            }
        storage = get_storage()
        model = ForecastModel.load(storage.local_path(ctx["artifact"]))
        mapping = SchemaMapping.model_validate(ctx["mapping"])
        joins = load_mapping_joins(ctx["org_id"], mapping) if mapping.joins else []

        def enrich(frame):  # noqa: ANN001, ANN202 - real future covariates from auxiliary tables
            from app.services.ingestion import coerce_covariates
            from app.services.joins import apply_joins

            frame = apply_joins(frame, joins) if joins else frame
            return coerce_covariates(frame.drop(columns=mapping.helper_columns, errors="ignore"), mapping)

        sc = ctx["scenario"]
        out, summary = forecasting.run_forecast(
            model=model,
            metrics=ctx["metrics"],
            parquet_path=storage.local_path(ctx["parquet"]),
            store_code=ctx["store_code"],
            horizon=ctx["horizon"],
            covariate_defaults=sc.get("covariate_defaults", {}),
            events=sc.get("events", []),
            compare_to_baseline=sc.get("compare_to_baseline", False),
            promo_column=ctx["mapping"].get("promo_column"),
            open_column=ctx["mapping"].get("open_column"),
            price_column=ctx["mapping"].get("price_column"),
            scope_store_code=ctx["scope_store_code"],
            enrich=enrich,
            progress=progress,
        )
        key = ArtifactKeys.forecast_output(ctx["org_id"], fid)
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / "forecast.parquet"
            pq.write_table(pa.Table.from_pandas(out, preserve_index=False), local, compression="zstd")
            storage.put_file(key, local)
        progress.succeed(output_path=key, summary=summary)
        return {"rows": len(out)}
    except Exception as exc:
        logger.error("Forecast failed for %s\n%s", fid, traceback.format_exc())
        progress.fail(_err(exc))
        raise
