"""Artifact storage: local filesystem or S3-compatible (AWS S3, MinIO, R2...).

Keys are always POSIX-style and tenant-prefixed - see `ArtifactKeys`:

    orgs/{org_id}/datasets/{dataset_id}/raw/{filename}
    orgs/{org_id}/datasets/{dataset_id}/canonical.parquet
    orgs/{org_id}/models/{model_run_id}/model.{pkl|pt}
    orgs/{org_id}/models/{model_run_id}/plots/{name}.png
    orgs/{org_id}/forecasts/{forecast_run_id}/forecast.parquet
    orgs/{org_id}/reports/{forecast_run_id}/{name}.{pdf|csv}

Workers that need a real file path call `local_path(key)`; on S3 the object is downloaded into
`STORAGE_CACHE_DIR` first. Writers produce a local temp file and call `put_file`.
"""

from __future__ import annotations

import shutil
import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import BinaryIO

from app.core.config import get_settings
from app.core.exceptions import NotFoundError


class ArtifactKeys:
    @staticmethod
    def _org(org_id: uuid.UUID) -> str:
        return f"orgs/{org_id}"

    @classmethod
    def dataset_raw(cls, org_id: uuid.UUID, dataset_id: uuid.UUID, filename: str) -> str:
        safe = Path(filename).name.replace(" ", "_")
        return f"{cls._org(org_id)}/datasets/{dataset_id}/raw/{safe}"

    @classmethod
    def dataset_parquet(cls, org_id: uuid.UUID, dataset_id: uuid.UUID) -> str:
        return f"{cls._org(org_id)}/datasets/{dataset_id}/canonical.parquet"

    @classmethod
    def model_artifact(cls, org_id: uuid.UUID, run_id: uuid.UUID, ext: str) -> str:
        return f"{cls._org(org_id)}/models/{run_id}/model.{ext}"

    @classmethod
    def model_plot(cls, org_id: uuid.UUID, run_id: uuid.UUID, name: str) -> str:
        return f"{cls._org(org_id)}/models/{run_id}/plots/{name}.png"

    @classmethod
    def forecast_output(cls, org_id: uuid.UUID, forecast_id: uuid.UUID) -> str:
        return f"{cls._org(org_id)}/forecasts/{forecast_id}/forecast.parquet"

    @classmethod
    def report(cls, org_id: uuid.UUID, forecast_id: uuid.UUID, name: str) -> str:
        return f"{cls._org(org_id)}/reports/{forecast_id}/{name}"


class ArtifactStorage(ABC):
    @abstractmethod
    def put_stream(self, key: str, stream: BinaryIO) -> int: ...

    @abstractmethod
    def put_file(self, key: str, path: Path) -> None: ...

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def local_path(self, key: str) -> Path:
        """Return a readable local file for `key` (downloading to cache if remote)."""

    @abstractmethod
    def iter_bytes(self, key: str, chunk_size: int = 1024 * 1024) -> Iterator[bytes]: ...

    @abstractmethod
    def delete_prefix(self, prefix: str) -> None: ...


class LocalStorage(ArtifactStorage):
    def __init__(self, root: str) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        p = (self.root / key).resolve()
        if self.root not in p.parents and p != self.root:
            raise ValueError("Storage key escapes root")
        return p

    def put_stream(self, key: str, stream: BinaryIO) -> int:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as fh:
            shutil.copyfileobj(stream, fh, length=4 * 1024 * 1024)
        return p.stat().st_size

    def put_file(self, key: str, path: Path) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        if path.resolve() != p:
            shutil.copyfile(path, p)

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def local_path(self, key: str) -> Path:
        p = self._path(key)
        if not p.exists():
            raise NotFoundError(f"Artifact not found: {key}")
        return p

    def iter_bytes(self, key: str, chunk_size: int = 1024 * 1024) -> Iterator[bytes]:
        with self.local_path(key).open("rb") as fh:
            while chunk := fh.read(chunk_size):
                yield chunk

    def delete_prefix(self, prefix: str) -> None:
        p = self._path(prefix)
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()


class S3Storage(ArtifactStorage):
    def __init__(self) -> None:
        import boto3

        s = get_settings()
        self.bucket = s.s3_bucket
        self.cache = Path(s.storage_cache_dir).resolve()
        self.cache.mkdir(parents=True, exist_ok=True)
        self.client = boto3.client(
            "s3",
            endpoint_url=s.s3_endpoint_url,
            region_name=s.s3_region,
            aws_access_key_id=s.s3_access_key_id,
            aws_secret_access_key=s.s3_secret_access_key,
        )

    def put_stream(self, key: str, stream: BinaryIO) -> int:
        self.client.upload_fileobj(stream, self.bucket, key)
        return int(self.client.head_object(Bucket=self.bucket, Key=key)["ContentLength"])

    def put_file(self, key: str, path: Path) -> None:
        self.client.upload_file(str(path), self.bucket, key)

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False

    def local_path(self, key: str) -> Path:
        target = self.cache / key
        if not target.exists():
            if not self.exists(key):
                raise NotFoundError(f"Artifact not found: {key}")
            target.parent.mkdir(parents=True, exist_ok=True)
            self.client.download_file(self.bucket, key, str(target))
        return target

    def iter_bytes(self, key: str, chunk_size: int = 1024 * 1024) -> Iterator[bytes]:
        body = self.client.get_object(Bucket=self.bucket, Key=key)["Body"]
        yield from body.iter_chunks(chunk_size)

    def delete_prefix(self, prefix: str) -> None:
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            objs = [{"Key": o["Key"]} for o in page.get("Contents", [])]
            if objs:
                self.client.delete_objects(Bucket=self.bucket, Delete={"Objects": objs})


@lru_cache
def get_storage() -> ArtifactStorage:
    s = get_settings()
    return S3Storage() if s.storage_backend == "s3" else LocalStorage(s.storage_local_root)
