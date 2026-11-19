# syntax=docker/dockerfile:1.7
# One image for the API and the Celery workers (CPU). For GPU workers see Dockerfile.gpu.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgomp1: LightGBM OpenMP runtime. curl: container healthchecks.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
RUN mkdir -p app && touch app/__init__.py && pip install . && rm -rf app

COPY . .
RUN pip install --no-deps . \
 && useradd --create-home --uid 10001 storepulse \
 && mkdir -p /data/storage && chown -R storepulse /data /app
USER storepulse

EXPOSE 8000
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000 --proxy-headers"]
