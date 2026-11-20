"""Lean local stack: PostgreSQL + API + Celery worker + Next.js, no Docker and no Redis.

Everything it writes lives under <repo>/.local (git-ignored): the database cluster, uploaded data
and model artifacts, the job queue (PostgreSQL is also the Celery broker), logs, temp files and the
pip / npm / matplotlib caches - nothing lands on C: beyond the tools you already installed.

PostgreSQL 16 comes from the `pgserver` wheel's bundled binaries inside backend/.venv
(pip install pgserver); this script drives initdb / pg_ctl / psql directly on a fixed port.

    cd D:\\StorePulse_AI\\backend
    .venv\\Scripts\\python -m scripts.local_stack up            # db + migrations + api + worker + ui
    .venv\\Scripts\\python -m scripts.local_stack up --no-ui    # backend only
    .venv\\Scripts\\python -m scripts.local_stack env           # PowerShell env lines for manual runs
    .venv\\Scripts\\python -m scripts.local_stack db stop       # stop PostgreSQL
    .venv\\Scripts\\python -m scripts.local_stack clean --all   # delete .local (db, uploads, caches)
"""

from __future__ import annotations

import argparse
import os
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parent
LOCAL = Path(os.environ.get("STOREPULSE_LOCAL_DIR", ROOT / ".local"))
PGDATA = LOCAL / "pgdata"
LOGS = LOCAL / "logs"
PG_PORT = int(os.environ.get("LOCAL_PG_PORT", "55432"))
API_PORT = int(os.environ.get("LOCAL_API_PORT", "8000"))
UI_PORT = int(os.environ.get("LOCAL_UI_PORT", "3000"))
DB = {"user": "storepulse", "password": "storepulse", "name": "storepulse"}
IS_WIN = os.name == "nt"


def pg_bin(name: str) -> str:
    try:
        from pgserver._commands import POSTGRES_BIN_PATH
    except ImportError as exc:  # pragma: no cover - guidance only
        raise SystemExit(
            "PostgreSQL binaries missing: run  .venv\\Scripts\\python -m pip install pgserver"
        ) from exc
    return str(Path(POSTGRES_BIN_PATH) / (f"{name}.exe" if IS_WIN else name))


def lean_env() -> dict[str, str]:
    """Process environment that keeps every cache / temp / artifact on the repo drive."""
    for d in ("tmp", "pip-cache", "npm-cache", "mpl", "storage", "logs"):
        (LOCAL / d).mkdir(parents=True, exist_ok=True)
    secret_file = LOCAL / "jwt_secret"
    if not secret_file.exists():
        secret_file.write_text(secrets.token_urlsafe(48))
    env = dict(os.environ)
    env.update(
        {
            "ENVIRONMENT": "local",
            "JWT_SECRET_KEY": env.get("JWT_SECRET_KEY") or secret_file.read_text().strip(),
            "POSTGRES_HOST": "127.0.0.1",
            "POSTGRES_PORT": str(PG_PORT),
            "POSTGRES_USER": DB["user"],
            "POSTGRES_PASSWORD": DB["password"],
            "POSTGRES_DB": DB["name"],
            # PostgreSQL as the Celery broker (kombu SQLAlchemy transport). Full URL: the celery CLI
            # also reads CELERY_BROKER_URL from the environment.
            "CELERY_BROKER_URL": (
                f"sqla+postgresql+psycopg://{DB['user']}:{DB['password']}@127.0.0.1:{PG_PORT}/{DB['name']}"
            ),
            "STORAGE_BACKEND": "local",
            "STORAGE_LOCAL_ROOT": str(LOCAL / "storage"),
            "STORAGE_CACHE_DIR": str(LOCAL / "storage" / ".cache"),
            "CORS_ORIGINS": f"http://localhost:{UI_PORT},http://127.0.0.1:{UI_PORT}",
            "NEXT_PUBLIC_API_URL": f"http://localhost:{API_PORT}/api/v1",
            "NEXT_TELEMETRY_DISABLED": "1",
            "TEMP": str(LOCAL / "tmp"),
            "TMP": str(LOCAL / "tmp"),
            "TMPDIR": str(LOCAL / "tmp"),
            "PIP_CACHE_DIR": str(LOCAL / "pip-cache"),
            "npm_config_cache": str(LOCAL / "npm-cache"),
            "MPLCONFIGDIR": str(LOCAL / "mpl"),
            "PYTHONUNBUFFERED": "1",
        }
    )
    return env


# ----------------------------------------------------------------------------------------
# PostgreSQL
# ----------------------------------------------------------------------------------------


def _run(cmd: list[str], **kw: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, **kw)  # type: ignore[call-overload]


def db_running() -> bool:
    return PGDATA.exists() and _run([pg_bin("pg_ctl"), "-D", str(PGDATA), "status"]).returncode == 0


def db_init() -> None:
    if (PGDATA / "PG_VERSION").exists():
        return
    PGDATA.parent.mkdir(parents=True, exist_ok=True)
    print(f"[db] initdb {PGDATA}")
    r = _run(
        [pg_bin("initdb"), "-D", str(PGDATA), "-U", "postgres", "-A", "trust", "-E", "UTF8", "--locale=C"]
    )
    if r.returncode:
        raise SystemExit(r.stdout + r.stderr)


def psql(sql: str, db: str = "postgres") -> str:
    r = _run([pg_bin("psql"), "-h", "127.0.0.1", "-p", str(PG_PORT), "-U", "postgres", "-d", db, "-tAc", sql])
    if r.returncode:
        raise SystemExit(r.stderr)
    return r.stdout.strip()


def db_start() -> None:
    db_init()
    LOGS.mkdir(parents=True, exist_ok=True)
    if not db_running():
        print(f"[db] starting PostgreSQL on 127.0.0.1:{PG_PORT}")
        # No captured pipes: the postmaster inherits them on Windows and run() would wait forever.
        rc = subprocess.run(
            [
                pg_bin("pg_ctl"),
                "-D",
                str(PGDATA),
                "-l",
                str(LOGS / "postgres.log"),
                "-o",
                f"-p {PG_PORT} -h 127.0.0.1",
                "-w",
                "start",
            ],
            env=lean_env(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        if rc:
            raise SystemExit(f"PostgreSQL failed to start - see {LOGS / 'postgres.log'}")
    if psql(f"SELECT 1 FROM pg_roles WHERE rolname='{DB['user']}'") != "1":
        psql(f"CREATE ROLE {DB['user']} LOGIN PASSWORD '{DB['password']}'")
    if psql(f"SELECT 1 FROM pg_database WHERE datname='{DB['name']}'") != "1":
        psql(f"CREATE DATABASE {DB['name']} OWNER {DB['user']}")
    print(f"[db] ready: postgresql://{DB['user']}@127.0.0.1:{PG_PORT}/{DB['name']}")


def db_stop() -> None:
    if db_running():
        _run([pg_bin("pg_ctl"), "-D", str(PGDATA), "-m", "fast", "-w", "stop"])
        print("[db] stopped")


def migrate() -> None:
    print("[db] alembic upgrade head")
    r = subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], cwd=BACKEND, env=lean_env())
    if r.returncode:
        raise SystemExit("migrations failed")


# ----------------------------------------------------------------------------------------
# Processes
# ----------------------------------------------------------------------------------------


def _pump(name: str, proc: subprocess.Popen[str], log: Path) -> None:
    with log.open("a", encoding="utf-8") as fh:
        for line in proc.stdout or []:
            fh.write(line)
            fh.flush()
            sys.stdout.write(f"[{name}] {line}")


def _kill(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    if IS_WIN:  # npm / uvicorn spawn children: kill the whole tree
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
    else:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()


def up(args: argparse.Namespace) -> None:
    # Windows consoles default to cp1252; child output (e.g. Next.js "▲") must never kill the pumps.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    db_start()
    migrate()
    env = lean_env()
    py = sys.executable
    procs: dict[str, subprocess.Popen[str]] = {}
    specs: list[tuple[str, list[str], Path]] = [
        (
            "api",
            [py, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(API_PORT)],
            BACKEND,
        ),
    ]
    if not args.no_worker:
        specs.append(
            (
                "worker",
                [
                    py,
                    "-m",
                    "celery",
                    "-A",
                    "app.workers.celery_app",
                    "worker",
                    "-Q",
                    "ingestion,training,forecasting",
                    "-P",
                    "solo",
                    "--loglevel=INFO",
                ],
                BACKEND,
            )
        )
    if not args.no_ui:
        npm = "npm.cmd" if IS_WIN else "npm"
        specs.append(("ui", [npm, "run", "dev", "--", "--port", str(UI_PORT)], ROOT / "frontend"))
    for name, cmd, cwd in specs:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        procs[name] = proc
        threading.Thread(target=_pump, args=(name, proc, LOGS / f"{name}.log"), daemon=True).start()
    print(
        f"\nStorePulse local stack (logs in {LOGS}):\n"
        f"  API   http://localhost:{API_PORT}/api/v1/docs\n"
        + (f"  UI    http://localhost:{UI_PORT}\n" if not args.no_ui else "")
        + "Ctrl+C to stop.\n"
    )
    try:
        while all(p.poll() is None for p in procs.values()):
            time.sleep(1)
        dead = [n for n, p in procs.items() if p.poll() is not None]
        print(f"process exited: {dead} - stopping the rest")
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs.values():
            _kill(p)
        if not args.keep_db:
            db_stop()


def print_env() -> None:
    env = lean_env()
    keys = [
        "JWT_SECRET_KEY",
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "POSTGRES_DB",
        "CELERY_BROKER_URL",
        "STORAGE_LOCAL_ROOT",
        "STORAGE_CACHE_DIR",
        "CORS_ORIGINS",
        "NEXT_PUBLIC_API_URL",
        "TEMP",
        "TMP",
        "PIP_CACHE_DIR",
        "npm_config_cache",
        "MPLCONFIGDIR",
    ]
    for k in keys:
        print(f"$env:{k} = '{env[k]}'")


def clean(all_: bool) -> None:
    if db_running():
        db_stop()
    targets = ["tmp", "logs", "mpl"] + (
        ["pgdata", "storage", "pip-cache", "npm-cache", "jwt_secret"] if all_ else []
    )
    for t in targets:
        p = LOCAL / t
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            p.unlink()
        print(f"removed {p}")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)  # progress visible when output is redirected
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("up", help="start db, migrate, run api + worker + ui")
    u.add_argument("--no-ui", action="store_true")
    u.add_argument("--no-worker", action="store_true")
    u.add_argument("--keep-db", action="store_true", help="leave PostgreSQL running after Ctrl+C")
    d = sub.add_parser("db", help="PostgreSQL only")
    d.add_argument("action", choices=["start", "stop", "status", "psql"])
    sub.add_parser("migrate", help="start db and run alembic upgrade head")
    sub.add_parser("env", help="print PowerShell env lines for running pieces by hand")
    c = sub.add_parser("clean", help="delete temp/logs (--all: also db, uploads, caches)")
    c.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if args.cmd == "up":
        up(args)
    elif args.cmd == "db":
        if args.action == "start":
            db_start()
        elif args.action == "stop":
            db_stop()
        elif args.action == "status":
            print("running" if db_running() else "stopped", f"(port {PG_PORT}, data {PGDATA})")
        else:
            db_start()
            os.execv(
                pg_bin("psql"),
                [pg_bin("psql"), "-h", "127.0.0.1", "-p", str(PG_PORT), "-U", DB["user"], DB["name"]],
            )
    elif args.cmd == "migrate":
        db_start()
        migrate()
    elif args.cmd == "env":
        print_env()
    else:
        clean(args.all)
    return 0


if __name__ == "__main__":
    sys.exit(main())
