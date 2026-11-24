# StorePulse AI — technical report (IEEE conference format)

`main.tex` → `main.pdf` (13 pages: IEEE two-column body and references, then single-column appendices). Template: `IEEEtran`
(`conference` option). Bibliography: `references.bib`, IEEE numbered style (`IEEEtran.bst`); every
citation, figure, table, section and appendix reference is a clickable link.

## Build

Any one of:

```powershell
# Tectonic (used here; fetches packages on first run)
tectonic main.tex

# TeX Live / MiKTeX
latexmk -pdf main.tex                       # pdfLaTeX + BibTeX
latexmk -xelatex main.tex                   # XeLaTeX + BibTeX
```

On this machine Tectonic lives in `..\.local\bin\tectonic.exe` with its cache in
`..\.local\tectonic-cache` (set `TECTONIC_CACHE_DIR`). Overleaf also works: upload the folder.

Packages: IEEEtran, cite, amsmath, graphicx, booktabs, multirow, xcolor, tikz (arrows.meta,
positioning, shapes), listings, xurl, hyperref, cleveref, balance; under XeLaTeX fontspec with the
TeX Gyre Termes font files, under pdfLaTeX newtxtext.

## Placeholders to fill in

Red text in the PDF. Edit the macros at the top of `main.tex`:
`\Program` (course / degree programme), `\Supervisor`, `\Campus` (faculty and campus).
The IEEE conference template has no logo slot, so no university logo is included in the report.

## Evidence and how to regenerate it

| Folder | Content |
|---|---|
| `data/` | Every number in the tables and charts, as CSV (see below) |
| `data/raw_runs/` | JSON + logs of the local CPU reruns (`scripts/run_pipeline.py`) |
| `figures/*.pdf` | Charts drawn by `scripts/make_figures.py` from `data/*.csv` |
| `figures/screens/` | Screenshots of the running app (demo data, 1440×900, account details hidden); crops by `scripts/crop_screens.py` |
| `scripts/` | `collect_results.py`, `bench_api.py`, `make_figures.py`, `crop_screens.py`, `contrast.py`, `verify_refs.py` |

Sources of the data files:

- `model_comparison.csv`, `residuals_by_*.csv`, `interval_coverage.csv`, `ensemble_holdout.csv` —
  GPU runs in `../outputs/rossmann_h42_gpu` and `../outputs/rossmann_h30_all` (OTH JupyterHub,
  NVIDIA RTX 6000 Ada; `run.json` holds host, versions and git commit) plus the local reruns.
  `../outputs/` is git-ignored; copy it from the GPU host or re-run `scripts/train_cli.py`.
- `model_runs_local.csv`, `datasets.csv`, `job_timings.csv` — read from the running local app's API.
- `api_latency*.csv` — `bench_api.py` (30 sequential calls per endpoint, laptop, loopback).
- `pytest_results.xml`, `pytest_output.txt` — full backend test suite (57 passed).
- `contrast.csv` — WCAG contrast of the UI theme tokens.
- `reference_check.txt` — Crossref lookup of every DOI in `references.bib` (all found).

Regenerate (from `backend/`, Python venv of the project, local stack running for the API parts):

```powershell
.venv\Scripts\python -m scripts.run_pipeline rossmann --families seasonal_naive lightgbm --horizon 42 --lgb-rounds 4000 --out ..\report\data\raw_runs\local_rossmann_h42.json
.venv\Scripts\python -m scripts.run_pipeline rossmann --families seasonal_naive lightgbm --horizon 30 --lgb-rounds 4000 --out ..\report\data\raw_runs\local_rossmann_h30.json
.venv\Scripts\python ..\report\scripts\collect_results.py
.venv\Scripts\python ..\report\scripts\bench_api.py --n 30
.venv\Scripts\python ..\report\scripts\make_figures.py
.venv\Scripts\python ..\report\scripts\crop_screens.py
.venv\Scripts\python ..\report\scripts\contrast.py
.venv\Scripts\python ..\report\scripts\verify_refs.py ..\report\references.bib
```

Screenshots were captured with Playwright from the production build (`local_stack up --present`)
of commit `d3fadc8`; re-capture by signing in to the demo workspaces listed in `.local/logins.txt`.

## Known gaps (also stated in the report)

- One fixed train/validation split per horizon; no rolling-origin error bars.
- 42-day ARIMA/SARIMA come from the local app runs (same data, stores and window); only WAPE and
  RMSSE were recorded for them.
- The ablation table reproduces values documented in `CLAUDE.md`; they were not re-measured.
- Latency is single-client on loopback; no load test.
