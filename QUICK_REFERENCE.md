# Quick Reference

Command cheat-sheet. Run everything from the project root with the virtualenv
active (`.venv\Scripts\activate` on Windows, `source .venv/bin/activate` on
POSIX) — or prefix each command with `.venv\Scripts\python.exe`.

---

## Pipeline (`scripts/pipeline.py --stage <name>`)

| Stage | What it does |
|---|---|
| `collect` | Fetch NASA POWER hourly data (UTC), shift to local time, write `data/raw/hyderabad_nasa_*.csv` |
| `collect-openmeteo` | Fetch the Open-Meteo ERA5 archive → `data/raw/hyderabad_openmeteo_hourly.csv` |
| `compare-sources` | Fetch Open-Meteo (if needed) + write `data/processed/source_comparison.json` and `.png` |
| `clean` | `-999` sentinels → NaN, hourly continuity, pressure kPa→hPa, short-gap interpolation, `data_quality_report.json` |
| `features` | Solar geometry + clear-sky + lag/rolling + Open-Meteo `om_ghi`/`om_dhi` → `data/processed/forecast_dataset.csv` (71 cols) |
| `train` | Benchmark all 8 models on validation, testing **and** prospective |
| `evaluate` | `analyze_models.py` + `generate_report.py` → `models/best_models.json` |
| `diagnostics` | `model_diagnostics.py` → `models/feature_importance_matrix.csv` / `.json` |
| `full` | `clean → features → train → evaluate → diagnostics` |
| `all` | `collect` first, then `full` |

```bash
python scripts/pipeline.py --stage full          # rebuild everything from raw
python scripts/pipeline.py --stage features      # just re-engineer features
python scripts/pipeline.py --stage compare-sources
```

## Model training (`scripts/train_models.py`)

| Task | Command |
|---|---|
| Benchmark all models, one split | `python scripts/train_models.py --models persistence,ridge,random_forest,extra_trees,gradient_boost,xgboost,lightgbm,catboost --split validation` |
| One model, one split | `python scripts/train_models.py --models catboost --split testing` |
| Specific horizons (days) | `python scripts/train_models.py --models lightgbm --horizons 1,7,30 --split validation` |
| One target | `python scripts/train_models.py --targets GHI --split validation` |
| Prospective, winners only | `python scripts/train_models.py --split prospective --select-best` |

Flags: `--targets` (default `GHI,temperature`), `--horizons` (default `1,2,3,5,7,10,14,30`),
`--models` (default `ridge,random_forest`), `--split` `{training,testing,validation,prospective}`,
`--select-best`, `--config`, `--data`.

## Analysis & reporting

| Task | Command | Output |
|---|---|---|
| Model comparison + best-model selection | `python scripts/analyze_models.py` | `models/best_models.json`, `{mae,rmse,r2}_pivot.csv` |
| Comparison report + generalization gap | `python scripts/generate_report.py` | stdout |
| Feature importance | `python scripts/model_diagnostics.py` | `models/feature_importance_matrix.csv` / `.json`, `importance_*.csv` |
| NASA POWER vs Open-Meteo agreement | `python scripts/compare_sources.py` | `data/processed/source_comparison.json` / `.png` |

## API + dashboard

```bash
# Terminal 1 - API (http://localhost:8000, docs at /docs)
.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000

# Terminal 2 - Streamlit dashboard (http://localhost:8501)
.venv\Scripts\python.exe -m streamlit run dashboard\app.py

# Or both via Docker
docker compose up -d --build       # docker compose down to stop
```

Key endpoints: `/health`, `/api/config`, `/api/results`, `/api/best-models`,
`/api/generalization-gap`, `/api/feature-importance`, `/api/forecast`,
`/api/forecast/production` (May 2026 hourly forecast).

```bash
curl "http://localhost:8000/api/forecast/production?target=GHI&horizon_hours=168&limit=744"
```

## Quality checks (what CI runs)

```bash
pylint --fail-under=9.5 scripts api dashboard
pyright --project pyrightconfig.json
pytest -q                                  # 39 tests
```

`black scripts/ api/` and `isort scripts/ api/` are configured (`pyproject.toml`)
but not enforced by CI.

---

## Key files

| Path | Purpose |
|---|---|
| `config/config.yaml` | Location, data splits, horizons, physical ranges, Open-Meteo settings |
| `data/raw/*.csv` | NASA POWER + Open-Meteo hourly downloads (git-ignored) |
| `data/processed/forecast_dataset.csv` | Engineered features, 57,696 rows × 71 cols (git-ignored) |
| `data/processed/data_quality_report.json` | Cleaning report |
| `data/processed/source_comparison.json` | NASA POWER vs Open-Meteo agreement |
| `models/*.joblib` | Trained artifacts, `{target}_{d}d_{model}_{split}.joblib` (git-ignored) |
| `models/training_results.json` | Every (target, horizon, model, split) metric |
| `models/best_models.json` | Lowest-RMSE model per target/horizon (validation) |
| `models/feature_importance.json` | Per-config feature importances |
| `.github/workflows/ci.yml` | pylint + pyright + pytest on Python 3.13 |
| `requirements.txt` / `requirements-ci.txt` | full stack / lean CI subset |

## Models

`persistence` (t-24 h baseline), `ridge`, `random_forest`, `extra_trees`,
`gradient_boost`, `xgboost`, `lightgbm`, `catboost`. Artifacts are
`sklearn.Pipeline(SimpleImputer, <estimator>)`; persistence is a plain dict.

## Forecast horizons

1, 2, 3, 5, 7, 10, 14, 30 days (`config.yaml → forecast.horizons`, expressed as
hours in the API: 24 … 720).
