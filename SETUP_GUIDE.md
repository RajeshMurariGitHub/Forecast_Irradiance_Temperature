# Setup Guide

Environment setup, install, and first run. For day-to-day commands see
[QUICK_REFERENCE.md](QUICK_REFERENCE.md); for the project overview see
[README.md](README.md).

---

## 1. Python environment

Requires **Python 3.11+** (CI runs 3.13). Use a project virtualenv — do not
install into a base Anaconda environment.

```powershell
cd C:\Users\rajes\Projects\Forecast_Irradiance_Temperature
python -m venv .venv
.venv\Scripts\Activate.ps1        # PowerShell   (source .venv/bin/activate on POSIX)
python -m pip install --upgrade pip
```

## 2. Install dependencies

```powershell
# Full stack (training, dashboard, notebooks, deep-learning extras)
pip install -r requirements.txt

# OR the lean subset CI uses (no tensorflow/torch/jupyter) - enough to run the
# pipeline, API, dashboard and tests
pip install -r requirements-ci.txt
```

Verify:

```powershell
python verify_dependencies.py
```

## 3. Get the data and build models

The raw CSVs and trained `.joblib` artifacts are **not** committed (see
`.gitignore`). Regenerate them:

```powershell
# Everything from scratch: fetch NASA POWER -> clean -> features -> train -> evaluate -> diagnostics
python scripts/pipeline.py --stage all

# If data/raw/*.csv already exist, skip the download:
python scripts/pipeline.py --stage full
```

`--stage all` also needs an internet connection (NASA POWER + Open-Meteo APIs,
no keys required). A full run trains 8 models × 8 horizons × 2 targets × 3
splits and takes ~1.5-2.5 h; individual stages
(`clean`, `features`, `train`, `evaluate`, `diagnostics`) can be run alone.

## 4. Run the app

```powershell
# Terminal 1
.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
# Terminal 2
.venv\Scripts\python.exe -m streamlit run dashboard\app.py
```

API at http://localhost:8000/docs, dashboard at http://localhost:8501. The
dashboard reads `API_URL` (default `http://localhost:8000`). Docker Compose
(`docker compose up -d --build`) starts both.

## 5. Quality checks

```powershell
pylint --fail-under=9.5 scripts api dashboard
pyright --project pyrightconfig.json
pytest -q
```

These three are the CI gates (`.github/workflows/ci.yml`). `black` and `isort`
are configured in `pyproject.toml` but not enforced.

---

## Configuration files

| File | Purpose |
|---|---|
| `config/config.yaml` | Location, data splits, forecast horizons, physical ranges, Open-Meteo settings — the only file you normally edit |
| `pyrightconfig.json` | Type checking (basic mode, Python 3.13); pandas-stub noise categories downgraded to warnings |
| `.pylintrc` | Line length 100; `init-hook` puts `scripts/` on `sys.path` for cross-script imports |
| `pyproject.toml` | Project metadata + black/isort/mypy/pytest config |
| `.vscode/settings.json` | Editor interpreter + analysis paths |

## VS Code / Pylance

If imports show as unresolved even though the code runs:

1. `Ctrl+Shift+P` → **Python: Select Interpreter** → pick `.venv\Scripts\python.exe`.
2. `Ctrl+Shift+P` → **Python: Restart Language Server**.
3. `pyrightconfig.json` already adds `scripts/` and `scripts/utils/` to the
   analysis path, so `from scripts.utils import ...` and bare `import
   analyze_models` (in the report scripts) resolve.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError` in the terminal | Wrong interpreter — `python -c "import sys; print(sys.executable)"` should point inside `.venv` |
| `No module named 'catboost'` during training | `pip install catboost` (in `requirements.txt`, sometimes skipped) |
| Dashboard: *"Could not reach the API"* | The API terminal isn't running or crashed |
| Port 8000/8501 in use | `Get-NetTCPConnection -LocalPort 8000` → `Stop-Process -Id <pid> -Force` |
| `/api/forecast*` returns *"No trained result"* / *"Model not found"* | Run `python scripts/pipeline.py --stage train` first |
| Training slow | Run one split/model/horizon at a time, e.g. `--models lightgbm --horizons 1 --split validation` |

## References

- NASA POWER API — https://power.larc.nasa.gov/docs/
- Open-Meteo Historical Weather API — https://open-meteo.com/en/docs/historical-weather-api
- pvlib (solar geometry / clear-sky) — https://pvlib-python.readthedocs.io/
