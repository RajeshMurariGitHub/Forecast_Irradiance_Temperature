# Solar Irradiance & Temperature Forecasting
**Location:** Hyderabad, India (17.385°N, 78.487°E)

---

## 📋 Project Overview

This project forecasts solar irradiance (GHI) and temperature for **1, 2, 3, 5, 7, 10, 14, and 30 days ahead** using:
- Historical NASA POWER weather data (2020–2026)
- Advanced EDA and feature engineering
- Multiple ML models (Ridge, Random Forest, Extra Trees, Gradient Boosting, XGBoost, LightGBM, CatBoost)
- Interactive Streamlit dashboard with forecasts

---

## 📊 Data Architecture

```
FORECAST ORIGIN (cutoff time)
        │
        ├── Historical data (lag features)
        ├── Solar geometry (calculated)
        └── Clear-sky irradiance (modeled)
        │
        ▼
      MODEL
        │
        ▼
    Monthly Forecast
        │
        ▼
   Compare with Actual
```

---

## 📅 Data Splits

| Phase | Period | Purpose |
|-------|--------|---------|
| **TRAINING** | 2020-01-01 → 2024-12-31 | Model development |
| **TESTING** | 2025-01-01 → 2025-12-31 | Model evaluation |
| **VALIDATION** | 2026-01-01 → 2026-04-30 | Final validation |
| **PROSPECTIVE** | 2026-05-01 → 2026-05-31 | True forecast (unseen) |

---

## 📁 Project Structure

```
Forecast_Irradiance_Temperature/
│
├── data/
│   ├── raw/                    # Raw NASA POWER data
│   └── processed/              # Cleaned, frozen datasets
│
├── notebooks/
│   └── 02_data_cleaning_qa.ipynb
│
├── scripts/
│   ├── utils/
│   │   ├── data_loader.py
│   │   ├── feature_engineering.py
│   │   ├── solar_geometry.py
│   │   ├── clear_sky_model.py
│   │   └── metrics.py
│   │
│   ├── pipeline.py             # Full ETL pipeline (collect → clean → features)
│   ├── train_models.py        # Model training / benchmark script
│   ├── analyze_models.py      # Summary tables and metric pivots
│   ├── generate_report.py     # Comparison report + generalization gap
│   └── model_diagnostics.py   # Feature importance extraction
│
├── models/                     # Trained model artifacts + training_results.json
│
├── api/
│   ├── main.py                 # FastAPI app (metrics, forecasts, importance)
│   └── services.py             # Data/model access helpers
│
├── dashboard/
│   └── app.py                  # Streamlit dashboard
│
├── config/
│   └── config.yaml             # Project settings
│
├── logs/                       # Execution logs
│
├── requirements.txt            # Python dependencies
├── docker-compose.yml          # Docker configuration
└── README.md                   # This file
```

---

## 🔧 Key Phases

### Phase 1: Data Collection ✓
- Collect hourly weather data from NASA POWER API
- File: `notebooks/01_data_collection.ipynb`

### Phase 2: Data Cleaning & Quality Assurance 🔄
- Check missing values, duplicates, continuity
- Validate physical ranges
- Retain nighttime GHI = 0
- Create data quality report
- File: `notebooks/02_data_cleaning_qa.ipynb`

### Phase 3: Exploratory Data Analysis (EDA) 📊
- **Temporal structure:** hourly, daily, monthly, yearly, seasonal patterns
- **Target distributions:** GHI and temperature distributions (daytime, nighttime, seasonal)
- **Weather relationships:** GHI ↔ cloud cover, humidity, temperature, DNI, DHI
- **Solar geometry analysis:** elevation, zenith, azimuth, sunrise/sunset
- **Clear-sky analysis:** actual vs clear-sky irradiance
- File: `notebooks/03_eda.ipynb`

### Phase 4: Feature Engineering 🔧
- Temporal features (hour, day, month, season, weekday)
- Lag features (1h, 3h, 6h, 12h, 24h, 48h, 7-day)
- Rolling statistics (mean, std, min, max)
- Solar geometry (elevation, zenith, azimuth)
- Clear-sky irradiance and clearness indices
- File: `notebooks/04_feature_engineering.ipynb`

### Phase 5: Forecast Dataset Building 📦
- Create train/test/validation/prospective splits
- Apply information availability rules
- Prevent data leakage
- File: `notebooks/05_forecast_dataset.ipynb`

### Phase 6: Model Development & Training 🤖
- Persistence baseline (t-24 h) + Ridge, Random Forest, Extra Trees,
  Gradient Boosting, XGBoost, LightGBM, CatBoost
- Benchmark loop over every target × horizon × model, per split
- File: `scripts/train_models.py`

### Phase 7: Model Evaluation & Selection ✅
- Comparison tables + generalization gap across all models and horizons
- Lowest-RMSE model per target/horizon written to `models/best_models.json`
- Feature-importance matrix (`models/feature_importance_matrix.csv` / `.json`)
- Files: `scripts/analyze_models.py`, `scripts/generate_report.py`, `scripts/model_diagnostics.py`

### Phase 8: Dashboard & API 🎨
- Streamlit dashboard: **Production Forecast** (May 2026, winning model badge),
  model comparison, forecast explorer, feature importance, generalization gap
- FastAPI backend: metrics, feature importance, split forecasts, and
  `/api/forecast/production` (744 hourly May 2026 predictions from the CatBoost
  prospective model, with Random Forest fallback)
- Dockerized via `docker-compose.yml` (api + dashboard services)
- Files: `api/main.py`, `api/services.py`, `dashboard/app.py`, `docker-compose.yml`

---

## 📦 Installation & Setup

### 1. Clone & Setup Environment
```bash
cd Forecast_Irradiance_Temperature
python -m venv venv
venv\Scripts\activate  # Windows
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Full Pipeline
```bash
# Already have data/raw/*.csv — clean → features → train → evaluate → diagnostics
python scripts/pipeline.py --stage full

# From scratch — also re-downloads NASA POWER data first
python scripts/pipeline.py --stage all
```
`--stage full` runs, in order:
1. **data prep** — drop `-999` sentinels, hourly continuity, pressure → hPa,
   solar geometry / clear-sky / lag / rolling features (`data/processed/forecast_dataset.csv`)
2. **train all models** — benchmark loop over `persistence, ridge, random_forest,
   extra_trees, gradient_boost, xgboost, lightgbm, catboost` on the validation and
   testing splits, then `--select-best` for the prospective (May 2026) split
3. **evaluate** — `analyze_models.py` + `generate_report.py` → `models/best_models.json`
   (lowest-RMSE model per target/horizon)
4. **diagnostics** — `model_diagnostics.py` → `models/feature_importance_matrix.csv` / `.json`

Individual steps: `--stage {collect,clean,features,train,evaluate,diagnostics}`.

### 4. Launch the Docker Network (API + Dashboard)

```bash
docker compose up -d --build      # docker compose down to stop
```
- API: http://localhost:8000 (docs at http://localhost:8000/docs)
- Dashboard: http://localhost:8501

`./models` and `./data` mount read-only, so the containers serve whatever
`--stage full` last produced.

**Run locally instead:**
```bash
# Terminal 1: start the API
.venv\Scripts\python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000

# Terminal 2: start the dashboard
.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

### 5. View the May 2026 CatBoost Forecast

In the dashboard open the **🚀 Production Forecast** tab and pick a target/horizon
in the sidebar. For GHI at 120/168/240/336 h the winning-model badge shows
**CatBoost** with its validation RMSE, and the chart plots all 744 hourly May 2026
predictions against observed values. Direct from the API:
```bash
curl "http://localhost:8000/api/forecast/production?target=GHI&horizon_hours=120"
```

---

## 🎯 Key Features & Specifications

### Data Features (Raw)
| Feature | NASA Name | Unit | Role |
|---------|-----------|------|------|
| Temperature | T2M | °C | Target/Feature |
| Dew Point | T2MDEW | °C | Feature |
| Humidity | RH2M | % | Feature |
| Cloud Cover | CLOUD_AMT | % | Feature |
| Wind Speed | WS10M | m/s | Feature |
| Precipitation | PRECTOTCORR | mm | Feature |
| Pressure | PS | hPa | Feature |
| GHI | ALLSKY_SFC_SW_DWN | W/m² | **TARGET** |
| DNI | ALLSKY_SFC_SW_DNI | W/m² | Feature |
| DHI | ALLSKY_SFC_SW_DIFF | W/m² | Feature |

### Target Variables
- **GHI (Global Horizontal Irradiance):** 1-30 day ahead forecasts
- **Temperature:** 1-30 day ahead forecasts

### Forecast Horizons
1, 2, 3, 5, 7, 10, 14, 30 days

### Evaluation Metrics
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- R² Score
- Horizon-wise metrics

---

## 🚀 Quick Start

```bash
# One command: data prep → train all models → evaluate → diagnostics
python scripts/pipeline.py --stage full
```

Run pieces individually if you prefer:
```bash
# Benchmark a single split
python scripts/train_models.py \
    --targets GHI,temperature \
    --horizons 1,2,3,5,7,10,14,30 \
    --models persistence,ridge,random_forest,extra_trees,gradient_boost,xgboost,lightgbm,catboost \
    --split validation

# Lock the best validation model per target/horizon, score the prospective split
python scripts/train_models.py --split prospective --select-best

# Reports + best-model selection + feature importance
python scripts/analyze_models.py
python scripts/generate_report.py
python scripts/model_diagnostics.py
```

```python
# Load a trained artifact directly
import joblib, json
best = json.load(open("models/best_models.json"))
model = joblib.load(best["best_per_horizon"]["GHI"]["7"]["model_path"])  # CatBoost, 7-day GHI
```

---

## 📚 References

- **NASA POWER API:** https://power.larc.nasa.gov/docs/
- **Solar Geometry:** Duffie & Beckman (2013)
- **Clear-Sky Models:** Ineichen & Perez (2002)
- **ML Frameworks:** XGBoost, LightGBM, scikit-learn, TensorFlow/Keras

---

## 👤 Author Notes

- **Location:** Hyderabad (17.385°N, 78.487°E)
- **Data Period:** 2020-01-01 → 2026-05-31 (ongoing)
- **Frequency:** Hourly
- **Data Source:** NASA POWER API

---

**Last Updated:** 2026-08-31
