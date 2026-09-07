# Quick Start Guide

## 0. End-to-End: pipeline → models → May 2026 forecast

All commands run from the project root with the virtualenv active
(`.venv\Scripts\activate` on Windows).

### 0.1 Build data, train every model, evaluate, and generate diagnostics

```bash
# Already have data/raw/*.csv: clean -> features -> train -> evaluate -> diagnostics
python scripts/pipeline.py --stage full

# From scratch (also re-downloads NASA POWER data first)
python scripts/pipeline.py --stage all
```

`--stage full` runs, in order:

1. `clean_and_freeze_data` – drop `-999` sentinels, hourly continuity, pressure → hPa
2. `build_forecast_dataset` – solar geometry, clear-sky, lag/rolling features
3. `train_models.py` – benchmark loop over
   `persistence, ridge, random_forest, extra_trees, gradient_boost, xgboost, lightgbm, catboost`
   on the validation and testing splits, then `--select-best` for the prospective split
4. `analyze_models.py` + `generate_report.py` – comparison tables and
   `models/best_models.json` (lowest-RMSE model per target/horizon)
5. `model_diagnostics.py` – `models/feature_importance_matrix.csv` / `.json`

Run any step alone with `--stage {clean,features,train,evaluate,diagnostics}`.

### 0.2 Spin up the Docker network (API + dashboard)

```bash
docker compose up -d --build
```

- API:       http://localhost:8000  (docs at `/docs`)
- Dashboard: http://localhost:8501

`./models` and `./data` are mounted read-only, so the containers serve whatever
`--stage full` last produced. `docker compose down` to stop.

### 0.3 View the May 2026 CatBoost forecast

Open the dashboard → **🚀 Production Forecast** tab. Pick a target and horizon in
the sidebar; for GHI at 120/168/240/336 h the winning model badge reads
**CatBoost** with its validation RMSE, and the chart shows all 744 hourly
predictions for May 2026 against the observed values.

Same data straight from the API:

```bash
curl "http://localhost:8000/api/forecast/production?target=GHI&horizon_hours=120"
```

Missing a CatBoost prospective artifact for a horizon falls back to the report's
best model, then Random Forest.

### 0.4 Run locally instead of Docker

```bash
# Terminal 1 – API
.venv\Scripts\python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000

# Terminal 2 – dashboard
.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

---

## 1. Load and Explore Data

```python
from scripts.utils import DataLoader

# Initialize loader
loader = DataLoader(config_path='config/config.yaml')

# Load raw data
df_raw = loader.load_raw_data('data/raw/hyderabad_nasa_Jan_2020_July_2026_hourly_new.csv')
print(df_raw.head())

# Load processed data (after cleaning)
df = loader.load_processed_data('data/processed/hyderabad_cleaned.csv')

# Split by period
splits = loader.split_by_period(df)
print(splits['train'].shape)
print(splits['test'].shape)
```

---

## 2. Calculate Solar Geometry

```python
from scripts.utils import SolarGeometry

# Initialize with location
sg = SolarGeometry(
    latitude=17.385, 
    longitude=78.487, 
    elevation=500, 
    timezone='Asia/Kolkata'
)

# Calculate solar position
times = df.index
solar_pos = sg.calculate_solar_position(times)
print(solar_pos[['solar_elevation', 'solar_azimuth']].head())

# Add all solar features to dataframe
df = sg.add_solar_features(df)
```

---

## 3. Calculate Clear-Sky Irradiance

```python
from scripts.utils import ClearSkyModel

# Initialize model
csm = ClearSkyModel(
    latitude=17.385,
    longitude=78.487,
    elevation=500,
    timezone='Asia/Kolkata',
    model='ineichen'  # or 'haurwitz'
)

# Calculate clear-sky
times = df.index
cs = csm.calculate_clearsky(times, linke_turbidity=3.0)
print(cs.head())

# Add all clear-sky features to dataframe
df = csm.add_clearsky_features(df, ghi_col='GHI', linke_turbidity=3.0)
```

---

## 4. Engineer Features

```python
from scripts.utils import FeatureEngineer

# Initialize
fe = FeatureEngineer(df)

# Add features step by step
fe.add_temporal_features() \
  .add_lag_features(['GHI', 'temperature'], lags=[1, 6, 12, 24]) \
  .add_rolling_features(['GHI', 'humidity'], windows=[6, 12, 24]) \
  .handle_missing_values(method='forward_fill', limit=24)

# Get engineered dataframe
df_engineered = fe.get_dataframe()
print(f"Features added: {len(fe.get_feature_list())}")
print(df_engineered.head())
```

---

## 5. Evaluate Forecasts

```python
from scripts.utils import SolarForecastMetrics, HorizonEvaluator
import numpy as np

# Method 1: Single horizon evaluation
y_true = np.array([100, 200, 150, 300])
y_pred = np.array([110, 190, 160, 290])

metrics = SolarForecastMetrics.calculate_all_metrics(y_true, y_pred)
print(f"MAE: {metrics['mae']:.2f}")
print(f"RMSE: {metrics['rmse']:.2f}")
print(f"R²: {metrics['r2']:.4f}")

# Method 2: Horizon-wise evaluation
y_true_df = pd.DataFrame({
    'horizon_1': [...],
    'horizon_2': [...],
    'horizon_3': [...],
})
y_pred_df = y_true_df.copy()  # Replace with actual predictions

evaluator = HorizonEvaluator(y_true_df, y_pred_df)
results = evaluator.evaluate_by_horizon()
print(results)
```

---

## 6. Data Quality Checks

```python
from scripts.utils import DataLoader

loader = DataLoader()

# Check continuity
df = loader.load_raw_data()
gaps, gap_series = loader.validate_continuity(df, freq='H')
print(f"Found {gaps} time gaps")

# Get statistics
stats = loader.get_statistics(df)
print(stats)
```

---

## 7. Pipeline: Full Workflow

```python
from scripts.utils import DataLoader, SolarGeometry, ClearSkyModel, FeatureEngineer

# 1. Load
loader = DataLoader()
df = loader.load_processed_data()

# 2. Solar Geometry
sg = SolarGeometry(17.385, 78.487, 500, 'Asia/Kolkata')
df = sg.add_solar_features(df)

# 3. Clear-Sky
csm = ClearSkyModel(17.385, 78.487, 500, 'Asia/Kolkata')
df = csm.add_clearsky_features(df)

# 4. Feature Engineering
fe = FeatureEngineer(df)
fe.add_temporal_features() \
  .add_lag_features(['GHI', 'temperature'], lags=[1, 6, 12, 24]) \
  .add_rolling_features(['GHI', 'temperature'], windows=[6, 12, 24]) \
  .handle_missing_values()

df_features = fe.get_dataframe()

# 5. Save
df_features.to_csv('data/processed/features_engineered.csv')
print(f"Created {len(fe.get_feature_list())} features")
```

---

## 8. Configuration Access

```python
import yaml

# Load config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Access various settings
location = config['location']
print(f"Latitude: {location['latitude']}")

data_config = config['data']
print(f"Raw dir: {data_config['raw_dir']}")

models = config['models']
print(f"Available models: {[m['name'] for m in models if m['enabled']]}")

ranges = config['physical_ranges']
print(f"GHI range: {ranges['GHI']}")
```

---

## 9. Common Operations

### Get Daytime Data Only
```python
daytime = df[df['GHI'] > 10].copy()
```

### Get Specific Time Period
```python
start = pd.Timestamp('2024-01-01')
end = pd.Timestamp('2024-12-31')
df_2024 = df[(df.index >= start) & (df.index <= end)]
```

### Aggregate to Daily
```python
df_daily = df.resample('D').agg({
    'GHI': 'mean',
    'temperature': 'mean',
    'humidity': 'mean',
})
```

### Calculate Monthly Statistics
```python
df['month'] = df.index.month
monthly_stats = df.groupby('month')[['GHI', 'temperature']].agg(['mean', 'std', 'min', 'max'])
```

### Check Distribution
```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(df['GHI'], bins=50, edgecolor='black')
axes[0].set_title('GHI Distribution')

axes[1].hist(df['temperature'], bins=50, edgecolor='black')
axes[1].set_title('Temperature Distribution')

axes[2].hist(df['humidity'], bins=50, edgecolor='black')
axes[2].set_title('Humidity Distribution')

plt.tight_layout()
plt.show()
```

---

## 10. Tips & Best Practices

### Memory Efficiency
```python
# Use parquet for large datasets
df.to_parquet('data/processed/df.parquet')
df = pd.read_parquet('data/processed/df.parquet')
```

### Logging
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Processing started")
logger.warning("Check this value")
logger.error("Something went wrong")
```

### Parallel Processing
```python
from joblib import Parallel, delayed

# Process in parallel
results = Parallel(n_jobs=-1)(
    delayed(process_chunk)(chunk) 
    for chunk in chunks
)
```

### Data Validation
```python
# Check for outliers using IQR
Q1 = df['GHI'].quantile(0.25)
Q3 = df['GHI'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['GHI'] < Q1 - 1.5*IQR) | (df['GHI'] > Q3 + 1.5*IQR)]
```

---

## Troubleshooting

### Import Errors
```bash
# Make sure you're in project directory
cd Forecast_Irradiance_Temperature

# Add project to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or install in development mode
pip install -e .
```

### Missing Data Issues
```python
# Check where data is missing
missing = df.isna()
print(missing.sum())

# Forward fill for small gaps
df.fillna(method='ffill', limit=3, inplace=True)

# Interpolate for smoother results
df.interpolate(method='linear', inplace=True)
```

### Memory Issues with Large Datasets
```python
# Process in chunks
chunk_size = 10000
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    # Process chunk
    process(chunk)
```

---

## Project Structure Reference

```
scripts/
  ├── pipeline.py          → orchestrator (--stage full / all / individual)
  ├── train_models.py      → benchmark loop + prospective --select-best
  ├── analyze_models.py    → summaries + models/best_models.json
  ├── generate_report.py   → comparison report + generalization gap
  ├── model_diagnostics.py → models/feature_importance_matrix.csv / .json
  └── utils/               → data_loader, solar_geometry, clear_sky_model,
                             feature_engineering, metrics

api/                   → FastAPI service (main.py, services.py)
dashboard/app.py       → Streamlit dashboard
config/config.yaml     → all configuration
data/raw|processed/    → NASA POWER data + forecast_dataset.csv
models/                → *.joblib artifacts, training_results.json,
                         best_models.json, feature_importance*
```

---

For more information, see:
- [README.md](README.md) - Full documentation
- [SETUP_GUIDE.md](SETUP_GUIDE.md) - Environment setup
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Command cheat-sheet
- [config/config.yaml](config/config.yaml) - All configuration parameters
