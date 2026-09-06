# Quick Reference Guide

## Essential Commands

### Verify Setup
```bash
# Check Python environment
python --version
python -c "import sys; print(sys.executable)"

# Verify all dependencies
python verify_dependencies.py

# Compile-check all scripts
python -m py_compile scripts/*.py scripts/utils/*.py
```

### Data Pipeline

| Task | Command |
|------|---------|
| **Collect Data** | `python scripts/pipeline.py --stage collect` |
| **Clean Data** | `python scripts/pipeline.py --stage clean` |
| **Engineer Features** | `python scripts/pipeline.py --stage features` |
| **Full Pipeline** | `python scripts/pipeline.py --stage all` |

### Model Training

| Task | Command |
|------|---------|
| **Benchmark all models** | `python scripts/train_models.py --targets GHI,temperature --models ridge,random_forest,extra_trees,gradient_boost,xgboost,lightgbm,catboost --split validation` |
| **Train one model** | `python scripts/train_models.py --models catboost --split testing` |
| **Specific horizons** | `python scripts/train_models.py --models lightgbm --horizons 1,7,30 --split validation` |
| **Prospective (best per horizon)** | `python scripts/train_models.py --split prospective --select-best` |

### Analysis & Reporting

| Task | Command |
|------|---------|
| **Model Comparison** | `python scripts/analyze_models.py` |
| **Feature Importance** | `python scripts/model_diagnostics.py` |
| **Generalization Gap** | `python scripts/generate_report.py` |

### Code Quality

| Task | Command |
|------|---------|
| **Format Code** | `black scripts/` |
| **Sort Imports** | `isort scripts/` |
| **Type Check** | `mypy scripts/` |
| **Lint Code** | `pylint scripts/` |
| **Run Tests** | `pytest tests/ -v` |
| **All Checks** | `black scripts/ && isort scripts/ && mypy scripts/ && pylint scripts/` |

## Directory Structure Quick Lookup

```
scripts/              → Execution entrypoints
├── pipeline.py      → Data collection & processing
├── train_models.py  → Model training
├── analyze_models.py → Performance comparison
└── utils/           → Utility modules

config/              → Configuration
└── config.yaml      → Project settings (locations, horizons, etc.)

data/                → Data artifacts
├── raw/             → NASA POWER API downloads
└── processed/       → Engineered features

models/              → Trained models & results
├── *.pkl            → Serialized model files
└── training_results.json → Consolidated metrics
```

## Configuration Quick Lookup

| File | Purpose | Edit When |
|------|---------|-----------|
| `config/config.yaml` | Location, horizons, paths | Changing location or time ranges |
| `.vscode/settings.json` | VS Code settings | Using different Python interpreter |
| `pyrightconfig.json` | Type checking | Changing Python version or paths |
| `pyproject.toml` | Project metadata | Adding/removing dependencies |
| `.pylintrc` | Code style rules | Adjusting linting tolerance |

## Common Development Tasks

### Adding a New Dependency

1. Add to `requirements.txt` (pinned version)
2. Add to `pyproject.toml` (flexible version constraint)
3. Install: `pip install -r requirements.txt`
4. Verify: `python verify_dependencies.py`

### Creating a New Script

1. Place in `scripts/` directory
2. Add imports: `from scripts.utils.feature_engineering import engineer_features`
3. Import `config/config.yaml` if needed: `yaml.safe_load(open('config/config.yaml'))`
4. Run quality checks: `black script.py && isort script.py && mypy script.py`

### Modifying Model Configuration

Edit `scripts/train_models.py`:
- Line ~50: `MODELS` dictionary (adjust hyperparameters)
- Line ~60: `HORIZONS` list (forecast hours)
- Line ~70: `TARGET_COLS` list (GHI, temperature, etc.)

Example:
```python
MODELS = {
    'ridge': Ridge(alpha=1.0),  # Adjust alpha
    'random_forest': RandomForestRegressor(n_estimators=100),  # Increase estimators
}
HORIZONS = [1, 3, 6, 12, 24]  # Custom forecast horizons
```

### Debugging a Training Run

```bash
# Run with debug output (edit train_models.py, add prints)
python scripts/train_models.py --split validation --models ridge | head -100

# Check intermediate data
python -c "
import json
with open('models/training_results.json', 'r') as f:
    results = json.load(f)
    print(f'Training completed: {len(results)} models')
    for r in results[:3]:
        print(f'  {r[\"model\"]:20} {r[\"target\"]:15} R²={r[\"r2_val\"]:6.4f}')
"
```

## Performance Optimization

| Issue | Solution |
|-------|----------|
| **Training slow** | Use `--split training` only (skip validation/testing) |
| | Reduce RandomForest `n_estimators` (e.g., 25 instead of 50) |
| | Filter `HORIZONS` to fewer values |
| **Memory issues** | Process data in chunks in `feature_engineering.py` |
| **Import warnings in VS Code** | Restart IDE after config changes |
| **Models not found** | Check `models/` directory exists, run full pipeline first |

## Typical Workflow

### Phase 1: Initial Setup (One-time)
```bash
cd C:\Users\rajes\Projects\Forecast_Irradiance_Temperature
python verify_dependencies.py  # Confirm environment
# Restart VS Code to apply configuration
```

### Phase 2: Data Preparation
```bash
python scripts/pipeline.py --stage all  # Collect & process NASA POWER data
# Output: data/processed/forecast_dataset.csv (57,696 rows × 69 features)
```

### Phase 3: Model Training
```bash
# Training split (baseline performance)
python scripts/train_models.py --split training

# Validation split (generalization)
python scripts/train_models.py --split validation

# Testing split (final evaluation)
python scripts/train_models.py --split testing
```

### Phase 4: Analysis & Reporting
```bash
python scripts/analyze_models.py      # Model comparison
python scripts/model_diagnostics.py   # Feature importance
python scripts/generate_report.py     # Generalization gap
```

### Phase 5: Iteration
```bash
# Improve model performance
# - Edit hyperparameters in train_models.py
# - Add new features in utils/feature_engineering.py
# - Try new models (e.g., XGBoost, LSTM)
# - Repeat Phase 3-4
```

## File Locations Reference

| Data | Location |
|------|----------|
| **Raw NASA Data** | `data/raw/hyderabad_*.csv` |
| **Cleaned Data** | `data/processed/hyderabad_cleaned.csv` |
| **Features** | `data/processed/forecast_dataset.csv` |
| **Models** | `models/ridge_*.pkl`, `models/random_forest_*.pkl` |
| **Results** | `models/training_results.json` |
| **Feature Importance** | `models/importance_*.csv` (32 files) |
| **Reports** | Various CSV files in `models/` (metric pivots, summaries) |

## Environment Details

- **Python**: 3.13.9 (Anaconda)
- **Location**: `C:\Users\rajes\anaconda3\python.exe`
- **Key Packages**: pandas 2.3.3, numpy 2.4.6, scikit-learn 1.8.0, pvlib 0.15.2

## Success Indicators

✓ **Setup Complete When**:
- `python verify_dependencies.py` shows all 10 packages available
- `python scripts/pipeline.py --stage collect` runs without errors
- VS Code shows no import warnings after restart
- `python scripts/train_models.py --split training` trains at least 2 models

✓ **Pipeline Working When**:
- `data/processed/forecast_dataset.csv` exists and has 57,696+ rows
- `models/training_results.json` contains model results
- `models/importance_*.csv` files created (one per model)

✓ **Ready for Analysis When**:
- Can run `analyze_models.py`, `model_diagnostics.py`, `generate_report.py`
- Model outputs display R², MAE, RMSE metrics
- Feature importance rankings computed for all models
