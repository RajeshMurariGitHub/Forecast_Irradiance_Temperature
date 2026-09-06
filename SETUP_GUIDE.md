# Project Configuration & Setup Guide

## Overview
This document outlines the Python environment configuration for the Solar Irradiance and Temperature Forecasting project.

## Environment Status ✓

### Python Version
- **Executable**: `C:\Users\rajes\anaconda3\python.exe`
- **Version**: Python 3.13.9 (Anaconda)
- **Verified**: ✓ All 10 critical dependencies available

### Installed Dependencies

| Package | Version | Status |
|---------|---------|--------|
| pandas | 2.3.3 | ✓ |
| numpy | 2.4.6 | ✓ |
| scikit-learn | 1.8.0 | ✓ |
| scipy | 1.18.0 | ✓ |
| pvlib | 0.15.2 | ✓ |
| pyyaml | 6.0.3 | ✓ |
| joblib | 1.5.3 | ✓ |
| requests | 2.32.5 | ✓ |

## Configuration Files

### 1. `.vscode/settings.json` 
**Purpose**: VS Code workspace settings for Python development
**Key Configurations**:
- Python interpreter: `C:\Users\rajes\anaconda3\python.exe`
- Pylance analysis enabled with type checking
- Import resolution via `extraPaths` pointing to `scripts/` and `scripts/utils/`
- Code formatting with Black (100 char lines)
- Import sorting with isort

**Why This Matters**: 
- Ensures VS Code uses the correct Python interpreter (your Anaconda environment)
- Enables intelligent code completion and type hints
- Resolves import warnings for project modules

### 2. `pyrightconfig.json`
**Purpose**: Pyright/Pylance type checking configuration
**Key Settings**:
- Type checking mode: `basic` (catches common errors without strict enforcement)
- Python version: `3.13`
- Include patterns: `scripts/**/*.py`
- Exclude patterns: Tests, caches, virtual environments
- Extra paths: Include project's scripts directory

**Why This Matters**:
- Provides Pylance with explicit instructions for module discovery
- Enables type inference without requiring type annotations everywhere
- Supports modern Python features (3.13)

### 3. `pyproject.toml`
**Purpose**: Modern Python project metadata and tool configuration (PEP 517/518 standard)
**Contains**:
- Project metadata (name, version, description, dependencies)
- Build system configuration
- Tool configurations: Black, isort, mypy, pytest, pylint
- Optional dependency groups: dev, ml, dl, viz

**Why This Matters**:
- Single source of truth for project configuration
- Enables installation via `pip install -e .` for development
- Standardized tool configuration (instead of scattered config files)
- Supports optional dependency groups for different use cases

### 4. `.pylintrc`
**Purpose**: Pylint linting configuration
**Key Settings**:
- Line length: 100 characters
- Disabled warnings for common patterns in data science (R0913, etc.)
- Extension allowlist for numpy, pandas, sklearn
- Design rules for max methods, args, lines, etc.

**Why This Matters**:
- Provides consistent code quality checks
- Customized for data science workflows (less strict on design rules)
- Prevents false positives for scientific computing patterns

### 5. `requirements.txt`
**Purpose**: Pinned dependency versions for reproducibility
**Contains**: 40+ packages including:
- Core: pandas, numpy, scipy, scikit-learn
- Solar: pvlib
- ML additional: xgboost, lightgbm, catboost, statsmodels
- Deep Learning: tensorflow, torch, keras
- Visualization: matplotlib, seaborn, plotly

**Why This Matters**:
- Ensures consistent environment across machines
- Enables reproducible results
- Documents exact versions used for model training

## Project Structure

```
Forecast_Irradiance_Temperature/
├── .vscode/
│   └── settings.json              # VS Code workspace configuration
├── config/
│   └── config.yaml                # Project configuration (locations, horizons, paths)
├── data/
│   ├── raw/                       # NASA POWER API downloads
│   ├── processed/                 # Cleaned and engineered features
│   └── predictions/               # Model predictions
├── models/
│   ├── ridge_*.pkl                # Ridge regression models (16)
│   ├── random_forest_*.pkl        # Random Forest models (16)
│   └── training_results.json      # Consolidated training results
├── scripts/
│   ├── __init__.py               # Package marker
│   ├── pipeline.py               # Main ETL orchestration
│   ├── train_models.py           # Model training entrypoint
│   ├── analyze_models.py         # Model comparison & visualization
│   ├── model_diagnostics.py      # Feature importance extraction
│   ├── generate_report.py        # Generalization gap analysis
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── feature_engineering.py # Temporal & lag features
│   │   ├── solar_geometry.py      # Solar position calculations
│   │   └── clear_sky_model.py     # Ineichen clear-sky model
│   └── tests/
│       └── test_*.py              # Unit tests
├── pyrightconfig.json             # Pyright type checking config
├── pyproject.toml                 # Modern Python project config
├── .pylintrc                      # Pylint configuration
├── requirements.txt               # Pinned dependencies
└── README.md                      # Project documentation
```

## Resolving Import Warnings in VS Code

### Issue
Import warnings appear in VS Code (e.g., "pandas could not be resolved") even though:
- ✓ All packages are installed
- ✓ Python runtime imports work correctly
- ✓ Scripts execute without errors

### Root Cause
VS Code's Pylance language server wasn't configured to use your Anaconda Python environment.

### Solution (Implemented)

1. **Updated `.vscode/settings.json`**:
   - Set `python.defaultInterpreterPath` to your Anaconda executable
   - Added `python.analysis.extraPaths` for local module discovery
   - Configured Pylance for intelligent analysis

2. **Created `pyrightconfig.json`**:
   - Tells Pyright where to find modules
   - Sets appropriate Python version target
   - Excludes cache/test directories from analysis

3. **Created `pyproject.toml`**:
   - Standard project metadata
   - Tool configurations for consistency

4. **Created `.pylintrc`**:
   - Custom linting rules for data science code
   - Module allowlists for scientific packages

### Next Steps

1. **Restart VS Code** to reload configuration:
   - Close VS Code completely
   - Reopen the project
   - Pylance will reinitialize with new settings

2. **Verify Resolution**:
   - Open any Python file
   - Hover over imports (e.g., `import pandas`)
   - No red squiggly lines should appear
   - IntelliSense should show autocomplete suggestions

### If Warnings Persist

1. **Check Python Interpreter Selection**:
   - Press `Ctrl+Shift+P` → "Python: Select Interpreter"
   - Ensure it shows: `./anaconda/python.exe`

2. **Clear Pylance Cache**:
   - Delete: `C:\Users\rajes\AppData\Roaming\Code\extensions\ms-python.vscode-pylance-*\.vscode\`
   - Restart VS Code

3. **Force Pylance Restart**:
   - Open Command Palette: `Ctrl+Shift+P`
   - Type: "Python: Restart Language Server"
   - Press Enter

## Running the Pipeline

### 1. Data Collection & Preprocessing
```bash
python scripts/pipeline.py --stage all
```

### 2. Train Baseline Models
```bash
python scripts/train_models.py \
  --targets GHI,temperature \
  --models ridge,random_forest \
  --split training validation testing
```

### 3. Analyze Model Performance
```bash
python scripts/analyze_models.py
python scripts/model_diagnostics.py
python scripts/generate_report.py
```

## Development Workflow

### Code Formatting
```bash
black scripts/
```

### Import Sorting
```bash
isort scripts/
```

### Type Checking
```bash
mypy scripts/
```

### Linting
```bash
pylint scripts/
```

### Testing
```bash
pytest tests/ -v
```

### Full Quality Check
```bash
black scripts/ && isort scripts/ && mypy scripts/ && pylint scripts/ && pytest tests/
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'X'"
**Solution**: Verify Python interpreter:
```bash
python -c "import sys; print(sys.executable)"
```
Should output: `C:\Users\rajes\anaconda3\python.exe`

### Issue: Imports work in terminal but not in VS Code
**Solution**: 
1. Restart VS Code
2. Select correct interpreter via `Ctrl+Shift+P` → "Python: Select Interpreter"
3. Clear Pylance cache

### Issue: Model training hangs or is slow
**Solution**:
- RandomForest with 50 estimators × 8 horizons × 2 targets takes ~12 minutes
- Use `--split` to train specific splits only
- Reduce estimators for faster iteration: Edit `train_models.py` line with `n_estimators=`

## References

- **Pyright Documentation**: https://github.com/microsoft/pyright
- **Pylance User Guide**: https://github.com/microsoft/pylance-release
- **PEP 517/518**: https://peps.python.org/pep-0517/
- **PEP 621**: https://peps.python.org/pep-0621/ (pyproject.toml)

## Support

For configuration issues:
1. Check `verify_dependencies.py` output
2. Review this guide's troubleshooting section
3. Examine `.vscode/settings.json` and `pyrightconfig.json` for syntax errors
4. Check VS Code output pane for Pylance diagnostics
