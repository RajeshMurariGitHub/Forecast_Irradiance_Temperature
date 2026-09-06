"""Train baseline forecasting models for irradiance and temperature."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get_split_bounds(split_name: str, config: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return an inclusive start and exclusive hourly split boundary."""
    split_cfg = config["data_split"][split_name]
    start = pd.Timestamp(split_cfg["start"])
    end_exclusive = pd.Timestamp(split_cfg["end"]).normalize() + pd.Timedelta(days=1)
    return start, end_exclusive


def get_split_frame(df: pd.DataFrame, split_name: str, config: dict) -> pd.DataFrame:
    start, end_exclusive = get_split_bounds(split_name, config)
    mask = (df.index >= start) & (df.index < end_exclusive)
    return df.loc[mask].copy()


def load_forecast_frame(config: dict, path: str | Path | None = None) -> pd.DataFrame:
    if path is None:
        data_cfg = config["data"]
        filename = data_cfg["forecast_dataset_filename"]
        path = Path(data_cfg["processed_dir"]) / filename
    path = Path(path)

    if path.exists():
        selected = path
    else:
        csv_path = path.with_suffix(".csv")
        parquet_path = path.with_suffix(".parquet")
        if csv_path.exists():
            selected = csv_path
        elif parquet_path.exists():
            selected = parquet_path
        else:
            raise FileNotFoundError(
                f"Forecast dataset not found at {path} or {csv_path} or {parquet_path}"
            )

    if selected.suffix.lower() == ".parquet":
        df = pd.read_parquet(selected)
    else:
        df = pd.read_csv(selected, index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def prepare_horizon_dataset(
    df: pd.DataFrame,
    target: str,
    horizon_days: int,
    origin_start: pd.Timestamp | None = None,
    origin_end_exclusive: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Build direct-horizon samples available at each forecast origin.

    ``origin_end_exclusive`` limits both forecast origins and their outcomes to
    the requested split. This prevents a phase from learning or scoring targets
    that belong to its successor.
    """
    horizon_hours = horizon_days * 24
    work = df.copy()
    future_target = f"{target}_future_{horizon_hours}h"
    work[future_target] = work[target].shift(-horizon_hours)
    if origin_start is not None:
        work = work.loc[work.index >= origin_start]
    if origin_end_exclusive is not None:
        latest_origin = origin_end_exclusive - pd.Timedelta(hours=horizon_hours)
        work = work.loc[work.index < latest_origin]
    work = work.dropna(subset=[future_target])

    exclude = {target, future_target}
    feature_cols = [col for col in work.columns if col not in exclude and col != "timestamp"]
    X = work[feature_cols].copy()
    y = work[future_target].astype(float)
    return X, y, feature_cols


def model_artifact_path(
    model_dir: Path, target: str, horizon_days: int, model_name: str, split_name: str
) -> Path:
    """Use day-based horizon names everywhere model artifacts are addressed."""
    return model_dir / f"{target}_{horizon_days}d_{model_name}_{split_name}.joblib"


def build_model(model_name: str):
    model_name = model_name.lower()
    if model_name == "ridge":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0, random_state=42)),
            ]
        )
    if model_name == "random_forest":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=50,
                        max_depth=15,
                        random_state=42,
                        n_jobs=1,
                        min_samples_leaf=5,
                    ),
                ),
            ]
        )
    if model_name == "extra_trees":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=50,
                        max_depth=15,
                        random_state=42,
                        n_jobs=1,
                        min_samples_leaf=5,
                    ),
                ),
            ]
        )
    if model_name == "gradient_boost":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=100,
                        max_depth=4,
                        learning_rate=0.1,
                        subsample=0.8,
                        random_state=42,
                    ),
                ),
            ]
        )
    if model_name == "xgboost":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=200,
                        max_depth=6,
                        learning_rate=0.1,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42,
                        n_jobs=1,
                        tree_method="hist",
                    ),
                ),
            ]
        )
    if model_name == "lightgbm":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    LGBMRegressor(
                        n_estimators=200,
                        max_depth=-1,
                        num_leaves=31,
                        learning_rate=0.1,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42,
                        n_jobs=1,
                        verbose=-1,
                    ),
                ),
            ]
        )
    if model_name == "catboost":
        from catboost import CatBoostRegressor

        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    CatBoostRegressor(
                        iterations=400,
                        depth=6,
                        learning_rate=0.1,
                        random_state=42,
                        thread_count=1,
                        verbose=False,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unsupported model: {model_name}")


def evaluate_model(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    # R² is undefined for a single observation (zero variance in y_true).
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else None
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2,
    }


def train_split_model(
    df: pd.DataFrame,
    target: str,
    horizon_days: int,
    model_name: str,
    split_name: str,
    config: dict,
    model_dir: Path,
) -> dict:
    """Benchmark one (target, horizon, model) on a split and persist the artifact.

    ``model_name == "persistence"`` is the naive baseline: the forecast for every
    horizon is the last value observed 24 h before the forecast origin (previous
    day, same hour). Nothing is fitted; the saved artifact records only the rule.
    """
    split_start, split_end = get_split_bounds(split_name, config)
    train_start, train_end = get_split_bounds("training", config)
    X, y, feature_cols = prepare_horizon_dataset(df, target, horizon_days, split_start, split_end)
    train_X, train_y, _ = prepare_horizon_dataset(df, target, horizon_days, train_start, train_end)

    common_features = [col for col in feature_cols if col in train_X.columns and col in X.columns]
    X = X[common_features].copy()
    train_X = train_X[common_features].copy()

    # Drop rows with missing target values and then rely on the model imputer for feature NaNs.
    X = X.loc[y.notna()].copy()
    y = y.loc[X.index]
    train_X = train_X.loc[train_y.notna()].copy()
    train_y = train_y.loc[train_X.index]

    persistence_lag_hours = 24
    if model_name == "persistence":
        # Naive baseline: carry the t-24 h observation forward to every horizon.
        model = {
            "type": "persistence",
            "target": target,
            "lag_hours": persistence_lag_hours,
        }
        preds = df[target].shift(persistence_lag_hours).reindex(X.index).to_numpy()
        keep = ~np.isnan(preds)
        X, y, preds = X.loc[keep], y.loc[keep], preds[keep]
        fitted_features = [f"{target}_persist_t-{persistence_lag_hours}h"]
    else:
        # To avoid train leakage into validation/test, we only fit on the training period.
        model = build_model(model_name)
        model.fit(train_X, train_y)
        preds = model.predict(X)
        fitted_features = common_features

    metrics = evaluate_model(y, preds)

    model_path = model_artifact_path(model_dir, target, horizon_days, model_name, split_name)
    joblib.dump(model, model_path)

    return {
        "target": target,
        "split": split_name,
        "horizon_days": horizon_days,
        "horizon_hours": horizon_days * 24,
        "model": model_name,
        "rows": int(len(X)),
        # Persist the exact columns the model was fitted on so the serving layer
        # rebuilds X with the same feature contract.
        "features": fitted_features,
        "metrics": metrics,
        "model_path": str(model_path),
    }


def train_prospective_model(
    df: pd.DataFrame,
    target: str,
    horizon_days: int,
    model_name: str,
    config: dict,
    model_dir: Path,
) -> dict:
    """Lock a model at the validation cutoff and score one future origin.

    This intentionally produces one prediction per direct horizon. A full May
    trajectory would require future weather inputs or a recursive model, neither
    of which is represented by the current feature contract.
    """
    train_start, _ = get_split_bounds("training", config)
    _, validation_end = get_split_bounds("validation", config)
    _, prospective_end = get_split_bounds("prospective", config)
    origin = validation_end - pd.Timedelta(hours=1)
    horizon_hours = horizon_days * 24
    target_time = origin + pd.Timedelta(hours=horizon_hours)
    if target_time >= prospective_end:
        raise ValueError(
            f"The {horizon_days}d horizon from {origin} is outside the prospective split."
        )

    train_X, train_y, feature_cols = prepare_horizon_dataset(
        df, target, horizon_days, train_start, validation_end
    )
    X = df.loc[[origin], feature_cols].copy()
    y = df.loc[[target_time], target].astype(float)

    if model_name == "persistence":
        model = {"type": "persistence", "target": target, "lag_hours": 24}
        preds = df[target].reindex([origin - pd.Timedelta(hours=24)]).to_numpy()
        feature_cols = [f"{target}_persist_t-24h"]
    else:
        model = build_model(model_name)
        model.fit(train_X, train_y)
        preds = model.predict(X)
    metrics = evaluate_model(y, preds)
    model_path = model_artifact_path(model_dir, target, horizon_days, model_name, "prospective")
    joblib.dump(model, model_path)

    return {
        "target": target,
        "split": "prospective",
        "evaluation_mode": "fixed_origin",
        "forecast_origin": origin.isoformat(),
        "target_timestamp": target_time.isoformat(),
        "horizon_days": horizon_days,
        "horizon_hours": horizon_hours,
        "model": model_name,
        "rows": 1,
        "features": feature_cols,
        "metrics": metrics,
        "model_path": str(model_path),
    }


def parse_horizons(value: str | None) -> list[int]:
    """Parse comma-separated forecast horizons expressed in days (e.g. 1,2,3 or 1d,2d,3d)."""
    if value is None:
        return [1, 2, 3, 5, 7, 10, 14, 30]
    days = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if token.endswith("d"):
            days.append(int(token[:-1]))
        elif token.endswith("h"):
            days.append(max(1, int(token[:-1]) // 24))
        else:
            days.append(int(token))
    return days


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline forecast models.")
    parser.add_argument("--config", default="config/config.yaml", help="Project config path")
    parser.add_argument("--data", default=None, help="Forecast-ready dataset path (CSV/Parquet)")
    parser.add_argument("--targets", default="GHI,temperature", help="Comma-separated target names")
    parser.add_argument(
        "--horizons", default=None, help="Comma-separated horizons in days, e.g. 1,3,7 or 1d,3d,7d"
    )
    parser.add_argument(
        "--models",
        default="ridge,random_forest",
        help=(
            "Comma-separated model names (persistence, ridge, random_forest, "
            "extra_trees, gradient_boost, xgboost, lightgbm, catboost)"
        ),
    )
    parser.add_argument(
        "--split",
        default="validation",
        choices=["training", "testing", "validation", "prospective"],
        help="Split to evaluate",
    )
    parser.add_argument(
        "--select-best",
        action="store_true",
        help="For prospective evaluation, refit only the highest-validation-R2 model per target and horizon.",
    )
    return parser.parse_args()


def select_best_validation_models(
    results_path: Path, targets: list[str], horizons: list[int]
) -> list[tuple[str, int, str]]:
    """Select one model per target/horizon using the persisted validation results."""
    if not results_path.exists():
        raise FileNotFoundError(
            f"Validation results are required for --select-best: {results_path}"
        )
    with open(results_path, "r", encoding="utf-8") as fh:
        results = json.load(fh)

    best: dict[tuple[str, int], dict] = {}
    requested = {(target, horizon) for target in targets for horizon in horizons}
    for result in results:
        key = (result["target"], result["horizon_days"])
        if result["split"] != "validation" or key not in requested:
            continue
        if key not in best or result["metrics"]["r2"] > best[key]["metrics"]["r2"]:
            best[key] = result

    missing = requested.difference(best)
    if missing:
        formatted = ", ".join(f"{target}/{horizon}d" for target, horizon in sorted(missing))
        raise ValueError(f"No validation result found for: {formatted}")
    return [
        (target, horizon, result["model"]) for (target, horizon), result in sorted(best.items())
    ]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    config = load_config(args.config)
    df = load_forecast_frame(config, args.data)

    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    horizons = parse_horizons(args.horizons)

    output = model_dir / "training_results.json"
    results = []
    if args.split == "prospective" and args.select_best:
        model_runs = select_best_validation_models(output, targets, horizons)
    else:
        model_runs = [
            (target, horizon_days, model_name)
            for target in targets
            for horizon_days in horizons
            for model_name in models
        ]

    for target, horizon_days, model_name in model_runs:
        if target not in df.columns:
            logger.warning("Target %s not found in dataset; skipping.", target)
            continue
        if args.split == "prospective":
            result = train_prospective_model(
                df=df,
                target=target,
                horizon_days=horizon_days,
                model_name=model_name,
                config=config,
                model_dir=model_dir,
            )
        else:
            result = train_split_model(
                df=df,
                target=target,
                horizon_days=horizon_days,
                model_name=model_name,
                split_name=args.split,
                config=config,
                model_dir=model_dir,
            )
        results.append(result)

    # Replace matching evaluations so dashboard metrics cannot remain stale after retraining.
    all_results = results
    if output.exists():
        with open(output, "r", encoding="utf-8") as fh:
            existing = json.load(fh)
        result_key = lambda item: (
            item["target"],
            item["split"],
            item["horizon_hours"],
            item["model"],
        )
        updated = {result_key(item): item for item in existing}
        updated.update({result_key(item): item for item in results})
        all_results = list(updated.values())
    with open(output, "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2)
    logger.info("Saved model training results to %s (%d total results)", output, len(all_results))


if __name__ == "__main__":
    main()
