"""Data and model access helpers shared by the FastAPI service."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"
MODELS_DIR = BASE_DIR / "models"
RESULTS_PATH = MODELS_DIR / "training_results.json"
BEST_MODEL_METADATA_PATH = MODELS_DIR / "best_models.json"

# Served when the evaluation report is missing or its chosen artifact is absent.
FALLBACK_MODEL = "random_forest"

# Production forecast defaults: the winning CatBoost model locked at the
# validation cutoff, projected across May 2026.
PRODUCTION_SPLIT = "prospective"
PRODUCTION_MODEL = "catboost"
PRODUCTION_PERIOD = ("2026-05-01", "2026-05-31")


def get_split_bounds(split_name: str, config: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    split_cfg = config["data_split"][split_name]
    start = pd.Timestamp(split_cfg["start"])
    end_exclusive = pd.Timestamp(split_cfg["end"]).normalize() + pd.Timedelta(days=1)
    return start, end_exclusive


def model_artifact_path(target: str, horizon_hours: int, model_name: str, split_name: str) -> Path:
    return MODELS_DIR / f"{target}_{horizon_hours // 24}d_{model_name}_{split_name}.joblib"


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@lru_cache(maxsize=1)
def load_results() -> list[dict]:
    if not RESULTS_PATH.exists():
        return []
    with open(RESULTS_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def load_forecast_frame() -> pd.DataFrame:
    config = load_config()
    data_cfg = config["data"]
    path = Path(data_cfg["processed_dir"]) / data_cfg["forecast_dataset_filename"]
    path = BASE_DIR / path if not path.is_absolute() else path
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def get_split_frame(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    config = load_config()
    start, end_exclusive = get_split_bounds(split_name, config)
    mask = (df.index >= start) & (df.index < end_exclusive)
    return df.loc[mask].copy()


def prepare_horizon_dataset(
    df: pd.DataFrame, target: str, horizon_hours: int, feature_cols: list[str]
):
    work = df.copy()
    future_target = f"{target}_future_{horizon_hours}h"
    work[future_target] = work[target].shift(-horizon_hours)
    work = work.dropna(subset=[future_target])
    common_cols = [col for col in feature_cols if col in work.columns]
    X = work[common_cols].copy()
    y = work[future_target].astype(float)
    return X, y


@lru_cache(maxsize=64)
def load_model(target: str, horizon_hours: int, model_name: str, split_name: str):
    if horizon_hours % 24:
        raise ValueError("Only whole-day forecast horizons are supported.")
    model_path = model_artifact_path(target, horizon_hours, model_name, split_name)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)


@lru_cache(maxsize=1)
def load_best_model_metadata() -> dict:
    """Selection written by scripts/analyze_models.py (``models/best_models.json``)."""
    if not BEST_MODEL_METADATA_PATH.exists():
        return {}
    with open(BEST_MODEL_METADATA_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def best_model_name(target: str, horizon_hours: int) -> str | None:
    """Model the evaluation report picked (lowest RMSE) for this target/horizon."""
    metadata = load_best_model_metadata()
    horizon_days = str(horizon_hours // 24)
    try:
        return metadata["best_per_horizon"][target][horizon_days]["model"]
    except (KeyError, TypeError):
        return None


def resolve_forecast_model(
    target: str, horizon_hours: int, split_name: str, model_name: str | None = None
):
    """Resolve the model to serve for a forecast request.

    ``model_name=None`` -> use the ``best_model`` from the evaluation report
    (``models/best_models.json``). Any missing artifact -- an explicit request,
    the report's pick, or the report itself -- falls back to the Random Forest
    model for the same split. Non-predicting baselines (e.g. persistence) are
    skipped during automatic resolution.

    Returns ``(effective_model_name, loaded_model)``.
    """
    candidates: list[str] = []
    if model_name:
        candidates.append(model_name)
    else:
        chosen = best_model_name(target, horizon_hours)
        if chosen and chosen != "persistence":
            candidates.append(chosen)
    if FALLBACK_MODEL not in candidates:
        candidates.append(FALLBACK_MODEL)

    tried: list[str] = []
    for name in candidates:
        path = model_artifact_path(target, horizon_hours, name, split_name)
        tried.append(str(path))
        if not path.exists():
            continue
        model = load_model(target, horizon_hours, name, split_name)
        if hasattr(model, "predict"):
            return name, model

    raise FileNotFoundError(
        f"No usable model artifact for {target}/{horizon_hours}h ({split_name}); tried: "
        + ", ".join(tried)
    )


def find_result(target: str, horizon_hours: int, model_name: str, split_name: str) -> dict | None:
    for result in load_results():
        if (
            result["target"] == target
            and result["horizon_hours"] == horizon_hours
            and result["model"] == model_name
            and result["split"] == split_name
        ):
            return result
    return None


def load_feature_importance(
    target: str, horizon_hours: int, model_name: str
) -> pd.DataFrame | None:
    if horizon_hours % 24:
        return None
    csv_path = MODELS_DIR / f"importance_{target}_{horizon_hours // 24}d_{model_name}.csv"
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


def _prospective_feature_cols(target: str, horizon_hours: int) -> list[str]:
    """Feature contract the prospective models for this target/horizon were fit on."""
    for result in load_results():
        if (
            result["target"] == target
            and result["horizon_hours"] == horizon_hours
            and result["split"] == PRODUCTION_SPLIT
        ):
            return list(result["features"])
    raise FileNotFoundError(
        f"No {PRODUCTION_SPLIT} training record for {target}/{horizon_hours}h."
    )


def ingest_history_window(
    target: str,
    horizon_hours: int,
    feature_cols: list[str],
    period_start: str = PRODUCTION_PERIOD[0],
    period_end: str = PRODUCTION_PERIOD[1],
) -> "tuple[pd.DataFrame, pd.Series]":
    """Latest historical sliding window feeding a direct-horizon forecast.

    Every hour ``t`` in ``[period_start, period_end]`` is predicted from the
    observed feature row at ``t - horizon_hours``. Returns ``(X, actual)`` both
    indexed by the target hour ``t``; ``actual`` is NaN where the outcome has not
    been observed yet.
    """
    frame = load_forecast_frame()
    target_index = pd.date_range(start=f"{period_start} 00:00", end=f"{period_end} 23:00", freq="h")
    origin_index = target_index - pd.Timedelta(hours=horizon_hours)

    known = origin_index.isin(frame.index)
    target_index, origin_index = target_index[known], origin_index[known]
    if len(target_index) == 0:
        raise ValueError(
            f"No history {horizon_hours}h before {period_start}..{period_end} in the dataset."
        )

    cols = [col for col in feature_cols if col in frame.columns]
    features = frame.loc[origin_index, cols].copy()
    features.index = target_index

    if target in frame.columns:
        actual = pd.Series(frame[target]).reindex(target_index).astype(float)
    else:
        actual = pd.Series(float("nan"), index=target_index, name=target)
    return features, actual


def production_forecast(
    target: str,
    horizon_hours: int,
    model_name: str | None = PRODUCTION_MODEL,
    period_start: str = PRODUCTION_PERIOD[0],
    period_end: str = PRODUCTION_PERIOD[1],
    limit: int | None = None,
) -> dict:
    """Hourly forward forecast for May 2026 from the winning prospective model.

    Defaults to the CatBoost prospective artifact; a missing artifact falls back
    through :func:`resolve_forecast_model` (report best, then Random Forest).
    """
    effective_model, model = resolve_forecast_model(
        target, horizon_hours, PRODUCTION_SPLIT, model_name
    )
    feature_cols = _prospective_feature_cols(target, horizon_hours)
    features, actual = ingest_history_window(
        target, horizon_hours, feature_cols, period_start, period_end
    )
    predicted = pd.Series(model.predict(features), index=features.index)

    observed = actual.dropna()
    metrics: dict | None = None
    if len(observed):
        error = predicted.loc[observed.index] - observed
        metrics = {
            "mae": round(float(error.abs().mean()), 4),
            "rmse": round(float((error**2).mean() ** 0.5), 4),
            "n_observed": int(len(observed)),
        }

    timestamps = [pd.Timestamp(ts).isoformat() for ts in predicted.index][:limit]
    predicted_out = [round(float(v), 4) for v in predicted][:limit]
    actual_out = [None if pd.isna(v) else round(float(v), 4) for v in actual][:limit]

    return {
        "target": target,
        "horizon_hours": horizon_hours,
        "model": effective_model,
        "requested_model": model_name,
        "period": {"start": period_start, "end": period_end},
        "origin_offset_hours": horizon_hours,
        "timestamps": timestamps,
        "predicted": predicted_out,
        "actual": actual_out,
        "metrics": metrics,
    }


def available_combinations() -> list[dict]:
    combos = []
    for result in load_results():
        combos.append(
            {
                "target": result["target"],
                "horizon_hours": result["horizon_hours"],
                "model": result["model"],
                "split": result["split"],
            }
        )
    return combos
