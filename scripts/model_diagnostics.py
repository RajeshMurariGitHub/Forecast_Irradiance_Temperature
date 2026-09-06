"""Analyze trained models for feature importance and diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_DIR = Path("models")
RESULTS_FILE = MODEL_DIR / "training_results.json"
BEST_MODELS_FILE = MODEL_DIR / "best_models.json"
MATRIX_CSV = MODEL_DIR / "feature_importance_matrix.csv"
MATRIX_JSON = MODEL_DIR / "feature_importance.json"


def _unwrap_estimator(model):
    """Return the final predictor of a pipeline (or the model itself)."""
    if hasattr(model, "named_steps"):
        steps = model.named_steps
        return steps.get("model") or list(steps.values())[-1]
    return model


def _resolve_feature_names(model, estimator, fallback: list[str] | None) -> list[str] | None:
    """Feature names straight from the fitted artifact, not from run metadata.

    Order: the pipeline's pre-processing step (fit on the named DataFrame) ->
    CatBoost's ``feature_names_`` -> a bare estimator's ``feature_names_in_`` ->
    the supplied metadata fallback.
    """
    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            names = getattr(step, "feature_names_in_", None)
            if names is not None:
                return [str(name) for name in names]
    for attr in ("feature_names_", "feature_names_in_"):
        names = getattr(estimator, attr, None)
        if names is not None and len(names):
            return [str(name) for name in names]
    return list(fallback) if fallback else None


def _importance_values(estimator) -> np.ndarray | None:
    """Tree/boosting importances, or absolute linear coefficients."""
    if hasattr(estimator, "feature_importances_"):
        return np.asarray(estimator.feature_importances_, dtype=float)
    coef = getattr(estimator, "coef_", None)
    if coef is not None:
        coef = np.asarray(coef, dtype=float)
        if coef.ndim == 1:
            return np.abs(coef)
    return None


def extract_feature_importance(
    model_path: Path, fallback_features: list[str] | None = None
) -> dict[str, float]:
    """Feature -> importance for one saved model (empty for non-predictive baselines)."""
    try:
        model = joblib.load(model_path)
    except (OSError, ValueError) as exc:
        print(f"Error loading {model_path}: {exc}")
        return {}

    estimator = _unwrap_estimator(model)
    importances = _importance_values(estimator)
    if importances is None:  # e.g. the persistence baseline (a plain dict)
        return {}

    names = _resolve_feature_names(model, estimator, fallback_features)
    if names is None or len(names) != len(importances):
        print(
            f"Skipping {model_path.name}: {len(importances)} importances vs "
            f"{0 if names is None else len(names)} feature names"
        )
        return {}
    return {name: float(value) for name, value in zip(names, importances)}


def _metadata_features(results: list[dict]) -> dict[tuple[str, int, str], list[str]]:
    """(target, horizon_days, model) -> features recorded at train time (fallback only)."""
    lookup: dict[tuple[str, int, str], list[str]] = {}
    for result in results:
        horizon_days = int(result.get("horizon_days", result["horizon_hours"] // 24))
        lookup[(result["target"], horizon_days, result["model"])] = list(result["features"])
    return lookup


def _select_models(results: list[dict], only_best: bool) -> list[tuple[str, str, int, str, Path]]:
    """(config_key, target, horizon_days, model_name, model_path), validation split only."""
    selected = []
    if only_best and BEST_MODELS_FILE.exists():
        best = json.loads(BEST_MODELS_FILE.read_text(encoding="utf-8"))
        for target, horizons in best.get("best_per_horizon", {}).items():
            for horizon_days, pick in horizons.items():
                days = int(horizon_days)
                key = f"{target}_{days}d_{pick['model']}"
                selected.append((key, target, days, pick["model"], Path(pick["model_path"])))
        return selected

    for result in results:
        if result["split"] != "validation":
            continue
        days = int(result.get("horizon_days", result["horizon_hours"] // 24))
        key = f"{result['target']}_{days}d_{result['model']}"
        selected.append((key, result["target"], days, result["model"], Path(result["model_path"])))
    return selected


def load_and_analyze_models(only_best: bool = False) -> None:
    """Extract feature importance for trained models and persist a clean matrix."""
    if not RESULTS_FILE.exists():
        print(f"Results file not found: {RESULTS_FILE}")
        return

    results = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    fallback = _metadata_features(results)

    importances_by_config: dict[str, dict[str, float]] = {}
    for config_key, target, horizon_days, model_name, model_path in _select_models(
        results, only_best
    ):
        if not model_path.exists():
            continue
        meta_features = fallback.get((target, horizon_days, model_name))
        importance = extract_feature_importance(model_path, meta_features)
        if importance:
            importances_by_config[config_key] = importance

    if not importances_by_config:
        print("No feature importances could be extracted.")
        return

    print("=" * 100)
    print("TOP 10 FEATURES BY MODEL CONFIGURATION")
    print("=" * 100)
    for config_key in sorted(importances_by_config):
        top = sorted(importances_by_config[config_key].items(), key=lambda kv: kv[1], reverse=True)
        print(f"\n{config_key}:")
        for rank, (feature, value) in enumerate(top[:10], 1):
            print(f"  {rank:2}. {feature:30} -> {value:10.4f}")

        pd.DataFrame(top, columns=["Feature", "Importance"]).to_csv(
            MODEL_DIR / f"importance_{config_key}.csv", index=False
        )

    matrix = pd.DataFrame(importances_by_config).sort_index()
    matrix.index.name = "feature"
    matrix.to_csv(MATRIX_CSV)
    MATRIX_JSON.write_text(
        json.dumps(
            {config: dict(sorted(vals.items(), key=lambda kv: kv[1], reverse=True))
             for config, vals in importances_by_config.items()},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved feature importance matrix -> {MATRIX_CSV}")
    print(f"Saved feature importance JSON   -> {MATRIX_JSON}")


def main() -> None:
    """Run diagnostics."""
    parser = argparse.ArgumentParser(description="Feature importance diagnostics.")
    parser.add_argument(
        "--only-best",
        action="store_true",
        help="Analyse only the winning models from models/best_models.json.",
    )
    args = parser.parse_args()
    load_and_analyze_models(only_best=args.only_best)


if __name__ == "__main__":
    main()
