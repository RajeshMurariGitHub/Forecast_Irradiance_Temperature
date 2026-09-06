"""FastAPI service exposing model metrics, feature importance, and forecasts."""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from api import services

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Solar Irradiance & Temperature Forecast API",
    description="Serves trained model metrics, feature importance, and forecasts for the Hyderabad solar project.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/config")
def get_config() -> dict:
    config = services.load_config()
    horizon_days = config.get("forecast", {}).get("horizons", [])
    return {
        "project": config.get("project", {}),
        "location": config.get("location", {}),
        "targets": config.get("forecast", {}).get("targets", []),
        "horizons": [int(days) * 24 for days in horizon_days],
        "splits": list(config.get("data_split", {}).keys()),
    }


@app.get("/api/combinations")
def get_combinations() -> list[dict]:
    """List every trained (target, horizon, model, split) combination available."""
    return services.available_combinations()


@app.get("/api/results")
def get_results(
    target: str | None = None,
    split: str | None = None,
    model: str | None = None,
    horizon_hours: int | None = None,
) -> list[dict]:
    results = services.load_results()
    filtered = []
    for r in results:
        if target and r["target"] != target:
            continue
        if split and r["split"] != split:
            continue
        if model and r["model"] != model:
            continue
        if horizon_hours and r["horizon_hours"] != horizon_hours:
            continue
        filtered.append(
            {
                "target": r["target"],
                "split": r["split"],
                "horizon_hours": r["horizon_hours"],
                "model": r["model"],
                "rows": r["rows"],
                "metrics": r["metrics"],
            }
        )
    return filtered


@app.get("/api/best-models")
def get_best_models(split: str = "validation") -> list[dict]:
    """Return the best-performing model (by R²) for each target/horizon on a given split."""
    results = [
        r
        for r in services.load_results()
        if r["split"] == split and r["metrics"].get("r2") is not None
    ]
    best: dict[tuple, dict] = {}
    for r in results:
        key = (r["target"], r["horizon_hours"])
        if key not in best or r["metrics"]["r2"] > best[key]["metrics"]["r2"]:
            best[key] = r
    return [
        {
            "target": r["target"],
            "horizon_hours": r["horizon_hours"],
            "model": r["model"],
            "metrics": r["metrics"],
        }
        for r in sorted(best.values(), key=lambda x: (x["target"], x["horizon_hours"]))
    ]


@app.get("/api/generalization-gap")
def get_generalization_gap() -> list[dict]:
    """Compare validation vs testing R² for every trained (target, horizon, model) combo."""
    results = services.load_results()
    val = {
        (r["target"], r["horizon_hours"], r["model"]): r
        for r in results
        if r["split"] == "validation"
    }
    test = {
        (r["target"], r["horizon_hours"], r["model"]): r for r in results if r["split"] == "testing"
    }

    rows = []
    for key, val_r in val.items():
        test_r = test.get(key)
        if test_r is None:
            continue
        target, horizon_hours, model = key
        gap = val_r["metrics"]["r2"] - test_r["metrics"]["r2"]
        rows.append(
            {
                "target": target,
                "horizon_hours": horizon_hours,
                "model": model,
                "val_r2": val_r["metrics"]["r2"],
                "test_r2": test_r["metrics"]["r2"],
                "gap": round(gap, 4),
            }
        )
    return sorted(rows, key=lambda x: (x["target"], x["horizon_hours"], x["model"]))


@app.get("/api/feature-importance")
def get_feature_importance(
    target: str,
    horizon_hours: int,
    model: str,
    top_n: int = Query(default=10, ge=1, le=100),
) -> list[dict]:
    df = services.load_feature_importance(target, horizon_hours, model)
    if df is None:
        raise HTTPException(
            status_code=404, detail="Feature importance not found for this combination."
        )
    df = df.sort_values("Importance", ascending=False).head(top_n)
    return df.to_dict(orient="records")


@app.get("/api/forecast/production")
def get_production_forecast(
    target: str,
    horizon_hours: int,
    model: str = "catboost",
    period_start: str = "2026-05-01",
    period_end: str = "2026-05-31",
    limit: int = Query(default=744, ge=1, le=744),
) -> dict:
    """Hourly forward forecast for May 2026 from the winning (CatBoost) model.

    Each hour is predicted from the observed feature row ``horizon_hours`` earlier
    (the latest historical sliding window). A missing CatBoost prospective
    artifact falls back to the report's best model, then Random Forest.
    """
    try:
        return services.production_forecast(
            target, horizon_hours, model, period_start, period_end, limit
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/forecast")
def get_forecast(
    target: str,
    horizon_hours: int,
    model: str | None = None,
    split: str = "testing",
    limit: int = Query(default=200, ge=1, le=5000),
) -> dict:
    """Return actual vs predicted values for the most recent rows of a split.

    Omit ``model`` to serve the best model chosen by the evaluation report
    (``models/best_models.json``); a missing artifact falls back to Random Forest.
    """
    try:
        effective_model, trained_model = services.resolve_forecast_model(
            target, horizon_hours, split, model
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    result = services.find_result(target, horizon_hours, effective_model, split)
    if result is None:
        raise HTTPException(status_code=404, detail="No trained result found for this combination.")

    df = services.load_forecast_frame()
    split_df = services.get_split_frame(df, split)
    X, y = services.prepare_horizon_dataset(split_df, target, horizon_hours, result["features"])

    preds = trained_model.predict(X)
    tail_index = X.index[-limit:]
    return {
        "target": target,
        "horizon_hours": horizon_hours,
        "model": effective_model,
        "requested_model": model,
        "split": split,
        "timestamps": [ts.isoformat() for ts in tail_index],
        "actual": y.loc[tail_index].tolist(),
        "predicted": preds[-limit:].tolist(),
        "metrics": result["metrics"],
    }
