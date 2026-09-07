"""Coverage for the phase additions: persistence baseline, best-model selection,
model resolution + fallback, the production forecast window, and feature-name
resolution in the diagnostics script."""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api import services  # noqa: E402
from scripts import analyze_models, model_diagnostics  # noqa: E402
from scripts import compare_sources  # noqa: E402
from scripts.pipeline import (  # noqa: E402
    OPEN_METEO_HOURLY,
    _parse_open_meteo_response,
    add_open_meteo_features,
    local_offset_hours,
    local_tz_name,
)
from scripts.utils.solar_geometry import SolarGeometry  # noqa: E402
from scripts.train_models import (  # noqa: E402
    evaluate_model,
    get_split_bounds,
    prepare_horizon_dataset,
    train_split_model,
)

CONFIG = {
    "data_split": {
        "training": {"start": "2025-01-01", "end": "2025-01-10"},
        "validation": {"start": "2025-01-11", "end": "2025-01-15"},
    }
}


# --------------------------------------------------------------------------- #
# Persistence baseline (scripts/train_models.py)
# --------------------------------------------------------------------------- #
def test_evaluate_model_drops_r2_for_a_single_observation():
    single = evaluate_model(pd.Series([5.0]), np.array([4.0]))
    many = evaluate_model(pd.Series([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))

    assert single["r2"] is None and single["mae"] == 1.0
    assert many["r2"] == pytest.approx(1.0)


def test_persistence_split_model_carries_t_minus_24h_forward(tmp_path):
    idx = pd.date_range("2025-01-01", "2025-01-16 23:00", freq="h")
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {"GHI": rng.normal(300, 50, len(idx)), "temperature": rng.normal(25, 3, len(idx))},
        index=idx,
    )

    result = train_split_model(
        df, "GHI", horizon_days=1, model_name="persistence",
        split_name="validation", config=CONFIG, model_dir=tmp_path,
    )

    assert result["model"] == "persistence"
    assert result["features"] == ["GHI_persist_t-24h"]
    saved = joblib.load(tmp_path / Path(result["model_path"]).name)
    assert saved == {"type": "persistence", "target": "GHI", "lag_hours": 24}

    # RMSE must equal a plain 24h-lag forecast over the exact scored rows.
    start, end = get_split_bounds("validation", CONFIG)
    scored_x, scored_y, _ = prepare_horizon_dataset(df, "GHI", 1, start, end)
    preds = df["GHI"].shift(24).reindex(scored_x.index).to_numpy()
    keep = ~np.isnan(preds)
    expected_rmse = float(np.sqrt(np.mean((scored_y.to_numpy()[keep] - preds[keep]) ** 2)))
    assert result["rows"] == int(keep.sum())
    assert result["metrics"]["rmse"] == pytest.approx(expected_rmse, rel=1e-9)


# --------------------------------------------------------------------------- #
# Best-model selection (scripts/analyze_models.py)
# --------------------------------------------------------------------------- #
def _result(target, days, model, rmse, split="validation"):
    return {
        "target": target, "split": split, "horizon_days": days, "horizon_hours": days * 24,
        "model": model, "rows": 100, "features": ["a", "b"],
        "metrics": {"mae": rmse / 2, "rmse": rmse, "r2": 0.9},
        "model_path": f"models/{target}_{days}d_{model}_{split}.joblib",
    }


def test_select_best_models_picks_lowest_rmse_per_horizon_and_overall():
    results = [
        _result("GHI", 1, "catboost", 40.0),
        _result("GHI", 1, "random_forest", 30.0),   # winner @1d
        _result("GHI", 7, "catboost", 25.0),        # winner @7d
        _result("GHI", 7, "random_forest", 55.0),
        _result("GHI", 7, "ridge", 25.0, split="testing"),  # ignored: wrong split
    ]

    selection = analyze_models.select_best_models(analyze_models.validation_comparison_frame(results))

    per_horizon = selection["best_per_horizon"]["GHI"]
    assert per_horizon[1]["model"] == "random_forest"
    assert per_horizon[7]["model"] == "catboost"
    assert per_horizon[7]["model_path"].endswith("GHI_7d_catboost_validation.joblib")
    # overall = lowest mean RMSE across horizons: catboost (40+25)/2=32.5 vs rf (30+55)/2=42.5
    assert selection["best_overall"]["GHI"]["model"] == "catboost"


def test_write_best_model_metadata_round_trips(tmp_path):
    results = [_result("GHI", 1, "random_forest", 30.0), _result("GHI", 2, "catboost", 35.0)]
    selection = analyze_models.select_best_models(analyze_models.validation_comparison_frame(results))

    path = analyze_models.write_best_model_metadata(selection, path=tmp_path / "best.json")
    import json

    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["selection_metric"] == "rmse"
    assert written["best_per_horizon"]["GHI"]["1"]["model"] == "random_forest"


# --------------------------------------------------------------------------- #
# Model resolution + fallback (api/services.py)
# --------------------------------------------------------------------------- #
class _Stub:
    def predict(self, x):  # noqa: D401
        return np.zeros(len(x))


@pytest.fixture
def resolver(monkeypatch, tmp_path):
    """resolve_forecast_model wired to synthetic artifacts under tmp_path."""
    def fake_path(target, horizon_hours, model_name, split_name):
        return tmp_path / f"{target}_{horizon_hours // 24}d_{model_name}_{split_name}.joblib"

    loaded: dict[str, object] = {}

    def fake_load(target, horizon_hours, model_name, split_name):
        return loaded[model_name]

    monkeypatch.setattr(services, "model_artifact_path", fake_path)
    monkeypatch.setattr(services, "load_model", fake_load)
    monkeypatch.setattr(
        services, "load_best_model_metadata",
        lambda: {"best_per_horizon": {"GHI": {"5": {"model": "catboost"}}}},
    )

    def make(model_name, obj):
        fake_path("GHI", 120, model_name, "prospective").write_text("x", encoding="utf-8")
        loaded[model_name] = obj

    return make


def test_resolve_prefers_report_best_when_no_model_requested(resolver):
    resolver("catboost", _Stub())
    name, model = services.resolve_forecast_model("GHI", 120, "prospective", None)
    assert name == "catboost" and hasattr(model, "predict")


def test_resolve_falls_back_to_random_forest_when_artifact_missing(resolver):
    resolver("random_forest", _Stub())  # catboost file never created
    name, _ = services.resolve_forecast_model("GHI", 120, "prospective", None)
    assert name == "random_forest"


def test_resolve_honours_explicit_model(resolver):
    resolver("lightgbm", _Stub())
    resolver("random_forest", _Stub())
    name, _ = services.resolve_forecast_model("GHI", 120, "prospective", "lightgbm")
    assert name == "lightgbm"


def test_resolve_skips_non_predicting_persistence_artifact(resolver):
    resolver("catboost", {"type": "persistence"})  # no .predict
    resolver("random_forest", _Stub())
    name, _ = services.resolve_forecast_model("GHI", 120, "prospective", None)
    assert name == "random_forest"


def test_resolve_raises_when_nothing_usable(resolver):
    with pytest.raises(FileNotFoundError):
        services.resolve_forecast_model("GHI", 120, "prospective", None)


# --------------------------------------------------------------------------- #
# Production forecast window (api/services.py)
# --------------------------------------------------------------------------- #
def test_ingest_history_window_aligns_origin_to_target_minus_horizon(monkeypatch):
    idx = pd.date_range("2026-04-01", "2026-05-31 23:00", freq="h")
    frame = pd.DataFrame({"GHI": np.arange(len(idx), dtype=float), "feat": 1.0}, index=idx)
    monkeypatch.setattr(services, "load_forecast_frame", lambda: frame)

    features, actual = services.ingest_history_window("GHI", 24, ["feat"], "2026-05-01", "2026-05-31")

    assert len(features) == 744  # full hourly May
    assert features.index[0] == pd.Timestamp("2026-05-01 00:00")
    assert actual.iloc[0] == frame.loc["2026-05-01 00:00", "GHI"]


def test_production_forecast_predicts_744_hours_with_metrics(monkeypatch):
    idx = pd.date_range("2026-04-01", "2026-05-31 23:00", freq="h")
    frame = pd.DataFrame({"GHI": np.arange(len(idx), dtype=float), "feat": 2.0}, index=idx)
    monkeypatch.setattr(services, "load_forecast_frame", lambda: frame)
    monkeypatch.setattr(services, "_prospective_feature_cols", lambda t, h: ["feat"])
    monkeypatch.setattr(
        services, "resolve_forecast_model", lambda *a: ("catboost", _Stub())
    )

    out = services.production_forecast("GHI", 24, "catboost", limit=10)

    assert out["model"] == "catboost"
    assert out["period"] == {"start": "2026-05-01", "end": "2026-05-31"}
    assert len(out["predicted"]) == 10 and len(out["timestamps"]) == 10
    assert out["metrics"]["n_observed"] == 744  # metric uses the full overlap, not the limit
    assert out["metrics"]["mae"] > 0


# --------------------------------------------------------------------------- #
# Feature-name resolution (scripts/model_diagnostics.py)
# --------------------------------------------------------------------------- #
def _fit_pipeline(estimator, frame, y):
    pipe = Pipeline([("imputer", SimpleImputer()), ("model", estimator)])
    pipe.fit(frame, y)
    return pipe


def test_extract_feature_importance_reads_names_from_the_pipeline(tmp_path):
    frame = pd.DataFrame({"alpha": [1.0, 2, 3, 4], "beta": [4.0, 3, 2, 1]})
    y = pd.Series([1.0, 2, 3, 4])
    path = tmp_path / "m.joblib"
    joblib.dump(_fit_pipeline(RandomForestRegressor(n_estimators=5, random_state=0), frame, y), path)

    importance = model_diagnostics.extract_feature_importance(path, fallback_features=["wrong", "x"])

    assert set(importance) == {"alpha", "beta"}  # from imputer.feature_names_in_, not the fallback


def test_extract_feature_importance_uses_abs_coef_for_linear_models(tmp_path):
    frame = pd.DataFrame({"alpha": [1.0, 2, 3, 4], "beta": [0.0, 0, 0, 0]})
    y = pd.Series([2.0, 4, 6, 8])
    path = tmp_path / "ridge.joblib"
    joblib.dump(_fit_pipeline(Ridge(), frame, y), path)

    importance = model_diagnostics.extract_feature_importance(path)

    assert importance["alpha"] > importance["beta"] >= 0.0


def test_extract_feature_importance_returns_empty_for_persistence_dict(tmp_path):
    path = tmp_path / "persist.joblib"
    joblib.dump({"type": "persistence", "target": "GHI", "lag_hours": 24}, path)

    assert model_diagnostics.extract_feature_importance(path) == {}


# --------------------------------------------------------------------------- #
# Local-time handling (scripts/pipeline.py + scripts/utils/solar_geometry.py)
# --------------------------------------------------------------------------- #
def test_local_offset_and_tz_for_hyderabad():
    assert local_offset_hours(78.4867) == 5
    assert local_tz_name(78.4867) == "Etc/GMT-5"  # Etc/GMT-5 == UTC+5
    assert local_tz_name(-75.0) == "Etc/GMT+5"    # western hemisphere, UTC-5


def test_solar_position_is_localized_not_treated_as_utc():
    """Solar noon (max elevation) must fall near local midday, not ~5h off."""
    geo = SolarGeometry(latitude=17.385, longitude=78.487, elevation=500, timezone="Etc/GMT-5")
    times = pd.date_range("2025-06-15 00:00", periods=24, freq="h")  # tz-naive local

    elevation = geo.calculate_solar_position(times)["solar_elevation"]

    peak_hour = int(elevation.to_numpy().argmax())
    assert 11 <= peak_hour <= 13
    assert elevation.iloc[2] < 0  # 02:00 local is night


# --------------------------------------------------------------------------- #
# Open-Meteo ingest + source comparison (scripts/pipeline.py, compare_sources.py)
# --------------------------------------------------------------------------- #
def test_parse_open_meteo_response_maps_and_keeps_extra_cloud_layers():
    payload = {
        "hourly": {
            "time": ["2025-06-15T00:00", "2025-06-15T01:00"],
            "temperature_2m": [25.5, 24.9],
            "shortwave_radiation": [0.0, 0.0],
            "cloud_cover_low": [1, 0],
            "surface_pressure": [953.1, 953.2],
        }
    }
    df = _parse_open_meteo_response(payload)

    assert list(df.columns) == ["temperature", "cloud_cover_low", "pressure", "GHI"]
    assert df.index[0] == pd.Timestamp("2025-06-15 00:00")
    assert "shortwave_radiation" in OPEN_METEO_HOURLY and OPEN_METEO_HOURLY["shortwave_radiation"] == "GHI"


def test_add_open_meteo_features_shifts_radiation_and_builds_ratio(tmp_path):
    idx = pd.date_range("2025-06-15 00:00", periods=6, freq="h")
    nasa = pd.DataFrame({"GHI": [0.0, 0.0, 50.0, 100.0, 200.0, 300.0]}, index=idx)

    om_path = tmp_path / "om.csv"
    pd.DataFrame(
        {"GHI": [0.0, 10.0, 60.0, 120.0, 240.0, 360.0], "cloud_cover_low": [90, 80, 70, 60, 50, 40]},
        index=idx,
    ).to_csv(om_path)

    out = add_open_meteo_features(nasa.copy(), om_path)

    # radiation shifted -1h: om_ghi[t] == raw om GHI[t+1]
    assert out["om_ghi"].tolist()[:5] == [10.0, 60.0, 120.0, 240.0, 360.0]
    assert out["om_cloud_low"].tolist() == [90, 80, 70, 60, 50, 40]  # cloud not shifted
    # ratio = NASA GHI / shifted om_ghi, daytime only. At 03:00 that is 100 / 240.
    assert out.loc[idx[3], "om_ghi_ratio"] == pytest.approx(100.0 / 240.0, rel=1e-6)
    assert pd.isna(out.loc[idx[0], "om_ghi_ratio"])  # night


def test_add_open_meteo_features_no_file_is_a_noop(tmp_path):
    nasa = pd.DataFrame({"GHI": [1.0, 2.0]}, index=pd.date_range("2025-01-01", periods=2, freq="h"))
    out = add_open_meteo_features(nasa.copy(), tmp_path / "missing.csv")
    assert list(out.columns) == ["GHI"]


def test_pair_stats_and_best_lag():
    rng = np.random.default_rng(1)
    base = rng.normal(300, 80, 500)
    nasa = base + rng.normal(0, 5, 500)
    other = base + 20.0  # constant +20 bias, otherwise identical

    stats = compare_sources._pair_stats(nasa, other)
    assert stats["n"] == 500
    assert stats["pearson_r"] > 0.99
    assert stats["bias_nasa_minus_openmeteo"] == pytest.approx(-20, abs=2)

    idx = pd.date_range("2025-01-01", periods=500, freq="h")
    lag = compare_sources._best_lag(pd.Series(base, index=idx), pd.Series(base, index=idx).shift(-1))
    assert lag["best_lag_hours"] == 1  # other is 1h early -> shift +1 realigns


# --------------------------------------------------------------------------- #
# Input validation / path-traversal hardening (api/services.py + api/main.py)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "kwargs",
    [
        {"target": "../../etc/passwd"},
        {"model_name": "../../../models/x"},
        {"split_name": "../.."},
    ],
)
def test_validate_identifiers_rejects_path_traversal(kwargs):
    with pytest.raises(ValueError):
        services.validate_identifiers(**kwargs)


def test_validate_identifiers_accepts_known_values():
    services.validate_identifiers(target="GHI", model_name="catboost", split_name="validation")


def test_ingest_history_window_rejects_oversized_window(monkeypatch):
    monkeypatch.setattr(services, "load_forecast_frame", lambda: pd.DataFrame())
    with pytest.raises(ValueError, match="exceeds"):
        services.ingest_history_window("GHI", 24, ["feat"], "2026-01-01", "2030-01-01")


def test_api_rejects_traversal_and_dos_params():
    from fastapi.testclient import TestClient

    from api.main import app

    client = TestClient(app)
    assert client.get(
        "/api/forecast", params={"target": "GHI", "horizon_hours": 120, "model": "../x"}
    ).status_code == 400
    assert client.get(
        "/api/forecast/production",
        params={"target": "GHI", "horizon_hours": 120, "period_end": "2099-01-01"},
    ).status_code == 400
    assert client.get(
        "/api/feature-importance",
        params={"target": "../s", "horizon_hours": 168, "model": "catboost"},
    ).status_code == 400


def test_catboost_feature_names_are_recovered(tmp_path):
    catboost = pytest.importorskip("catboost")
    frame = pd.DataFrame({"alpha": [1.0, 2, 3, 4, 5], "beta": [5.0, 4, 3, 2, 1]})
    y = pd.Series([1.0, 2, 3, 4, 5])
    model = catboost.CatBoostRegressor(iterations=5, verbose=False)
    model.fit(frame, y)
    path = tmp_path / "cat.joblib"
    joblib.dump(model, path)

    importance = model_diagnostics.extract_feature_importance(path)

    assert set(importance) == {"alpha", "beta"}
