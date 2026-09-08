import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api import services
from api.main import get_config
from scripts.pipeline import (
    build_quality_report,
    chunk_date_ranges,
    replace_fill_values,
    standardize_raw_data,
)
from scripts.train_models import get_split_bounds, model_artifact_path, prepare_horizon_dataset
from scripts.utils.solar_geometry import SolarGeometry


def test_chunk_date_ranges_splits_long_windows():
    chunks = chunk_date_ranges("2020-01-01", "2026-07-31", max_days=365)

    assert chunks[0] == ("2020-01-01", "2020-12-31")
    assert chunks[-1] == ("2026-01-01", "2026-07-31")
    assert len(chunks) == 7


def test_split_bounds_include_the_whole_configured_end_date():
    config = {"data_split": {"testing": {"start": "2025-01-01", "end": "2025-12-31"}}}

    start, end_exclusive = get_split_bounds("testing", config)

    assert start == pd.Timestamp("2025-01-01 00:00")
    assert end_exclusive == pd.Timestamp("2026-01-01 00:00")


def test_horizon_samples_do_not_cross_the_split_boundary():
    index = pd.date_range("2025-01-01", periods=48, freq="h")
    df = pd.DataFrame({"GHI": range(48), "temperature": range(48)}, index=index)

    X, y, features = prepare_horizon_dataset(
        df,
        "GHI",
        horizon_days=1,
        origin_start=pd.Timestamp("2025-01-01"),
        origin_end_exclusive=pd.Timestamp("2025-01-03"),
    )

    assert list(features) == ["temperature"]
    assert X.index[-1] == pd.Timestamp("2025-01-01 23:00")
    assert y.index.equals(X.index)


def test_model_artifact_path_uses_day_horizons_consistently():
    path = model_artifact_path(Path("models"), "GHI", 1, "ridge", "validation")

    assert path.name == "GHI_1d_ridge_validation.joblib"


def test_api_load_model_uses_the_training_artifact_name(monkeypatch):
    expected = model_artifact_path(services.MODELS_DIR, "GHI", 1, "ridge", "validation")
    # No trained artifacts on a fresh checkout: stub the existence check + loader.
    monkeypatch.setattr(services.Path, "exists", lambda self: True)
    monkeypatch.setattr(services.joblib, "load", lambda path: path)
    services.load_model.cache_clear()

    assert services.load_model("GHI", 24, "ridge", "validation") == expected
    services.load_model.cache_clear()


def test_api_config_exposes_horizons_in_hours_for_dashboard_requests():
    assert get_config()["horizons"] == [24, 48, 72, 120, 168, 240, 336, 720]


def test_quality_report_counts_range_violations_without_outlier_removal():
    index = pd.date_range("2025-01-01", periods=2, freq="h")
    df = pd.DataFrame({"GHI": [0.0, 1500.0]}, index=index)
    config = {"physical_ranges": {"GHI": {"min": 0, "max": 1400}}}

    report = build_quality_report(df, config, duplicate_timestamps=0, missing_timestamps=0)

    assert report["physical_range_violations"] == {"GHI": 1}
    assert report["outlier_deletion_performed"] is False


def test_replace_fill_values_maps_nasa_power_sentinels_to_nan():
    index = pd.date_range("2025-01-01", periods=3, freq="h")
    df = pd.DataFrame({"GHI": [0.0, -999.0, 250.0], "temperature": [18.0, 19.0, -999.0]}, index=index)

    cleaned, replaced = replace_fill_values(df)

    assert replaced == 2
    assert cleaned["GHI"].isna().tolist() == [False, True, False]
    assert cleaned["temperature"].isna().tolist() == [False, False, True]


def test_standardize_raw_data_converts_pressure_kpa_to_hpa():
    index = pd.date_range("2025-01-01", periods=2, freq="h")
    df = pd.DataFrame({"pressure": [95.3, 95.4]}, index=index)

    standardized = standardize_raw_data(df)

    assert standardized["pressure"].round(1).tolist() == [953.0, 954.0]


def test_solar_flags_distinguish_sunrise_from_sunset(monkeypatch):
    index = pd.date_range("2025-01-01", periods=4, freq="h")
    geometry = SolarGeometry.__new__(SolarGeometry)
    elevations = pd.Series([-5.0, 3.0, 2.0, -4.0], index=index)
    solar_position = pd.DataFrame(
        {"solar_elevation": elevations, "solar_azimuth": 0.0, "solar_zenith": 90.0},
        index=index,
    )
    monkeypatch.setattr(geometry, "calculate_solar_position", lambda _: solar_position)
    monkeypatch.setattr(geometry, "calculate_declination", lambda _: [0.0] * len(index))
    monkeypatch.setattr(geometry, "calculate_hour_angle", lambda _: [0.0] * len(index))
    monkeypatch.setattr(geometry, "calculate_airmass", lambda _: [1.0] * len(index))

    result = geometry.add_solar_features(pd.DataFrame(index=index))

    assert result.loc[index[1], "is_sunrise"]
    assert result.loc[index[3], "is_sunset"]
    assert not result.loc[index[1], "is_sunset"]
