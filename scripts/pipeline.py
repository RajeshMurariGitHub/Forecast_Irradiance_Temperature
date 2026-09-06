"""NASA POWER data pipeline for the solar forecasting project."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import requests
import yaml

try:
    from scripts.utils import ClearSkyModel, FeatureEngineer, SolarGeometry
except ModuleNotFoundError:
    from utils import ClearSkyModel, FeatureEngineer, SolarGeometry

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def build_nasa_power_params(config: dict, start_date: str, end_date: str) -> dict:
    location = config["location"]
    start = pd.Timestamp(start_date).strftime("%Y%m%d")
    end = pd.Timestamp(end_date).strftime("%Y%m%d")
    return {
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "start": start,
        "end": end,
        "community": "RE",
        "parameters": (
            "T2M,T2MDEW,RH2M,CLOUD_AMT,WS10M,PRECTOTCORR,PS,"
            "ALLSKY_SFC_SW_DWN,ALLSKY_SFC_SW_DNI,ALLSKY_SFC_SW_DIFF"
        ),
        "format": "JSON",
    }


def chunk_date_ranges(start_date: str, end_date: str, max_days: int = 365) -> List[Tuple[str, str]]:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if end < start:
        raise ValueError(f"End date {end_date} is earlier than start date {start_date}.")

    if max_days <= 0:
        raise ValueError("max_days must be greater than zero.")

    ranges: List[Tuple[str, str]] = []
    current = start
    while current <= end:
        year_end = pd.Timestamp(year=current.year, month=12, day=31)
        chunk_end = min(year_end, end)
        chunk_start = current.strftime("%Y-%m-%d")
        chunk_stop = chunk_end.strftime("%Y-%m-%d")
        ranges.append((chunk_start, chunk_stop))
        current = chunk_end + pd.Timedelta(days=1)
    return ranges


def _parse_nasa_power_response(payload: dict) -> pd.DataFrame:
    if "properties" not in payload or "parameter" not in payload["properties"]:
        raise ValueError(f"Unexpected NASA POWER response. Keys: {list(payload.keys())}")

    parameters = payload["properties"]["parameter"]
    records: Dict[str, List[float]] = {key: [] for key in parameters}
    timestamps: List[pd.Timestamp] = []

    for param_name, values in parameters.items():
        for key, value in values.items():
            ts = pd.to_datetime(key, format="%Y%m%d%H")
            timestamps.append(ts)
            records[param_name].append(float(value) if value is not None else float("nan"))

    if not timestamps:
        raise ValueError("No NASA POWER data returned for the requested period.")

    time_index = sorted(set(timestamps))
    rows: List[dict] = []
    for ts in time_index:
        row: Dict[str, object] = {"timestamp": ts}
        for param_name, values in parameters.items():
            raw_key = ts.strftime("%Y%m%d%H")
            value = values.get(raw_key, None)
            row[param_name] = float(value) if value is not None else float("nan")
        rows.append(row)

    df = pd.DataFrame(rows).set_index("timestamp").sort_index()

    rename_map = {
        "T2M": "temperature",
        "T2MDEW": "dew_point",
        "RH2M": "humidity",
        "CLOUD_AMT": "cloud_cover",
        "WS10M": "wind_speed",
        "PRECTOTCORR": "precipitation",
        "PS": "pressure",
        "ALLSKY_SFC_SW_DWN": "GHI_energy",
        "ALLSKY_SFC_SW_DNI": "DNI",
        "ALLSKY_SFC_SW_DIFF": "DHI",
    }
    return df.rename(columns=rename_map)


def fetch_nasa_power_data(
    config: dict,
    output_path: str | Path,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    data_cfg = config["data"]
    location = config["location"]
    start_date = start_date or data_cfg["collection"]["start"]
    end_date = end_date or data_cfg["collection"]["end"]

    url = data_cfg["api_url"]
    logger.info(
        "Requesting NASA POWER data for %s, %s from %s to %s",
        location["name"],
        (location["latitude"], location["longitude"]),
        start_date,
        end_date,
    )

    chunked_frames: List[pd.DataFrame] = []
    for chunk_start, chunk_end in chunk_date_ranges(start_date, end_date):
        params = build_nasa_power_params(config, chunk_start, chunk_end)
        response = requests.get(url, params=params, timeout=180)
        response.raise_for_status()
        payload = response.json()
        chunk_df = _parse_nasa_power_response(payload)
        chunked_frames.append(chunk_df)

    if not chunked_frames:
        raise ValueError("No NASA POWER data returned for the requested period.")

    df = pd.concat(chunked_frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]

    if "GHI_W_m2" not in df.columns and "GHI_energy" in df.columns:
        df["GHI"] = df["GHI_energy"]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    logger.info("Saved %s rows to %s", len(df), output_path)
    return df


# NASA POWER encodes missing observations as -999. Treat anything at or below
# this threshold as missing so the sentinels never reach features or targets.
NASA_POWER_FILL_VALUE = -999.0
_SENTINEL_THRESHOLD = -900.0


def replace_fill_values(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Replace NASA POWER -999 fill values with NaN and report how many were found."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include="number").columns
    sentinel_mask = df[numeric_cols] <= _SENTINEL_THRESHOLD
    replaced = int(sentinel_mask.to_numpy().sum())
    df[numeric_cols] = df[numeric_cols].mask(sentinel_mask)
    return df, replaced


def standardize_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = "timestamp"
    df = df.sort_index()

    # Ensure a usable primary GHI column.
    if "GHI" not in df.columns and "GHI_energy" in df.columns:
        df["GHI"] = df["GHI_energy"]

    # Keep the target feature names described in the project.
    column_aliases = {
        "temperature": "temperature",
        "dew_point": "dew_point",
        "humidity": "humidity",
        "cloud_cover": "cloud_cover",
        "wind_speed": "wind_speed",
        "precipitation": "precipitation",
        "pressure": "pressure",
        "GHI_energy": "GHI_energy",
        "DNI": "DNI",
        "DHI": "DHI",
        "GHI": "GHI",
    }
    df = df.rename(columns={k: v for k, v in column_aliases.items() if k in df.columns})

    # Basic quality handling.
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # NASA POWER reports surface pressure in kPa; the project works in hPa.
    if "pressure" in df.columns and df["pressure"].median(skipna=True) < 200:
        df["pressure"] = df["pressure"] * 10.0

    return df


def build_quality_report(
    df: pd.DataFrame,
    config: dict,
    duplicate_timestamps: int,
    missing_timestamps: int,
    fill_values_replaced: int = 0,
    values_interpolated: int = 0,
) -> dict:
    """Summarize data quality without deleting statistically unusual observations."""
    range_violations = {}
    for column, bounds in config.get("physical_ranges", {}).items():
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        invalid = (values < bounds["min"]) | (values > bounds["max"])
        range_violations[column] = int(invalid.sum())

    return {
        "rows": int(len(df)),
        "start": df.index.min().isoformat(),
        "end": df.index.max().isoformat(),
        "duplicate_timestamps_removed": duplicate_timestamps,
        "missing_timestamps_inserted": missing_timestamps,
        "fill_values_replaced": fill_values_replaced,
        "values_interpolated": values_interpolated,
        "missing_values": {column: int(count) for column, count in df.isna().sum().items()},
        "physical_range_violations": range_violations,
        "frequency": "hourly",
        "outlier_deletion_performed": False,
    }


def clean_and_freeze_data(
    input_path: str | Path, output_path: str | Path, config: dict | None = None
) -> pd.DataFrame:
    df = pd.read_csv(input_path, index_col=0)
    df = standardize_raw_data(df)

    # Replace NASA POWER -999 fill values with NaN before any continuity work.
    df, fill_values_replaced = replace_fill_values(df)

    # Remove duplicate timestamps.
    duplicate_timestamps = int(df.index.duplicated().sum())
    df = df[~df.index.duplicated(keep="last")]

    # Enforce hourly continuity and fill missing timestamp rows if needed.
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="h")
    missing_timestamps = int(len(full_index.difference(df.index)))
    df = df.reindex(full_index)

    # Interpolate short gaps (fill values + inserted rows) so downstream lag and
    # rolling features are not built on NaNs. Longer gaps are left for the model
    # imputer to handle rather than fabricated here.
    numeric_cols = df.select_dtypes(include="number").columns
    na_before = int(df[numeric_cols].isna().to_numpy().sum())
    df[numeric_cols] = df[numeric_cols].interpolate(method="time", limit=6, limit_area="inside")
    values_interpolated = na_before - int(df[numeric_cols].isna().to_numpy().sum())

    # Irradiance and other physical quantities cannot be negative after interpolation.
    physical = ("GHI", "GHI_energy", "DNI", "DHI", "precipitation")
    non_negative = [col for col in physical if col in df.columns]
    df[non_negative] = df[non_negative].clip(lower=0.0)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    if config is not None:
        report_path = output_path.with_name("data_quality_report.json")
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(
                build_quality_report(
                    df,
                    config,
                    duplicate_timestamps,
                    missing_timestamps,
                    fill_values_replaced,
                    values_interpolated,
                ),
                fh,
                indent=2,
            )
        logger.info("Data quality report saved to %s", report_path)
    logger.info("Cleaned dataset saved to %s with %s rows", output_path, len(df))
    return df


def build_forecast_dataset(
    config: dict, input_path: str | Path, output_path: str | Path
) -> pd.DataFrame:
    df = pd.read_csv(input_path, index_col=0)
    df.index = pd.to_datetime(df.index)
    df.index.name = "timestamp"
    df = df.sort_index().copy()

    if "GHI" not in df.columns and "GHI_energy" in df.columns:
        df["GHI"] = df["GHI_energy"]

    if "temperature" not in df.columns and "T2M" in df.columns:
        df["temperature"] = df["T2M"]

    location = config["location"]
    if "solar_elevation" not in df.columns:
        solar = SolarGeometry(
            latitude=location["latitude"],
            longitude=location["longitude"],
            elevation=location.get("elevation", 0),
            timezone=location.get("timezone", "UTC"),
        )
        df = solar.add_solar_features(df)

    if "clear_sky_ghi" not in df.columns:
        clear_sky = ClearSkyModel(
            latitude=location["latitude"],
            longitude=location["longitude"],
            elevation=location.get("elevation", 0),
            timezone=location.get("timezone", "UTC"),
        )
        df = clear_sky.add_clearsky_features(df, ghi_col="GHI")

    df = (
        FeatureEngineer(df)
        .add_temporal_features()
        .add_lag_features(["GHI", "temperature", "humidity"], lags=[1, 6, 12, 24])
        .add_rolling_features(
            ["GHI", "temperature", "humidity"],
            windows=[6, 12, 24],
            stats=["mean", "std"],
        )
        .handle_missing_values(method="forward_fill", limit=24)
        .get_dataframe()
    )

    feature_cols = [c for c in df.columns if c not in {"GHI_energy", "DNI", "DHI"}]
    df = df[feature_cols].copy()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".parquet":
        df.to_parquet(output_path, index=True)
    else:
        df.to_csv(output_path)
    logger.info(
        "Forecast-ready dataset saved to %s with %s rows and %s columns",
        output_path,
        len(df),
        len(df.columns),
    )
    return df


SCRIPTS_DIR = Path(__file__).resolve().parent

# Every model in the benchmark loop, persistence baseline first.
ALL_MODELS = (
    "persistence,ridge,random_forest,extra_trees,gradient_boost,xgboost,lightgbm,catboost"
)

# Stages that run the full sequence after data prep.
_COLLECT = {"collect", "all"}
_CLEAN = {"clean", "all", "full"}
_FEATURES = {"features", "forecast", "all", "full"}
_TRAIN = {"train", "all", "full"}
_EVALUATE = {"evaluate", "all", "full"}
_DIAGNOSTICS = {"diagnostics", "all", "full"}


def _run_script(script: str, *args: str) -> None:
    cmd = [sys.executable, str(SCRIPTS_DIR / script), *args]
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def train_all_models(config_path: str) -> None:
    """Benchmark every model on validation and testing, then lock the prospective winners."""
    for split in ("validation", "testing"):
        _run_script(
            "train_models.py", "--config", config_path, "--models", ALL_MODELS, "--split", split
        )
    _run_script(
        "train_models.py", "--config", config_path, "--split", "prospective", "--select-best"
    )


def evaluate_models() -> None:
    """Summaries, comparison report, and the lowest-RMSE best-model selection."""
    _run_script("analyze_models.py")
    _run_script("generate_report.py")


def generate_diagnostics() -> None:
    """Feature-importance matrix for the trained models (models/feature_importance*)."""
    _run_script("model_diagnostics.py")


def run_stage(
    stage: str,
    config_path: str = "config/config.yaml",
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame | None:
    config = load_config(config_path)
    data_cfg = config["data"]
    raw_dir = Path(data_cfg["raw_dir"])
    processed_dir = Path(data_cfg["processed_dir"])
    raw_path = raw_dir / data_cfg["raw_filename"]
    processed_path = processed_dir / data_cfg["processed_filename"]
    forecast_path = processed_dir / data_cfg["forecast_dataset_filename"]

    if stage in _COLLECT:
        fetch_nasa_power_data(config, raw_path, start_date=start_date, end_date=end_date)

    if stage in _CLEAN:
        clean_and_freeze_data(raw_path, processed_path, config)

    frame: pd.DataFrame | None = None
    if stage in _FEATURES:
        frame = build_forecast_dataset(config, processed_path, forecast_path)

    if stage in _TRAIN:
        train_all_models(config_path)

    if stage in _EVALUATE:
        evaluate_models()

    if stage in _DIAGNOSTICS:
        generate_diagnostics()

    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the solar forecasting pipeline.")
    parser.add_argument(
        "--stage",
        choices=[
            "collect", "clean", "features", "forecast",
            "train", "evaluate", "diagnostics", "full", "all",
        ],
        default="all",
        help=(
            "Stage to run. 'full' = clean -> features -> train -> evaluate -> "
            "diagnostics; 'all' additionally re-collects raw NASA POWER data first."
        ),
    )
    parser.add_argument("--config", default="config/config.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--start-date",
        default=None,
        help="Overridden start date for NASA POWER collection (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Overridden end date for NASA POWER collection (YYYY-MM-DD).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    run_stage(args.stage, args.config, start_date=args.start_date, end_date=args.end_date)


if __name__ == "__main__":
    main()
