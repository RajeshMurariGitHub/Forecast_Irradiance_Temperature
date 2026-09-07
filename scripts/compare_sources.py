"""Compare the NASA POWER and Open-Meteo (ERA5) hourly datasets.

Reports per-variable agreement (correlation, bias, MAE, RMSE), day-only stats
for the irradiance channels, and the best time-lag for GHI (a check that both
sources sit on the same clock). Writes ``data/processed/source_comparison.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

try:
    from scripts.pipeline import replace_fill_values, standardize_raw_data
except ModuleNotFoundError:  # run directly as scripts/compare_sources.py
    from pipeline import replace_fill_values, standardize_raw_data

SHARED_VARIABLES = [
    "temperature", "dew_point", "humidity", "cloud_cover",
    "wind_speed", "precipitation", "pressure", "GHI", "DNI", "DHI",
]
IRRADIANCE = {"GHI", "DNI", "DHI"}
LAG_RANGE_HOURS = range(-3, 4)


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df[~df.index.duplicated(keep="last")]


def _load_nasa(path: Path) -> pd.DataFrame:
    """NASA POWER raw, with the same sentinel + pressure(kPa->hPa) fixes the pipeline applies."""
    df = standardize_raw_data(_read_csv(path))
    df, _ = replace_fill_values(df)
    if "GHI" not in df.columns and "GHI_energy" in df.columns:
        df["GHI"] = df["GHI_energy"]
    return df


def _load_openmeteo(path: Path) -> pd.DataFrame:
    """Open-Meteo archive is already in project units with nulls for missing."""
    return _read_csv(path)


def _pair_stats(nasa: np.ndarray, other: np.ndarray) -> dict:
    mask = np.isfinite(nasa) & np.isfinite(other)
    n = int(mask.sum())
    if n < 2:
        return {"n": n}
    a, b = nasa[mask], other[mask]
    diff = a - b
    denom = np.mean(np.abs(a)) or np.nan
    return {
        "n": n,
        "pearson_r": round(float(np.corrcoef(a, b)[0, 1]), 4),
        "bias_nasa_minus_openmeteo": round(float(np.mean(diff)), 4),
        "mae": round(float(np.mean(np.abs(diff))), 4),
        "rmse": round(float(np.sqrt(np.mean(diff**2))), 4),
        "nrmse_pct": round(float(np.sqrt(np.mean(diff**2)) / denom * 100), 2),
    }


def _best_lag(nasa: pd.Series, other: pd.Series) -> dict:
    """Shift Open-Meteo by +/-h and report the lag with the highest correlation."""
    scores = {}
    for lag in LAG_RANGE_HOURS:
        shifted = other.shift(lag)
        mask = nasa.notna() & shifted.notna()
        if mask.sum() > 100:
            scores[lag] = float(np.corrcoef(nasa[mask], shifted[mask])[0, 1])
    if not scores:
        return {}
    best = max(scores, key=scores.get)
    return {"best_lag_hours": int(best), "r_at_best_lag": round(scores[best], 4),
            "r_at_zero_lag": round(scores.get(0, float("nan")), 4)}


def compare(nasa: pd.DataFrame, other: pd.DataFrame) -> dict:
    common = nasa.index.intersection(other.index)
    nasa, other = nasa.loc[common], other.loc[common]

    report: dict = {
        "overlap_start": common.min().isoformat(),
        "overlap_end": common.max().isoformat(),
        "overlap_hours": int(len(common)),
        "variables": {},
        "openmeteo_only_columns": sorted(set(other.columns) - set(nasa.columns)),
    }

    daytime = nasa["GHI"] > 5 if "GHI" in nasa.columns else pd.Series(True, index=common)
    for var in SHARED_VARIABLES:
        if var not in nasa.columns or var not in other.columns:
            continue
        entry = _pair_stats(nasa[var].to_numpy(), other[var].to_numpy())
        if var in IRRADIANCE:
            entry["daytime"] = _pair_stats(
                nasa.loc[daytime, var].to_numpy(), other.loc[daytime, var].to_numpy()
            )
        if var == "GHI":
            entry["timing_check"] = _best_lag(nasa["GHI"], other["GHI"])
        report["variables"][var] = entry
    return report


def _markdown(report: dict) -> str:
    header = "| Variable | n | Pearson r | Bias (NASA-OM) | RMSE | nRMSE % |"
    lines = [header, "| " + " | ".join(["---"] * 6) + " |"]
    for var, s in report["variables"].items():
        if "rmse" not in s:
            continue
        lines.append(
            f"| {var} | {s['n']:,} | {s['pearson_r']} | "
            f"{s['bias_nasa_minus_openmeteo']} | {s['rmse']} | {s['nrmse_pct']} |"
        )
    return "\n".join(lines)


def _save_overlay_plot(nasa: pd.DataFrame, other: pd.DataFrame, path: Path, week: str) -> None:
    try:
        # matplotlib is optional and plotting-only.
        import matplotlib  # pylint: disable=import-outside-toplevel

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    except ImportError:
        return

    idx = pd.date_range(week, periods=24 * 7, freq="h")
    idx = idx.intersection(nasa.index).intersection(other.index)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for ax, var, unit in ((axes[0], "GHI", "W/m2"), (axes[1], "temperature", "degC")):
        ax.plot(idx, nasa.loc[idx, var], label="NASA POWER", lw=1.4)
        ax.plot(idx, other.loc[idx, var], label="Open-Meteo (ERA5)", lw=1.4, alpha=0.85)
        ax.set_ylabel(f"{var} ({unit})")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)
    axes[0].set_title(f"NASA POWER vs Open-Meteo - week of {week}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    with open("config/config.yaml", "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    data_cfg = config["data"]
    raw_dir = Path(data_cfg["raw_dir"])
    nasa_path = raw_dir / data_cfg["raw_filename"]
    om_path = raw_dir / data_cfg["open_meteo"]["raw_filename"]

    if not om_path.exists():
        raise FileNotFoundError(
            f"{om_path} not found - run: python scripts/pipeline.py --stage collect-openmeteo"
        )

    nasa, other = _load_nasa(nasa_path), _load_openmeteo(om_path)
    report = compare(nasa, other)

    processed = Path(data_cfg["processed_dir"])
    out = processed / data_cfg["open_meteo"]["comparison_filename"]
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _save_overlay_plot(nasa, other, processed / "source_comparison.png", week="2025-06-09")

    print("=" * 90)
    print("NASA POWER vs Open-Meteo (ERA5) - hourly agreement")
    print("=" * 90)
    print(f"Overlap: {report['overlap_start']} .. {report['overlap_end']} "
          f"({report['overlap_hours']:,} h)\n")
    print(_markdown(report))
    ghi = report["variables"].get("GHI", {})
    if ghi.get("daytime"):
        d = ghi["daytime"]
        print(f"\nGHI daytime only: r={d['pearson_r']}  bias={d['bias_nasa_minus_openmeteo']}  "
              f"RMSE={d['rmse']} W/m2  nRMSE={d['nrmse_pct']}%")
    if ghi.get("timing_check"):
        print(f"GHI timing: {ghi['timing_check']}")
    print(f"\nOpen-Meteo-only columns: {', '.join(report['openmeteo_only_columns']) or 'none'}")
    print(f"\n[ok] Saved {out}")
    png = processed / "source_comparison.png"
    if png.exists():
        print(f"[ok] Saved {png}")


if __name__ == "__main__":
    main()
