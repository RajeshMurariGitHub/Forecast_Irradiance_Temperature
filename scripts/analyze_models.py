"""Analyze and compare trained model results."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Friendly names for the two forecast targets.
TARGET_LABELS = {"GHI": "Irradiance", "temperature": "Temperature"}

BEST_MODEL_METADATA_PATH = Path("models/best_models.json")


def load_results(results_file: str = "models/training_results.json") -> list[dict]:
    """Load training results from JSON."""
    with open(results_file, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """``df[name]`` with the static type pinned to ``Series``."""
    series = df[name]
    assert isinstance(series, pd.Series)
    return series


def _rows(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Boolean-mask a frame with the static type pinned to ``DataFrame``."""
    result = df.loc[mask]
    assert isinstance(result, pd.DataFrame)
    return result


def _unique_str(df: pd.DataFrame, name: str) -> list[str]:
    return sorted({str(value) for value in _col(df, name)})


def _horizon_days(result: dict) -> int:
    """Forecast horizon in whole days, from whichever field is stored."""
    if "horizon_days" in result:
        return int(result["horizon_days"])
    return int(result["horizon_hours"]) // 24


def _r2_or_none(metrics: dict) -> float | None:
    value = metrics.get("r2")
    return None if value is None else float(value)


def create_summary_table(results: list[dict]) -> pd.DataFrame:
    """Create a summary table of model performance."""
    rows = []
    for result in results:
        r2 = _r2_or_none(result["metrics"])
        rows.append(
            {
                "Target": result["target"],
                "Horizon (days)": _horizon_days(result),
                "Model": result["model"].title(),
                "Split": result["split"].title(),
                "Rows": result["rows"],
                "MAE": f"{result['metrics']['mae']:.2f}",
                "RMSE": f"{result['metrics']['rmse']:.2f}",
                "R2": "n/a" if r2 is None else f"{r2:.4f}",
            }
        )
    return pd.DataFrame(rows)


def create_metric_pivot(results: list[dict], metric: str = "r2") -> pd.DataFrame:
    """Create a pivot table for a specific metric."""
    rows = [
        {
            "target": result["target"],
            "horizon": _horizon_days(result),
            "model": result["model"],
            "split": result["split"],
            metric: result["metrics"].get(metric),
        }
        for result in results
    ]
    return pd.DataFrame(rows).pivot_table(
        index=["target", "horizon"],
        columns=["model", "split"],
        values=metric,
        aggfunc="first",
    )


def format_markdown_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table (no extra deps)."""
    columns = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in df.itertuples(index=False):
        cells = ["" if pd.isna(value) else str(value) for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def validation_comparison_frame(results: list[dict], split: str = "validation") -> pd.DataFrame:
    """Tidy MAE/RMSE/R2 for every (target, horizon, model) on the given split."""
    rows = []
    for result in results:
        if result["split"] != split:
            continue
        metrics = result["metrics"]
        r2 = _r2_or_none(metrics)
        rows.append(
            {
                "target": result["target"],
                "label": TARGET_LABELS.get(result["target"], result["target"]),
                "horizon_days": _horizon_days(result),
                "model": result["model"],
                "mae": float(metrics["mae"]),
                "rmse": float(metrics["rmse"]),
                "r2": float("nan") if r2 is None else r2,
                "model_path": str(result.get("model_path", "")),
            }
        )
    return pd.DataFrame(
        rows,
        columns=["target", "label", "horizon_days", "model", "mae", "rmse", "r2", "model_path"],
    )


def _mean_metrics_by_model(subset: pd.DataFrame) -> pd.DataFrame:
    """Mean MAE/RMSE/R2 per model, ascending by RMSE."""
    rows = []
    for model in _unique_str(subset, "model"):
        model_rows = _rows(subset, _col(subset, "model").eq(model))
        rows.append(
            {
                "model": model,
                "mae": float(_col(model_rows, "mae").mean()),
                "rmse": float(_col(model_rows, "rmse").mean()),
                "r2": float(_col(model_rows, "r2").mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def select_best_models(frame: pd.DataFrame) -> dict:
    """Pick the lowest-RMSE model per (target, horizon) and overall per target."""
    best_per_horizon: dict[str, dict] = {}
    best_overall: dict[str, dict] = {}

    for target in _unique_str(frame, "target"):
        target_df = _rows(frame, _col(frame, "target").eq(target))

        per_horizon: dict[int, dict] = {}
        for horizon in sorted({int(value) for value in _col(target_df, "horizon_days")}):
            horizon_df = _rows(target_df, _col(target_df, "horizon_days").eq(horizon))
            winner = horizon_df.loc[_col(horizon_df, "rmse").idxmin()].to_dict()
            winner_r2 = winner["r2"]
            per_horizon[horizon] = {
                "model": str(winner["model"]),
                "rmse": round(float(winner["rmse"]), 4),
                "mae": round(float(winner["mae"]), 4),
                "r2": None if pd.isna(winner_r2) else round(float(winner_r2), 4),
                "model_path": str(winner["model_path"]),
            }
        best_per_horizon[target] = per_horizon

        ranked = _mean_metrics_by_model(target_df)
        top = ranked.iloc[0].to_dict()
        top_model = str(top["model"])
        paths = _col(_rows(target_df, _col(target_df, "model").eq(top_model)), "model_path")
        best_overall[target] = {
            "model": top_model,
            "mean_rmse": round(float(top["rmse"]), 4),
            "mean_mae": round(float(top["mae"]), 4),
            "model_paths": sorted(str(path) for path in paths),
        }

    return {"best_per_horizon": best_per_horizon, "best_overall": best_overall}


def write_best_model_metadata(
    selection: dict, split: str = "validation", path: Path = BEST_MODEL_METADATA_PATH
) -> Path:
    """Persist the best-model selection so downstream code can load one artifact."""
    metadata = {
        "selection_metric": "rmse",
        "split": split,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **selection,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def print_markdown_comparison(frame: pd.DataFrame, selection: dict) -> None:
    """Print one clean markdown table per target, models ranked by mean RMSE."""
    print("\n\n" + "=" * 100)
    print("VALIDATION MODEL COMPARISON (markdown, mean across horizons)")
    print("=" * 100)

    for target in _unique_str(frame, "target"):
        label = TARGET_LABELS.get(target, target)
        best_model = selection["best_overall"][target]["model"]
        ranked = _mean_metrics_by_model(_rows(frame, _col(frame, "target").eq(target)))

        display = pd.DataFrame(
            {
                "Model": [
                    f"**{name}**" if name == best_model else name
                    for name in _col(ranked, "model")
                ],
                "MAE": [f"{value:.3f}" for value in _col(ranked, "mae")],
                "RMSE": [f"{value:.3f}" for value in _col(ranked, "rmse")],
                "R2": [
                    "n/a" if pd.isna(value) else f"{value:.4f}"
                    for value in _col(ranked, "r2")
                ],
            }
        )

        print(f"\n### {label} (target: {target}) - best by RMSE: {best_model}\n")
        print(format_markdown_table(display))


def print_summary(results: list[dict]) -> None:
    """Print human-readable summary."""
    print("=" * 100)
    print("MODEL TRAINING SUMMARY")
    print("=" * 100)

    summary_table = create_summary_table(results)
    print(summary_table.to_string(index=False))
    print()

    # Group by target and model to show best performing horizons
    print("\nBEST PERFORMANCE BY TARGET & MODEL (R2 Score):")
    print("-" * 60)

    for target in _unique_str(summary_table, "Target"):
        print(f"\n{target}:")
        target_df = _rows(summary_table, _col(summary_table, "Target").eq(target))
        for model in _unique_str(target_df, "Model"):
            scored = _rows(
                target_df,
                _col(target_df, "Model").eq(model) & _col(target_df, "R2").ne("n/a"),
            )
            if scored.empty:
                continue
            best_row = scored.loc[_col(scored, "R2").astype(float).idxmax()].to_dict()
            print(
                f"  {model:20} -> Horizon {int(best_row['Horizon (days)']):3}d, "
                f"R2 = {best_row['R2']}"
            )

    # Overall statistics
    print("\n\nOVERALL STATISTICS:")
    print("-" * 60)
    r2_values = [
        r2 for r2 in (_r2_or_none(result["metrics"]) for result in results) if r2 is not None
    ]
    r2_series = pd.Series(r2_values, dtype="float64")

    print(f"Mean R2:   {r2_series.mean():.4f}")
    print(f"Std R2:    {r2_series.std():.4f}")
    print(f"Min R2:    {r2_series.min():.4f}")
    print(f"Max R2:    {r2_series.max():.4f}")
    print(f"Models Evaluated: {len(results)}")


def main() -> None:
    """Run analysis."""
    results = load_results()
    print_summary(results)

    # Load every trained model's validation metrics, compare them as markdown, and
    # lock in the lowest-RMSE model per target as the 'best_model'.
    frame = validation_comparison_frame(results)
    if frame.empty:
        print("\nNo validation results found; skipping best-model selection.")
    else:
        selection = select_best_models(frame)
        print_markdown_comparison(frame, selection)
        meta_path = write_best_model_metadata(selection)
        print(f"\n[ok] Saved best-model metadata to {meta_path}")
        for target, info in selection["best_overall"].items():
            label = TARGET_LABELS.get(target, target)
            print(f"  best {label}: {info['model']} (mean validation RMSE {info['mean_rmse']})")

    # Save metric pivots
    for metric in ["mae", "rmse", "r2"]:
        pivot = create_metric_pivot(results, metric)
        output = Path("models") / f"{metric}_pivot.csv"
        pivot.to_csv(output)
        print(f"\n[ok] Saved {metric} pivot to {output}")


if __name__ == "__main__":
    main()
