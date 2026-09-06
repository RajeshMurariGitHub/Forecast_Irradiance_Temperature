"""Generate comprehensive model comparison report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from analyze_models import (
    TARGET_LABELS,
    format_markdown_table,
    print_markdown_comparison,
    select_best_models,
    validation_comparison_frame,
    write_best_model_metadata,
)

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


def load_results(results_file: str = "models/training_results.json") -> list[dict]:
    """Load training results from JSON."""
    with open(results_file, "r", encoding="utf-8") as fh:
        return json.load(fh)


def create_comparison_table(results: list[dict]) -> pd.DataFrame:
    """Create a comprehensive comparison table grouped by split."""
    rows = []
    for result in results:
        row = {
            "Target": result["target"],
            "Horizon": result.get("horizon_days", result["horizon_hours"]),
            "Model": result["model"].replace("_", " ").title(),
            "Split": result["split"].title(),
            "Rows": result["rows"],
            "MAE": result["metrics"]["mae"],
            "RMSE": result["metrics"]["rmse"],
            "R²": result["metrics"]["r2"],
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


def print_comparison_by_split(results: list[dict]) -> None:
    """Print model comparison organized by split and target."""
    df = create_comparison_table(results)
    
    print("=" * 140)
    print("COMPREHENSIVE MODEL EVALUATION REPORT")
    print("=" * 140)
    
    for split in sorted(df["Split"].unique()):
        print(f"\n{'=' * 140}")
        print(f"SPLIT: {split}")
        print(f"{'=' * 140}\n")
        
        split_df = df[df["Split"] == split].copy()
        
        for target in sorted(split_df["Target"].unique()):
            target_df = split_df[split_df["Target"] == target].sort_values("Horizon")
            
            print(f"\nTarget: {target}")
            print("-" * 140)
            
            # Format output nicely
            display_df = target_df.copy()
            display_df["MAE"] = display_df["MAE"].apply(lambda x: f"{x:.2f}")
            display_df["RMSE"] = display_df["RMSE"].apply(lambda x: f"{x:.2f}")
            display_df["R²"] = display_df["R²"].apply(
                lambda x: "n/a" if x is None or pd.isna(x) else f"{x:.4f}"
            )
            
            print(display_df[["Horizon", "Model", "Rows", "MAE", "RMSE", "R²"]].to_string(index=False))


def print_model_ranking(results: list[dict]) -> None:
    """Print best performing models by target and horizon."""
    df = create_comparison_table(results)
    
    print("\n\n" + "=" * 140)
    print("BEST MODEL BY TARGET & HORIZON (Validation Split)")
    print("=" * 140)
    
    validation_df = df[df["Split"] == "Validation"].copy()
    
    for target in sorted(validation_df["Target"].unique()):
        print(f"\n{target}:")
        print("-" * 60)
        
        target_df = validation_df[validation_df["Target"] == target].copy()
        
        for horizon in sorted(target_df["Horizon"].unique()):
            horizon_df = target_df[target_df["Horizon"] == horizon].copy()
            best_row = horizon_df.loc[horizon_df["R²"].idxmax()]
            
            model = best_row["Model"]
            r2 = best_row["R²"]
            mae = best_row["MAE"]
            rmse = best_row["RMSE"]
            
            print(f"  {horizon:3}d -> {model:20} | R² = {r2:.4f} | MAE = {mae:.2f} | RMSE = {rmse:.2f}")


def print_generalization_gap(results: list[dict]) -> None:
    """Compare validation and testing performance."""
    df = create_comparison_table(results)
    
    # Pivot to compare splits
    val_df = df[df["Split"] == "Validation"].set_index(["Target", "Horizon", "Model"])
    test_df = df[df["Split"] == "Testing"].set_index(["Target", "Horizon", "Model"])
    
    print("\n\n" + "=" * 140)
    print("GENERALIZATION GAP ANALYSIS (Validation vs Testing - R² Score)")
    print("=" * 140)
    print()
    
    rows = []
    for idx in sorted(val_df.index):
        if idx in test_df.index:
            val_r2 = val_df.loc[idx, "R²"]
            test_r2 = test_df.loc[idx, "R²"]
            gap = val_r2 - test_r2
            
            target, horizon, model = idx
            rows.append({
                "Target": target,
                "Horizon": horizon,
                "Model": model,
                "Val R²": f"{val_r2:.4f}",
                "Test R²": f"{test_r2:.4f}",
                "Gap": f"{gap:.4f}",
                "Status": "✓ Good" if gap < 0.01 else "⚠ Moderate" if gap < 0.05 else "✗ High"
            })
    
    if rows:
        result_df = pd.DataFrame(rows)
        print(result_df.to_string(index=False))
        
        # Summary statistics
        gap_values = [float(r["Gap"]) for r in rows]
        print(f"\nMean generalization gap: {pd.Series(gap_values).mean():.4f}")
        print(f"Max generalization gap:  {pd.Series(gap_values).max():.4f}")
    else:
        print("No testing results available yet for comparison.")


def print_validation_markdown(results: list[dict]) -> None:
    """Print markdown RMSE matrices (horizon x model) for both targets."""
    frame = validation_comparison_frame(results)
    if frame.empty:
        print("\nNo validation results available for the markdown comparison.")
        return

    print("\n\n" + "=" * 140)
    print("VALIDATION RMSE BY MODEL (markdown)")
    print("=" * 140)

    for target, target_df in frame.groupby("target"):
        label = TARGET_LABELS.get(str(target), str(target))
        matrix = (
            target_df.pivot_table(index="horizon_days", columns="model", values="rmse")
            .round(2)
            .reset_index()
            .rename(columns={"horizon_days": "Horizon (d)"})
        )
        matrix["Horizon (d)"] = matrix["Horizon (d)"].astype(int)
        print(f"\n### {label} (target: {target})\n")
        print(format_markdown_table(matrix))


def select_and_save_best_model(results: list[dict]) -> None:
    """Select the lowest-RMSE model per target and write the metadata JSON."""
    frame = validation_comparison_frame(results)
    if frame.empty:
        return
    selection = select_best_models(frame)
    print_markdown_comparison(frame, selection)
    meta_path = write_best_model_metadata(selection)
    print(f"\n✓ Saved best-model metadata to {meta_path}")
    for target, info in selection["best_overall"].items():
        label = TARGET_LABELS.get(target, target)
        print(f"  best {label}: {info['model']} (mean validation RMSE {info['mean_rmse']})")


def main() -> None:
    """Generate full report."""
    results = load_results()
    print_comparison_by_split(results)
    print_model_ranking(results)
    print_generalization_gap(results)
    print_validation_markdown(results)
    select_and_save_best_model(results)


if __name__ == "__main__":
    main()
