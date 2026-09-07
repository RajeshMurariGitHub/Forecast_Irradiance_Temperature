"""Streamlit dashboard for the Solar Irradiance & Temperature Forecast project."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")
BEST_MODELS_PATH = Path(__file__).resolve().parent.parent / "models" / "best_models.json"

st.set_page_config(
    page_title="Solar Forecast Dashboard",
    page_icon="☀️",
    layout="wide",
)


@st.cache_data(ttl=300)
def api_get(path: str, params: dict | None = None):
    response = requests.get(f"{API_URL}{path}", params=params, timeout=30)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=300)
def get_config():
    return api_get("/api/config")


@st.cache_data(ttl=300)
def load_best_models() -> dict:
    """Read the evaluation report's best-model selection (models/best_models.json)."""
    try:
        return json.loads(BEST_MODELS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def main() -> None:
    st.title("☀️ Solar Irradiance & Temperature Forecast Dashboard")

    try:
        config = get_config()
    except requests.exceptions.RequestException as exc:
        st.error(f"Could not reach the API at {API_URL}: {exc}")
        st.stop()

    location = config["location"]
    st.caption(
        f"{location['name']}, {location['country']} "
        f"({location['latitude']}, {location['longitude']}) — "
        f"forecasting {', '.join(config['targets'])} for horizons {config['horizons']}h"
    )

    targets = config["targets"]
    horizons = config["horizons"]

    st.sidebar.header("Filters")
    split = st.sidebar.selectbox("Split", options=["validation", "testing"], index=0)
    target = st.sidebar.selectbox("Target", options=targets)
    horizon = st.sidebar.selectbox("Horizon (hours)", options=horizons)

    tab_production, tab_overview, tab_forecast, tab_importance, tab_gap = st.tabs(
        [
            "🚀 Production Forecast",
            "📊 Model Comparison",
            "📈 Forecast Explorer",
            "🔍 Feature Importance",
            "🧪 Generalization Gap",
        ]
    )

    with tab_production:
        render_production(target, horizon)

    with tab_overview:
        render_overview(split, target, horizon)

    with tab_forecast:
        render_forecast(target, horizon)

    with tab_importance:
        render_importance(target, horizon)

    with tab_gap:
        render_generalization_gap(target)


def render_winning_model_badge(target: str, horizon: int) -> None:
    """Metric badge showcasing the winning model and its validation RMSE."""
    best = load_best_models()
    per_horizon = best.get("best_per_horizon", {}).get(target, {})
    pick = per_horizon.get(str(horizon // 24)) or best.get("best_overall", {}).get(target)
    if not pick:
        return

    b1, b2, b3 = st.columns(3)
    b1.metric("Winning model", str(pick["model"]).replace("_", " ").title())
    rmse = pick.get("rmse", pick.get("mean_rmse"))
    b2.metric("Validation RMSE", "n/a" if rmse is None else f"{rmse:.3f}")
    b3.metric("Selection metric", best.get("selection_metric", "rmse").upper())


def render_production(target: str, horizon: int) -> None:
    st.subheader(f"Production forecast — {target}, {horizon}h horizon (May 2026)")
    render_winning_model_badge(target, horizon)

    try:
        data = api_get(
            "/api/forecast/production",
            params={"target": target, "horizon_hours": horizon, "limit": 744},
        )
    except requests.exceptions.HTTPError as exc:
        st.warning(f"No production forecast available for this combination: {exc}")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["timestamps"], y=data["actual"], name="Actual", mode="lines"))
    fig.add_trace(
        go.Scatter(x=data["timestamps"], y=data["predicted"], name="Predicted", mode="lines")
    )
    fig.update_layout(
        title=(
            f"{target} — May 2026 hourly forecast "
            f"({str(data['model']).replace('_', ' ').title()}, {horizon}h ahead)"
        ),
        xaxis_title="Timestamp",
        yaxis_title=target,
        legend_title="Series",
    )
    st.plotly_chart(fig, width="stretch")

    metrics = data.get("metrics") or {}
    m1, m2, m3 = st.columns(3)
    m1.metric("Served model", str(data["model"]).replace("_", " ").title())
    m2.metric("MAE (vs observed)", "n/a" if "mae" not in metrics else f"{metrics['mae']:.3f}")
    m3.metric(
        "Hours forecast",
        f"{len(data['predicted'])}",
        delta=None if "n_observed" not in metrics else f"{metrics['n_observed']} observed",
    )


def render_overview(split: str, target: str, horizon: int) -> None:
    st.subheader(f"Model comparison — {target}, {horizon}h horizon, {split} split")

    results = api_get(
        "/api/results", params={"target": target, "split": split, "horizon_hours": horizon}
    )
    if not results:
        st.warning("No results found for this combination.")
        return

    rows = [
        {
            "Model": r["model"].replace("_", " ").title(),
            "MAE": r["metrics"]["mae"],
            "RMSE": r["metrics"]["rmse"],
            "R²": r["metrics"]["r2"],
            "Rows": r["rows"],
        }
        for r in results
    ]
    df = pd.DataFrame(rows).sort_values("R²", ascending=False)
    st.dataframe(df, width="stretch", hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(df, x="Model", y="R²", color="Model", title="R² by model")
        st.plotly_chart(fig, width="stretch")
    with col2:
        fig = px.bar(df, x="Model", y="MAE", color="Model", title="MAE by model")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Best model per horizon (validation split)")
    best = api_get("/api/best-models", params={"split": "validation"})
    best_df = pd.DataFrame(
        [
            {
                "Target": b["target"],
                "Horizon (h)": b["horizon_hours"],
                "Best Model": b["model"].replace("_", " ").title(),
                "R²": b["metrics"]["r2"],
                "MAE": b["metrics"]["mae"],
                "RMSE": b["metrics"]["rmse"],
            }
            for b in best
        ]
    )
    st.dataframe(best_df, width="stretch", hide_index=True)


def render_forecast(target: str, horizon: int) -> None:
    st.subheader(f"Forecast explorer — {target}, {horizon}h horizon")

    col1, col2, col3 = st.columns(3)
    with col1:
        model = st.selectbox(
            "Model",
            options=[
                "ridge", "random_forest", "extra_trees",
                "gradient_boost", "xgboost", "lightgbm", "catboost",
            ],
            key="forecast_model",
        )
    with col2:
        split = st.selectbox("Split", options=["testing", "validation"], key="forecast_split")
    with col3:
        limit = st.slider("Points to show", min_value=24, max_value=1000, value=200, step=24)

    try:
        data = api_get(
            "/api/forecast",
            params={
                "target": target, "horizon_hours": horizon,
                "model": model, "split": split, "limit": limit,
            },
        )
    except requests.exceptions.HTTPError as exc:
        st.warning(f"No forecast available for this combination: {exc}")
        return

    pretty_model = model.replace("_", " ").title()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["timestamps"], y=data["actual"], name="Actual", mode="lines"))
    fig.add_trace(
        go.Scatter(x=data["timestamps"], y=data["predicted"], name="Predicted", mode="lines")
    )
    fig.update_layout(
        title=f"{target} — Actual vs Predicted ({pretty_model}, {horizon}h ahead)",
        xaxis_title="Timestamp",
        yaxis_title=target,
        legend_title="Series",
    )
    st.plotly_chart(fig, width="stretch")

    metrics = data["metrics"]
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{metrics['mae']:.3f}")
    m2.metric("RMSE", f"{metrics['rmse']:.3f}")
    m3.metric("R²", "n/a" if metrics.get("r2") is None else f"{metrics['r2']:.4f}")


def render_importance(target: str, horizon: int) -> None:
    st.subheader(f"Feature importance — {target}, {horizon}h horizon")

    model = st.selectbox(
        "Model",
        options=[
            "random_forest", "extra_trees", "gradient_boost",
            "xgboost", "lightgbm", "catboost", "ridge",
        ],
        key="importance_model",
    )
    top_n = st.slider("Top N features", min_value=5, max_value=30, value=10)

    try:
        data = api_get(
            "/api/feature-importance",
            params={"target": target, "horizon_hours": horizon, "model": model, "top_n": top_n},
        )
    except requests.exceptions.HTTPError:
        st.warning("No feature importance available for this combination.")
        return

    df = pd.DataFrame(data).sort_values("Importance", ascending=True)
    fig = px.bar(df, x="Importance", y="Feature", orientation="h", title=f"Top {top_n} features")
    st.plotly_chart(fig, width="stretch")


def render_generalization_gap(target: str) -> None:
    st.subheader(f"Generalization gap — {target} (Validation vs Testing R²)")

    data = api_get("/api/generalization-gap")
    df = pd.DataFrame([d for d in data if d["target"] == target])
    if df.empty:
        st.warning("No generalization gap data available.")
        return

    df["Model"] = df["model"].str.replace("_", " ").str.title()
    fig = px.bar(
        df,
        x="horizon_hours",
        y="gap",
        color="Model",
        barmode="group",
        title="Validation → Testing R² gap by horizon",
        labels={"horizon_hours": "Horizon (h)", "gap": "R² Gap"},
    )
    st.plotly_chart(fig, width="stretch")

    display_df = df[["Model", "horizon_hours", "val_r2", "test_r2", "gap"]].rename(
        columns={
            "horizon_hours": "Horizon (h)",
            "val_r2": "Val R²",
            "test_r2": "Test R²",
            "gap": "Gap",
        }
    )
    st.dataframe(display_df.sort_values(["Horizon (h)", "Gap"]), width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
