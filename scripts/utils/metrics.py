"""
Evaluation Metrics Module

Standard ML and domain-specific metrics for solar forecasting
"""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


class SolarForecastMetrics:
    """Calculate metrics for solar irradiance and temperature forecasting"""

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error"""
        return mean_absolute_error(y_true, y_pred)

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error"""
        return np.sqrt(mean_squared_error(y_true, y_pred))

    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
        """
        Mean Absolute Percentage Error

        Parameters
        ----------
        y_true : np.ndarray
            True values
        y_pred : np.ndarray
            Predicted values
        epsilon : float
            Small value to avoid division by zero

        Returns
        -------
        float
            MAPE in percentage
        """
        mask = np.abs(y_true) > epsilon
        return 100 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

    @staticmethod
    def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """R² Score (coefficient of determination)"""
        return r2_score(y_true, y_pred)

    @staticmethod
    def mean_bias_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Bias Error (directional bias)"""
        return np.mean(y_pred - y_true)

    @staticmethod
    def nmae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Normalized Mean Absolute Error
        Normalized by mean of true values
        """
        return mean_absolute_error(y_true, y_pred) / np.mean(np.abs(y_true))

    @staticmethod
    def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Normalized Root Mean Squared Error
        Normalized by mean of true values
        """
        return np.sqrt(mean_squared_error(y_true, y_pred)) / np.mean(y_true)

    @staticmethod
    def mbe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Bias Error"""
        return np.mean(y_true - y_pred)

    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                             metric_names: list = None) -> dict:
        """
        Calculate all standard metrics

        Parameters
        ----------
        y_true : np.ndarray
            True values
        y_pred : np.ndarray
            Predicted values
        metric_names : list, optional
            Specific metrics to calculate. If None, calculates all.

        Returns
        -------
        dict
            Dictionary with metric names and values
        """
        if metric_names is None:
            metric_names = ['mae', 'rmse', 'mape', 'r2', 'mbe']

        metrics = {}

        for name in metric_names:
            try:
                if name == 'mae':
                    metrics[name] = SolarForecastMetrics.mae(y_true, y_pred)
                elif name == 'rmse':
                    metrics[name] = SolarForecastMetrics.rmse(y_true, y_pred)
                elif name == 'mape':
                    metrics[name] = SolarForecastMetrics.mape(y_true, y_pred)
                elif name == 'r2':
                    metrics[name] = SolarForecastMetrics.r2(y_true, y_pred)
                elif name == 'mbe':
                    metrics[name] = SolarForecastMetrics.mbe(y_true, y_pred)
                elif name == 'nmae':
                    metrics[name] = SolarForecastMetrics.nmae(y_true, y_pred)
                elif name == 'nrmse':
                    metrics[name] = SolarForecastMetrics.nrmse(y_true, y_pred)
            except (ValueError, TypeError, ZeroDivisionError, IndexError) as exc:
                logger.warning("Could not calculate %s: %s", name, exc)
                metrics[name] = np.nan

        return metrics


class HorizonEvaluator:
    """Evaluate predictions by forecast horizon"""

    def __init__(self, y_true: pd.DataFrame, y_pred: pd.DataFrame):
        """
        Initialize evaluator

        Parameters
        ----------
        y_true : pd.DataFrame
            True values with columns like 'horizon_1', 'horizon_2', etc.
        y_pred : pd.DataFrame
            Predicted values with same columns
        """
        self.y_true = y_true
        self.y_pred = y_pred

    def evaluate_by_horizon(self) -> pd.DataFrame:
        """
        Evaluate metrics for each forecast horizon

        Returns
        -------
        pd.DataFrame
            Metrics indexed by horizon
        """
        results = []

        for col in self.y_true.columns:
            if col.startswith('horizon_'):
                y_t = self.y_true[col].dropna()
                y_p = self.y_pred.loc[y_t.index, col]

                # Extract horizon number
                horizon = int(col.split('_')[1])

                metrics = SolarForecastMetrics.calculate_all_metrics(
                    y_t.values, y_p.values
                )
                metrics['horizon'] = horizon
                results.append(metrics)

        return pd.DataFrame(results).set_index('horizon').sort_index()

    def plot_horizon_metrics(self, metric: str = 'rmse', ax=None):
        """
        Plot metric across horizons

        Parameters
        ----------
        metric : str
            Metric name to plot
        ax : matplotlib axis, optional

        Returns
        -------
        matplotlib figure
        """
        # matplotlib is an optional, plotting-only dependency
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

        results = self.evaluate_by_horizon()

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        ax.plot(results.index, results[metric], marker='o', linewidth=2, markersize=8)
        ax.set_xlabel('Forecast Horizon (days)')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} by Forecast Horizon')
        ax.grid(True, alpha=0.3)

        return ax


# Convenience functions
def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Evaluate forecast with all metrics"""
    return SolarForecastMetrics.calculate_all_metrics(y_true, y_pred)


def print_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """Print formatted metrics"""
    metrics = SolarForecastMetrics.calculate_all_metrics(y_true, y_pred)

    print("\n" + "="*50)
    print("FORECAST EVALUATION METRICS")
    print("="*50)

    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name.upper():10s}: {value:>10.4f}")

    print("="*50 + "\n")
