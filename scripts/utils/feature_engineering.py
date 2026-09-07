"""
Feature Engineering Module

Creates temporal, lag, rolling, solar geometry, and clear-sky features.
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Generate machine learning features from raw data"""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize feature engineer

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with timestamp index
        """
        self.df = df.copy()
        self.features_added = []

    def add_temporal_features(self) -> 'FeatureEngineer':
        """Add time-based features"""
        logger.info("Adding temporal features")

        times = self.df.index

        # Basic time components
        self.df['hour'] = times.hour
        self.df['day'] = times.day
        self.df['month'] = times.month
        self.df['quarter'] = times.quarter
        self.df['year'] = times.year
        self.df['week'] = times.isocalendar().week
        self.df['dayofweek'] = times.dayofweek  # 0=Monday, 6=Sunday
        self.df['dayofyear'] = times.dayofyear

        # Binary features
        self.df['is_weekend'] = (self.df['dayofweek'] >= 5).astype(int)

        # Cyclical encoding for hour and month (to capture circularity)
        self.df['hour_sin'] = np.sin(2 * np.pi * self.df['hour'] / 24)
        self.df['hour_cos'] = np.cos(2 * np.pi * self.df['hour'] / 24)

        self.df['month_sin'] = np.sin(2 * np.pi * self.df['month'] / 12)
        self.df['month_cos'] = np.cos(2 * np.pi * self.df['month'] / 12)

        # Season (0=Winter, 1=Spring, 2=Summer, 3=Fall)
        self.df['season'] = self.df['month'].map({
            12: 0, 1: 0, 2: 0,  # Winter
            3: 1, 4: 1, 5: 1,   # Spring
            6: 2, 7: 2, 8: 2,   # Summer
            9: 3, 10: 3, 11: 3  # Fall
        })

        # Time to/from midnight (useful for daily cycle patterns)
        self.df['hours_since_midnight'] = self.df['hour']
        self.df['hours_to_midnight'] = 24 - self.df['hour']

        self.features_added.extend([
            'hour', 'day', 'month', 'quarter', 'year', 'week', 'dayofweek', 'dayofyear',
            'is_weekend', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'season'
        ])

        temporal = [f for f in self.features_added if any(k in f for k in ("hour", "month", "day"))]
        logger.info("Added %d temporal features", len(temporal))

        return self

    def add_lag_features(self, columns: List[str], lags: List[int]) -> 'FeatureEngineer':
        """
        Add lag (shifted) features

        Parameters
        ----------
        columns : List[str]
            Columns to create lags for
        lags : List[int]
            Number of hours to lag (e.g., [1, 6, 12, 24])
        """
        logger.info("Adding lag features for %s columns with lags: %s", len(columns), lags)

        for col in columns:
            if col not in self.df.columns:
                logger.warning("Column %s not found, skipping", col)
                continue

            for lag in lags:
                lag_col = f"{col}_lag{lag}h"
                self.df[lag_col] = self.df[col].shift(lag)
                self.features_added.append(lag_col)

        logger.info("Added %d lag features", sum("lag" in f for f in self.features_added))

        return self

    def add_rolling_features(self, columns: List[str], windows: List[int],
                            stats: Optional[List[str]] = None) -> 'FeatureEngineer':
        """
        Add rolling window statistics

        Parameters
        ----------
        columns : List[str]
            Columns to compute rolling stats for
        windows : List[int]
            Window sizes in hours (e.g., [6, 12, 24])
        stats : List[str], optional
            Statistics to compute ('mean', 'std', 'min', 'max', 'median')
        """
        if stats is None:
            stats = ['mean', 'std', 'min', 'max']

        logger.info(
            "Adding rolling features for %d columns with windows: %s", len(columns), windows
        )

        for col in columns:
            if col not in self.df.columns:
                logger.warning("Column %s not found, skipping", col)
                continue

            for window in windows:
                rolling = self.df[col].rolling(window=window, min_periods=1)

                if 'mean' in stats:
                    self.df[f"{col}_rolling{window}h_mean"] = rolling.mean()
                    self.features_added.append(f"{col}_rolling{window}h_mean")

                if 'std' in stats:
                    self.df[f"{col}_rolling{window}h_std"] = rolling.std()
                    self.features_added.append(f"{col}_rolling{window}h_std")

                if 'min' in stats:
                    self.df[f"{col}_rolling{window}h_min"] = rolling.min()
                    self.features_added.append(f"{col}_rolling{window}h_min")

                if 'max' in stats:
                    self.df[f"{col}_rolling{window}h_max"] = rolling.max()
                    self.features_added.append(f"{col}_rolling{window}h_max")

        logger.info("Added %d rolling features", sum("rolling" in f for f in self.features_added))

        return self

    def add_difference_features(self, columns: List[str], diffs: List[int]) -> 'FeatureEngineer':
        """
        Add difference (delta) features

        Parameters
        ----------
        columns : List[str]
            Columns to compute differences for
        diffs : List[int]
            Number of steps back to compute difference (e.g., [1, 6, 24])
        """
        logger.info("Adding difference features")

        for col in columns:
            if col not in self.df.columns:
                logger.warning("Column %s not found, skipping", col)
                continue

            for diff in diffs:
                diff_col = f"{col}_diff{diff}h"
                self.df[diff_col] = self.df[col].diff(diff)
                self.features_added.append(diff_col)

        return self

    def add_interaction_features(
        self, feature_pairs: Optional[List[tuple]] = None
    ) -> "FeatureEngineer":
        """
        Add interaction features (product of two features)

        Parameters
        ----------
        feature_pairs : List[tuple], optional
            Pairs of column names to multiply
        """
        if feature_pairs is None:
            # Default: GHI interactions
            feature_pairs = [
                ('GHI', 'cloud_cover'),
                ('GHI', 'humidity'),
                ('temperature', 'humidity'),
            ]

        logger.info("Adding %s interaction features", len(feature_pairs))

        for col1, col2 in feature_pairs:
            if col1 not in self.df.columns or col2 not in self.df.columns:
                logger.warning("Columns %s or %s not found, skipping", col1, col2)
                continue

            interaction_col = f"{col1}_x_{col2}"
            self.df[interaction_col] = self.df[col1] * self.df[col2]
            self.features_added.append(interaction_col)

        return self

    def add_ratio_features(
        self, numerator: str, denominator: str, name: Optional[str] = None
    ) -> "FeatureEngineer":
        """
        Add ratio features (avoid division by zero)

        Parameters
        ----------
        numerator : str
            Numerator column
        denominator : str
            Denominator column
        name : str, optional
            Name of output column
        """
        if name is None:
            name = f"{numerator}_ratio_{denominator}"

        if numerator not in self.df.columns or denominator not in self.df.columns:
            logger.warning("Columns not found, skipping %s", name)
            return self

        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            self.df[name] = self.df[numerator] / self.df[denominator]
            self.df[name] = self.df[name].replace([np.inf, -np.inf], np.nan)

        self.features_added.append(name)

        return self

    def remove_nighttime_predictors(self, ghi_col: str = 'GHI',
                                   threshold: float = 10) -> 'FeatureEngineer':
        """
        Set certain predictors to 0/NaN during nighttime
        Nighttime defined as when GHI is very low

        Parameters
        ----------
        ghi_col : str
            GHI column name
        threshold : float
            GHI threshold for nighttime (W/m²)
        """
        logger.info("Setting nighttime predictors to NaN")

        nighttime = self.df[ghi_col] < threshold

        # Columns that should be NaN at night
        night_cols = [col for col in self.df.columns
                     if any(x in col for x in ['clear_sky', 'solar_elevation', 'clearness'])]

        for col in night_cols:
            if col in self.df.columns:
                self.df.loc[nighttime, col] = np.nan

        return self

    def handle_missing_values(
        self, method: str = "forward_fill", limit: int = 3
    ) -> "FeatureEngineer":
        """
        Handle missing values from lagging

        Parameters
        ----------
        method : str
            'forward_fill', 'backward_fill', or 'interpolate'
        limit : int
            Max number of consecutive fills
        """
        logger.info("Handling missing values using %s", method)

        initial_na = self.df.isna().sum().sum()

        if method == 'forward_fill':
            self.df = self.df.ffill(limit=limit)
        elif method == 'backward_fill':
            self.df = self.df.bfill(limit=limit)
        elif method == 'interpolate':
            self.df = self.df.interpolate(method='linear', limit=limit)

        final_na = self.df.isna().sum().sum()

        logger.info("Reduced missing values from %s to %s", initial_na, final_na)

        return self

    def get_dataframe(self) -> pd.DataFrame:
        """Get the engineered DataFrame"""
        return self.df

    def get_feature_list(self) -> List[str]:
        """Get list of engineered features"""
        return self.features_added

    def summary(self) -> str:
        """Print feature engineering summary"""
        summary = (
            f"Feature Engineering Summary\n"
            f"{'='*50}\n"
            f"Total rows: {len(self.df):,}\n"
            f"Total columns: {len(self.df.columns)}\n"
            f"Features engineered: {len(self.features_added)}\n"
            f"Missing values: {self.df.isna().sum().sum():,}\n"
        )

        return summary


# Convenience function
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply standard feature engineering pipeline

    Parameters
    ----------
    df : pd.DataFrame
        Input data with timestamp index

    Returns
    -------
    pd.DataFrame
        Engineered features DataFrame
    """
    fe = FeatureEngineer(df)

    # Add temporal features
    fe.add_temporal_features()

    # Add lag features for key variables
    fe.add_lag_features(['GHI', 'temperature', 'humidity'], lags=[1, 6, 12, 24])

    # Add rolling features
    fe.add_rolling_features(
        ['GHI', 'temperature', 'humidity'],
        windows=[6, 12, 24],
        stats=['mean', 'std']
    )

    # Handle missing values
    fe.handle_missing_values(method='forward_fill', limit=24)

    return fe.get_dataframe()
