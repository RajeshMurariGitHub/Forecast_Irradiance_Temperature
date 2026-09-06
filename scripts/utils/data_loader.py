"""
Data Loading and Processing Utilities
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def _split_end_exclusive(end: str) -> pd.Timestamp:
    """Configured split ``end`` dates are inclusive; return the exclusive bound."""
    return pd.Timestamp(end).normalize() + pd.Timedelta(days=1)


class DataLoader:
    """Load and manage datasets"""

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize data loader

        Parameters
        ----------
        config_path : str
            Path to configuration file
        """
        with open(config_path, "r", encoding="utf-8") as fh:
            self.config = yaml.safe_load(fh)

        self.data_config = self.config["data"]
        self.split_config = self.config["data_split"]

    def _read_csv(self, filepath: str | Path) -> pd.DataFrame:
        df = pd.read_csv(filepath, index_col=0)
        df.index = pd.to_datetime(df.index)
        df.index.name = "timestamp"
        return df.sort_index()

    def load_raw_data(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Load raw NASA POWER data

        Parameters
        ----------
        filepath : str, optional
            Path to raw CSV file. If None, uses config.

        Returns
        -------
        pd.DataFrame
            Raw data with timestamp index
        """
        path = filepath or Path(self.data_config["raw_dir"]) / self.data_config["raw_filename"]

        logger.info("Loading raw data from %s", path)
        df = self._read_csv(path)
        logger.info("Loaded %d rows with columns: %s", len(df), list(df.columns))
        return df

    def load_processed_data(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Load cleaned and processed data

        Parameters
        ----------
        filepath : str, optional
            Path to processed CSV. If None, uses config.

        Returns
        -------
        pd.DataFrame
            Processed data with timestamp index
        """
        path = filepath or (
            Path(self.data_config["processed_dir"]) / self.data_config["processed_filename"]
        )

        logger.info("Loading processed data from %s", path)
        df = self._read_csv(path)
        logger.info("Loaded %d rows with %d features", len(df), len(df.columns))
        return df

    def save_processed_data(self, df: pd.DataFrame, filepath: Optional[str] = None) -> None:
        """
        Save processed data to CSV

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to save
        filepath : str, optional
            Output path. If None, uses config.
        """
        path = Path(
            filepath
            or Path(self.data_config["processed_dir"]) / self.data_config["processed_filename"]
        )

        path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Saving processed data to %s", path)
        df.to_csv(path)
        logger.info("Saved %d rows", len(df))

    def split_by_period(self, df: pd.DataFrame) -> dict:
        """
        Split data by training/testing/validation/prospective periods.

        The configured ``end`` date is treated as inclusive (the whole day), which
        matches ``scripts.train_models.get_split_bounds`` and the API.

        Parameters
        ----------
        df : pd.DataFrame
            Full dataset with timestamp index

        Returns
        -------
        dict
            Keys: 'train', 'test', 'validation', 'prospective'; each value is a
            DataFrame for that period.
        """
        key_aliases = {
            "training": "train",
            "testing": "test",
            "validation": "validation",
            "prospective": "prospective",
        }

        splits = {}
        for config_key, out_key in key_aliases.items():
            period = self.split_config[config_key]
            start = pd.Timestamp(period["start"])
            end_exclusive = _split_end_exclusive(period["end"])

            mask = (df.index >= start) & (df.index < end_exclusive)
            splits[out_key] = df.loc[mask].copy()

            logger.info(
                "%s: %s rows (%s -> %s)",
                period.get("label", config_key),
                f"{len(splits[out_key]):,}",
                start,
                end_exclusive,
            )

        return splits

    _FREQ_DELTAS = {
        "h": pd.Timedelta(hours=1),
        "H": pd.Timedelta(hours=1),
        "D": pd.Timedelta(days=1),
    }

    def validate_continuity(self, df: pd.DataFrame, freq: str = "h") -> Tuple[int, pd.Series]:
        """
        Check for missing timestamps.

        Parameters
        ----------
        df : pd.DataFrame
            Data with timestamp index
        freq : str
            Expected sampling frequency: 'h' (hourly) or 'D' (daily).

        Returns
        -------
        tuple
            (count of gaps, Series of unexpected intervals)
        """
        try:
            expected = self._FREQ_DELTAS[freq]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported freq {freq!r}; expected one of {list(self._FREQ_DELTAS)}"
            ) from exc

        time_diffs = df.index.to_series().diff()
        gaps = time_diffs[time_diffs != expected].iloc[1:]  # Skip the leading NaT

        logger.info("Found %d time gaps (expected frequency: %s)", len(gaps), expected)
        return len(gaps), gaps

    def get_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get summary statistics"""
        return df.describe()


def load_raw_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """Convenience function to load raw data"""
    return DataLoader().load_raw_data(filepath)


def load_processed_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """Convenience function to load processed data"""
    return DataLoader().load_processed_data(filepath)
