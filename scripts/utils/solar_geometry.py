"""
Solar Geometry Calculations

Computes solar position (elevation, azimuth, zenith) using pvlib and ephem libraries.
Critical for understanding GHI patterns throughout the year.
"""

import logging

import numpy as np
import pandas as pd

try:
    from pvlib import location, solarposition
except ImportError:  # pragma: no cover - optional dependency
    location = solarposition = None
    print("Warning: pvlib not installed. Install with: pip install pvlib")

logger = logging.getLogger(__name__)


class SolarGeometry:
    """Calculate solar position and geometry features"""

    def __init__(
        self, latitude: float, longitude: float, elevation: float = 0, timezone: str = "UTC"
    ):
        """
        Initialize solar geometry calculator

        Parameters
        ----------
        latitude : float
            Latitude in degrees (positive North)
        longitude : float
            Longitude in degrees (positive East)
        elevation : float, optional
            Elevation in meters (default: 0)
        timezone : str, optional
            Timezone string (default: UTC)
        """
        self.latitude = latitude
        self.longitude = longitude
        self.elevation = elevation
        self.timezone = timezone

        # Create location object
        self.location = location.Location(
            latitude=latitude, longitude=longitude, altitude=elevation, tz=timezone
        )

        logger.info(
            "Initialized SolarGeometry for (%.4f N, %.4f E), elevation=%sm, tz=%s",
            latitude, longitude, elevation, timezone,
        )

    def calculate_solar_position(self, times: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Calculate solar position at given times

        Parameters
        ----------
        times : pd.DatetimeIndex
            Times for calculation

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: elevation, azimuth, zenith
        """
        logger.info("Calculating solar position for %d timestamps", len(times))

        # pvlib assumes a tz-naive index is UTC; localize to the caller's tz so
        # solar position matches the actual local clock of the observations.
        times_aware = times if times.tz is not None else times.tz_localize(self.timezone)
        solar_pos = solarposition.get_solarposition(
            times_aware,
            self.latitude,
            self.longitude,
            self.elevation,
            method="nrel_numpy",
            temperature=25,
        )

        # Return with the caller's original (naive) index; align by position.
        result = pd.DataFrame(index=times)
        result["solar_elevation"] = solar_pos["elevation"].to_numpy()
        result["solar_azimuth"] = solar_pos["azimuth"].to_numpy()
        result["solar_zenith"] = solar_pos["apparent_zenith"].to_numpy()

        return result

    def calculate_declination(self, times: pd.DatetimeIndex) -> np.ndarray:
        """
        Calculate solar declination (latitude of sub-solar point)

        Parameters
        ----------
        times : pd.DatetimeIndex
            Times for calculation

        Returns
        -------
        np.ndarray
            Declination in degrees
        """
        # Day of year (1-366)
        day_of_year = times.dayofyear.values

        # Declination in radians (Duffie & Beckman equation)
        declination_rad = 23.45 * np.pi / 180 * np.sin(2 * np.pi * (day_of_year - 81) / 365)

        return np.degrees(declination_rad)

    def calculate_hour_angle(self, times: pd.DatetimeIndex) -> np.ndarray:
        """
        Calculate solar hour angle

        Parameters
        ----------
        times : pd.DatetimeIndex
            Times for calculation

        Returns
        -------
        np.ndarray
            Hour angle in degrees (15° per hour)
        """
        # Solar time (local time, assuming solar time = local time for simplicity)
        solar_hours = times.hour + times.minute / 60

        # Hour angle = 15° per hour, 0° at solar noon (12:00)
        hour_angle = 15 * (solar_hours - 12)

        return hour_angle

    def calculate_airmass(self, elevation: np.ndarray) -> np.ndarray:
        """
        Calculate airmass (Kasten-Czeplak model)

        Parameters
        ----------
        elevation : np.ndarray
            Solar elevation in degrees

        Returns
        -------
        np.ndarray
            Airmass (1.0 at zenith, increases at horizon)
        """
        # Clip elevation to avoid issues at/below horizon
        elev = np.maximum(elevation, 0)

        # Kasten-Czeplak airmass model
        zenith_rad = np.radians(90 - elev)
        airmass = 1 / (
            np.cos(zenith_rad) + 0.50572 * (96.07995 - np.degrees(zenith_rad)) ** (-1.6364)
        )

        return airmass

    def add_solar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all solar geometry features to DataFrame

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with timestamp index

        Returns
        -------
        pd.DataFrame
            DataFrame with solar geometry features added
        """
        logger.info("Adding solar geometry features")

        times = df.index

        # Solar position
        solar_pos = self.calculate_solar_position(times)
        df = pd.concat([df, solar_pos], axis=1)

        # Additional features
        df["solar_declination"] = self.calculate_declination(times)
        df["solar_hour_angle"] = self.calculate_hour_angle(times)
        df["solar_airmass"] = self.calculate_airmass(df["solar_elevation"].values)

        # Daylight flag (elevation > 0)
        df["is_daylight"] = df["solar_elevation"] > 0

        # Sunrise/sunset flags
        df["is_sunrise"] = (df["solar_elevation"].shift(1) <= 0) & (df["solar_elevation"] > 0)
        df["is_sunset"] = (df["solar_elevation"].shift(1) > 0) & (df["solar_elevation"] <= 0)
        df["is_sunrise"] = df["is_sunrise"].fillna(False)
        df["is_sunset"] = df["is_sunset"].fillna(False)

        logger.info("Added solar features. Daylight hours: %s", df['is_daylight'].sum())

        return df


def get_solar_geometry(
    latitude: float, longitude: float, elevation: float = 0, timezone: str = "UTC"
) -> SolarGeometry:
    """Create SolarGeometry instance"""
    return SolarGeometry(latitude, longitude, elevation, timezone)


# Example usage
