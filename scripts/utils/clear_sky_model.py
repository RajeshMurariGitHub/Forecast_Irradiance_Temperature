"""
Clear-Sky Irradiance Models

Computes clear-sky GHI, DNI, DHI using Ineichen & Perez model.
Essential for creating clearness indices and understanding attenuation.
"""

import logging

import numpy as np
import pandas as pd

try:
    from pvlib import clearsky, location, solarposition
except ImportError:  # pragma: no cover - optional dependency
    clearsky = location = solarposition = None
    print("Warning: pvlib not installed. Install with: pip install pvlib")

logger = logging.getLogger(__name__)


class ClearSkyModel:
    """Calculate clear-sky irradiance"""

    def __init__(self, latitude: float, longitude: float, elevation: float = 0,
                 timezone: str = "UTC", model: str = "ineichen"):
        """
        Initialize clear-sky model

        Parameters
        ----------
        latitude : float
            Latitude in degrees
        longitude : float
            Longitude in degrees
        elevation : float, optional
            Elevation in meters
        timezone : str, optional
            Timezone
        model : str, optional
            Clear-sky model ('ineichen' or 'haurwitz')
        """
        self.latitude = latitude
        self.longitude = longitude
        self.elevation = elevation
        self.timezone = timezone
        self.model = model

        self.location = location.Location(
            latitude=latitude,
            longitude=longitude,
            altitude=elevation,
            tz=timezone
        )

        logger.info(
            "Initialized ClearSkyModel (%s) for (%.4f N, %.4f E)", model, latitude, longitude
        )

    def calculate_clearsky(
        self, times: pd.DatetimeIndex, linke_turbidity: float = 3.0
    ) -> pd.DataFrame:
        """
        Calculate clear-sky irradiance

        Parameters
        ----------
        times : pd.DatetimeIndex
            Times for calculation
        linke_turbidity : float, optional
            Linke turbidity factor (1-7, default 3.0 for clear skies)

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: cs_ghi, cs_dni, cs_dhi
        """
        logger.info(
            "Calculating clear-sky irradiance (%s model) for %d timestamps", self.model, len(times)
        )

        # pvlib assumes a tz-naive index is UTC; localize so the clear-sky curve
        # lines up with the actual local clock of the observations.
        times_aware = times if times.tz is not None else times.tz_localize(self.timezone)
        solar_pos = solarposition.get_solarposition(
            times_aware,
            latitude=self.latitude,
            longitude=self.longitude,
            altitude=self.elevation,
            method="nrel_numpy",
        )
        zenith = solar_pos["apparent_zenith"].to_numpy()

        if self.model == "ineichen":
            airmass = location.Location(
                latitude=self.latitude,
                longitude=self.longitude,
                altitude=self.elevation,
                tz=self.timezone,
            ).get_airmass(times=times_aware)["airmass_absolute"].to_numpy()
            cs = clearsky.ineichen(zenith, airmass, linke_turbidity, altitude=self.elevation)
        elif self.model == "haurwitz":
            cs = clearsky.haurwitz(zenith)
        else:
            raise ValueError(f"Unknown model: {self.model}")

        def _column(name: str) -> np.ndarray:
            # pvlib returns a dict of ndarrays for array input, a DataFrame for Series.
            values = np.asarray(cs[name]) if name in cs else np.zeros(len(times))
            return np.clip(values, 0.0, None)

        result = pd.DataFrame(index=times)
        result["clear_sky_ghi"] = _column("ghi")
        result["clear_sky_dni"] = _column("dni")
        result["clear_sky_dhi"] = _column("dhi")

        return result

    def calculate_clearness_indices(self, df: pd.DataFrame,
                                   ghi_col: str = 'GHI',
                                   cs_ghi_col: str = 'clear_sky_ghi') -> pd.DataFrame:
        """
        Calculate clearness indices (Kt, Kd)

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with GHI and clear-sky GHI
        ghi_col : str
            Name of GHI column
        cs_ghi_col : str
            Name of clear-sky GHI column

        Returns
        -------
        pd.DataFrame
            DataFrame with clearness indices
        """
        result = pd.DataFrame(index=df.index)

        # Clearness index (Kt) = GHI / clear-sky GHI
        # Only valid during daylight (clear_sky_ghi > 50 W/m²)
        daytime = df[cs_ghi_col] > 50

        result['clearness_index_kt'] = np.nan
        result.loc[daytime, 'clearness_index_kt'] = (
            df.loc[daytime, ghi_col] / df.loc[daytime, cs_ghi_col]
        ).clip(0, 1.2)  # Clip at 1.2 for overcast conditions

        # Diffuse fraction (Kd) = DHI / GHI
        if 'DHI' in df.columns:
            result['diffuse_fraction_kd'] = np.nan
            result.loc[daytime, 'diffuse_fraction_kd'] = (
                df.loc[daytime, 'DHI'] / df.loc[daytime, ghi_col]
            ).clip(0, 1)

        logger.info("Calculated clearness indices for %s daytime hours", daytime.sum())

        return result

    def add_clearsky_features(self, df: pd.DataFrame,
                             ghi_col: str = 'GHI',
                             linke_turbidity: float = 3.0) -> pd.DataFrame:
        """
        Add all clear-sky features to DataFrame

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with timestamp index
        ghi_col : str
            Name of GHI column
        linke_turbidity : float, optional
            Linke turbidity factor

        Returns
        -------
        pd.DataFrame
            DataFrame with clear-sky features added
        """
        logger.info("Adding clear-sky features")

        times = df.index

        # Calculate clear-sky irradiance
        cs = self.calculate_clearsky(times, linke_turbidity)
        df = pd.concat([df, cs], axis=1)

        # Calculate clearness indices
        indices = self.calculate_clearness_indices(df, ghi_col)
        df = pd.concat([df, indices], axis=1)

        # Calculate attenuation factor (actual / clear-sky)
        df['attenuation_factor'] = np.nan
        daytime = df['clear_sky_ghi'] > 50
        df.loc[daytime, 'attenuation_factor'] = (
            df.loc[daytime, ghi_col] / df.loc[daytime, 'clear_sky_ghi']
        ).clip(0, 1)

        return df


# Clearness classification based on Kt
def classify_sky_condition(kt: float) -> str:
    """
    Classify sky condition based on clearness index

    Parameters
    ----------
    kt : float
        Clearness index

    Returns
    -------
    str
        Sky condition: 'clear', 'partly_cloudy', 'overcast'
    """
    if kt < 0.3:
        return "overcast"
    if kt < 0.7:
        return "partly_cloudy"
    return "clear"


def add_sky_condition(df: pd.DataFrame, kt_col: str = "clearness_index_kt") -> pd.DataFrame:
    """Add sky condition classification"""
    df["sky_condition"] = df[kt_col].apply(
        lambda x: "night" if np.isnan(x) else classify_sky_condition(x)
    )
    return df


# Example usage
