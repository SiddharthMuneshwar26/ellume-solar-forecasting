"""Model features built from the forecast columns and the timestamp ONLY.

Never measured_*, rainfall_mm or status: they are not available at forecast time. The physics features come
from a PhysicsModel (B2b) whose parameters were fitted on the fold's training rows; at predict time that model
only reads timestamp + forecast columns.

No year or trend features: the test period (Jul-Aug 2025) is outside the training time range.
"""
import numpy as np
import pandas as pd
import pvlib

from src.solar import AZIMUTH, TILT, prepare_forecast

CLOUD_FEATURES = ["cloud", "cloud_s1", "cloud_s2", "cloud_day_mean"]
FEATURES = CLOUD_FEATURES + [
    "temp", "wind", "rh",
    "elevation", "azimuth", "cos_aoi",
    "cs_poa", "phys_poa", "cell_temp", "phys_power",
    "hour", "doy_sin", "doy_cos",
]
_FORBIDDEN = ("measured_", "rainfall", "status", "ac_power")


def _smooth_within_day(s, half_width):
    """Centred rolling mean over +/- half_width hours, never crossing midnight.

    The day-ahead forecast is stitched at midnight (audit A7), so neighbouring hours from another calendar
    day come from a different forecast issue. Computed on a full hourly grid so missing rows don't shift
    the window.
    """
    full = s.reindex(pd.date_range(s.index.min().normalize(),
                                   s.index.max().normalize() + pd.Timedelta(hours=23), freq="h"))
    sm = (full.groupby(full.index.normalize())
              .transform(lambda x: x.rolling(2 * half_width + 1, center=True, min_periods=1).mean()))
    return sm.reindex(s.index).values


def build_features(df, geo, phys):
    """Feature matrix (one row per input row, same order). `phys` = fitted PhysicsModel (B2b)."""
    fc = prepare_forecast(df)
    ts = pd.DatetimeIndex(fc["timestamp"])
    X = pd.DataFrame(index=range(len(fc)))
    X["cloud"] = fc["forecast_cloud_cover"].values
    X["temp"] = fc["forecast_temp_c"].values
    X["wind"] = fc["forecast_wind_ms"].values
    X["rh"] = fc["forecast_humidity_pct"].values
    cloud = pd.Series(X["cloud"].values, index=ts)
    X["cloud_s1"] = _smooth_within_day(cloud, 1)
    X["cloud_s2"] = _smooth_within_day(cloud, 2)
    day = ts.normalize()
    daylight = geo["elevation"].values > 0
    day_mean = cloud[daylight].groupby(day[daylight]).mean()
    X["cloud_day_mean"] = pd.Series(day).map(day_mean).values

    X["elevation"] = geo["elevation"].values
    X["azimuth"] = geo["azimuth"].values
    aoi = pvlib.irradiance.aoi(TILT, AZIMUTH, geo["apparent_zenith"].values, geo["azimuth"].values)
    X["cos_aoi"] = np.cos(np.radians(np.asarray(aoi)))

    comp = phys.components(geo, fc)
    for c in ["cs_poa", "phys_poa", "cell_temp", "phys_power"]:
        X[c] = comp[c].values

    X["hour"] = ts.hour
    doy = ts.dayofyear
    X["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    X["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    assert not any(c.startswith(_FORBIDDEN) for c in X.columns)
    assert X[FEATURES].notna().all().all(), "NaN in features"
    return X[FEATURES]
