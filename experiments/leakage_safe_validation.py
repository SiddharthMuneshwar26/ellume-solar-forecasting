"""Chronological leakage-safe validation of the frozen production approach.

This experiment is intentionally isolated from run.py and the production cleaning,
feature, and training entry points. Solar geometry and deterministic pvlib building
blocks are reused from src.solar; all fitted parameters and preprocessing are fold-local.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pvlib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import FEATURES
from src.solar import (
    AC_KW,
    AZIMUTH,
    DC_KWP,
    FORECAST_COLS,
    MONSOON_MONTHS,
    TEMP_COEFF,
    TILT,
    allsky_poa,
    clearsky,
    cloud_factor,
    fit_cloud_curve,
    fit_linke,
    geometry,
)


DATA = ROOT / "data"
RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

SEED = 42
M2_SEEDS = (42, 43, 44, 45, 46)
M2_LEAVES = 15
M2_ROUNDS = 295
MIN_PHYS_KW = 200.0
TL_MODE = "fitted_dry"

BASE_PARAMS = dict(
    objective="regression",
    learning_rate=0.03,
    min_child_samples=20,
    feature_fraction=0.9,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    seed=SEED,
    deterministic=True,
    force_row_wise=True,
    num_threads=4,
    verbose=-1,
)

MW_START = pd.Timestamp("2024-07-14 06:00")
MW_END = pd.Timestamp("2024-08-08 18:00")
POWER_FLAT_START = pd.Timestamp("2024-09-18 00:00")
POWER_FLAT_END = pd.Timestamp("2024-09-19 23:00")
POWER_FLAT_VALUE = 2187.44
POA_FLAT_START = pd.Timestamp("2024-08-25 00:00")
POA_FLAT_END = pd.Timestamp("2024-08-27 23:00")
POA_FLAT_VALUE = 412.6
SENTINELS = (-999.0, 9999.0)


@dataclass(frozen=True)
class Fold:
    name: str
    train_end_exclusive: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end_inclusive: pd.Timestamp


FOLDS = (
    Fold("F1", pd.Timestamp("2024-07-01"), pd.Timestamp("2024-07-01"), pd.Timestamp("2024-08-31 23:00")),
    Fold("F2", pd.Timestamp("2024-09-01"), pd.Timestamp("2024-09-01"), pd.Timestamp("2024-10-31 23:00")),
    Fold("F3", pd.Timestamp("2024-11-01"), pd.Timestamp("2024-11-01"), pd.Timestamp("2024-12-31 23:00")),
    Fold("F4", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-01"), pd.Timestamp("2025-02-28 23:00")),
    Fold("F5", pd.Timestamp("2025-03-01"), pd.Timestamp("2025-03-01"), pd.Timestamp("2025-04-30 23:00")),
    Fold("F6", pd.Timestamp("2025-05-01"), pd.Timestamp("2025-05-01"), pd.Timestamp("2025-06-30 23:00")),
)


def parse_timestamps(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    iso = pd.to_datetime(out["timestamp"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    dmy = pd.to_datetime(out["timestamp"], format="%d-%m-%Y %H:%M", errors="coerce")
    out["timestamp"] = iso.fillna(dmy)
    assert out["timestamp"].notna().all(), "unparseable training timestamps"
    return out.sort_values("timestamp", kind="stable").reset_index(drop=True)


def replace_sentinels(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    numeric = out.select_dtypes(include=[np.number]).columns
    out[numeric] = out[numeric].replace(list(SENTINELS), np.nan)
    return out


def expected_measured_power(frame: pd.DataFrame) -> pd.Series:
    tcorr = 1 + TEMP_COEFF * (frame["measured_module_temp_c"] - 25)
    return (10.8 * frame["measured_poa_wm2"] * tcorr).clip(upper=AC_KW)


def resolve_training_duplicates(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the documented duplicate rule only to the already cutoff training slice."""
    if not frame["timestamp"].duplicated(keep=False).any():
        return frame.reset_index(drop=True)
    out = frame.copy()
    duplicate_mask = out["timestamp"].duplicated(keep=False)
    geo = geometry(out.loc[duplicate_mask, "timestamp"])
    keep_indices: list[int] = []
    other_cols = [c for c in out.columns if c not in ("timestamp", "ac_power_kw")]
    for ts, group in out.loc[duplicate_mask].groupby("timestamp", sort=True):
        if group[other_cols].nunique(dropna=False).max() > 1:
            raise ValueError(f"duplicate rows disagree outside target at {ts}")
        if group["ac_power_kw"].nunique(dropna=False) <= 1:
            keep_indices.append(int(group.index[0]))
            continue
        exp = expected_measured_power(group.iloc[[0]]).iloc[0]
        vals = group["ac_power_kw"]
        qualifies = ((vals - exp).abs() <= 0.10 * exp) if pd.notna(exp) else pd.Series(False, index=group.index)
        if int(qualifies.sum()) == 1:
            keep_indices.append(int(qualifies[qualifies].index[0]))
        elif int(qualifies.sum()) == 2:
            first = int(group.index[0])
            out.loc[first, "ac_power_kw"] = vals.mean()
            keep_indices.append(first)
        elif float(geo.loc[[ts], "elevation"].iloc[0]) < 0:
            first = int(group.index[0])
            out.loc[first, "ac_power_kw"] = 0.0
            keep_indices.append(first)
        else:
            first = int(group.index[0])
            out.loc[first, "ac_power_kw"] = np.nan
            keep_indices.append(first)
    keep = set(keep_indices)
    keep.update(int(i) for i in out.index[~duplicate_mask])
    result = out.loc[sorted(keep)].sort_values("timestamp", kind="stable").reset_index(drop=True)
    assert result["timestamp"].is_unique
    return result


def clean_training_rows(raw_train: pd.DataFrame, fold: Fold) -> pd.DataFrame:
    """Row-local training target cleanup; deliberately no soiling normalization."""
    # Caller supplies only this fold's rows, before any target/sensor cleanup.
    source = raw_train.copy()
    source = parse_timestamps(source)
    assert (source["timestamp"] < fold.train_end_exclusive).all(), "future row passed into target cleaning"
    assert len(source) > 0
    assert source["timestamp"].max() < fold.valid_start
    source = replace_sentinels(source)


    mw_window = source["timestamp"].between(MW_START, MW_END)
    positive = mw_window & (source["ac_power_kw"] > 0)
    source.loc[positive, "ac_power_kw"] *= 1000.0


    source.loc[source["ac_power_kw"] > 10_100, "ac_power_kw"] = np.nan
    overshoot = source["ac_power_kw"].between(AC_KW, 10_100, inclusive="right")
    source.loc[overshoot, "ac_power_kw"] = AC_KW
    power_flat = (source["timestamp"].between(POWER_FLAT_START, POWER_FLAT_END)
                  & np.isclose(source["ac_power_kw"], POWER_FLAT_VALUE))
    source.loc[power_flat, "ac_power_kw"] = np.nan
    poa_flat = (source["timestamp"].between(POA_FLAT_START, POA_FLAT_END)
                & np.isclose(source["measured_poa_wm2"], POA_FLAT_VALUE))
    source.loc[poa_flat, "measured_poa_wm2"] = np.nan

    source = resolve_training_duplicates(source)
    geo = geometry(source["timestamp"])
    source["elevation"] = geo["elevation"].to_numpy()

    night_negative = (source["elevation"] < 0) & (source["ac_power_kw"] < 0)
    source.loc[night_negative, "ac_power_kw"] = 0.0


    exp = expected_measured_power(source)
    ratio = source["ac_power_kw"] / exp.where(exp > 0)
    poa = source["measured_poa_wm2"]
    spike = (poa > 100) & (ratio > 1.5)
    source.loc[spike, "ac_power_kw"] = np.nan
    source["training_target_kw"] = source["ac_power_kw"]


    status = source["status"].astype("string").str.strip().str.upper()
    exp_after = expected_measured_power(source)
    act_over_exp = source["ac_power_kw"] / exp_after.where(exp_after > 0)
    poa300 = source["measured_poa_wm2"] > 300
    low = poa300 & (act_over_exp < 0.5)
    excl = (
        status.eq("STOP").fillna(False)
        | status.eq("PARTIAL").fillna(False)
        | source["ac_power_kw"].isna()
        | (status.eq("STANDBY").fillna(False) & (source["elevation"] > 0) & poa300)
        | (status.isna() & low).fillna(False)
    )
    source["train_valid"] = ~excl
    source["status_clean"] = status

    assert source["timestamp"].max() < fold.valid_start
    return source.reset_index(drop=True)


def safe_forecasts(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Interpolate only within each target calendar day; never cross a day boundary."""
    out = frame.copy()
    ts = pd.DatetimeIndex(out["timestamp"])
    days = ts.normalize()
    counts: dict[str, int] = {}
    for col in FORECAST_COLS:
        s = pd.Series(out[col].to_numpy(dtype=float), index=ts)
        s = s.replace(list(SENTINELS), np.nan)
        before = int(s.isna().sum())
        filled = s.groupby(days).transform(lambda x: x.interpolate(method="time", limit_area="inside"))
        counts[col] = int(before - filled.isna().sum())
        counts[f"unresolved_{col}"] = int(filled.isna().sum())
        out[col] = filled.to_numpy()
    return out, counts


def all_required_forecasts_finite(frame: pd.DataFrame) -> np.ndarray:
    return np.isfinite(frame[FORECAST_COLS].to_numpy(dtype=float)).all(axis=1)


def valid_state_mask(frame: pd.DataFrame, geo: pd.DataFrame, target: np.ndarray) -> np.ndarray:
    """Offline scoring eligibility from row-local recorded validity/status rules."""
    status = frame["status"].astype("string").str.strip().str.upper()
    poa = frame["measured_poa_wm2"]
    module_temp = frame["measured_module_temp_c"]
    expected = (10.8 * poa * (1 + TEMP_COEFF * (module_temp - 25))).clip(upper=AC_KW)
    low = (poa > 300) & ((pd.Series(target, index=frame.index) / expected.where(expected > 0)) < 0.5)
    target_series = pd.Series(target, index=frame.index)
    ratio = target_series / expected.where(expected > 0)
    impossible_spike = (poa > 100) & (ratio > 1.5)
    over_rating_spike = target_series > 10_100
    frozen_logger = (
        frame["timestamp"].between(POWER_FLAT_START, POWER_FLAT_END)
        & np.isclose(target, POWER_FLAT_VALUE)
    )
    invalid = (
        status.eq("STOP").fillna(False)
        | status.eq("PARTIAL").fillna(False)
        | (status.eq("STANDBY").fillna(False) & (geo["elevation"].to_numpy() > 0) & (poa.to_numpy() > 300))
        | (status.isna().to_numpy() & low.fillna(False).to_numpy())
        | impossible_spike.fillna(False).to_numpy()
        | over_rating_spike.fillna(False).to_numpy()
        | frozen_logger.to_numpy()
    )
    return (~invalid) & np.isfinite(target)


def forecast_features(frame: pd.DataFrame, geo: pd.DataFrame, phys: "FoldPhysics") -> pd.DataFrame:
    """Experimental equivalent of production features with only same-day imputation."""
    fc, _ = safe_forecasts(frame)
    ts = pd.DatetimeIndex(fc["timestamp"])
    days = ts.normalize()
    X = pd.DataFrame(index=np.arange(len(fc)))
    X["cloud"] = fc["forecast_cloud_cover"].to_numpy()
    X["temp"] = fc["forecast_temp_c"].to_numpy()
    X["wind"] = fc["forecast_wind_ms"].to_numpy()
    X["rh"] = fc["forecast_humidity_pct"].to_numpy()

    cloud = pd.Series(X["cloud"].to_numpy(), index=ts)
    full_idx = pd.date_range(ts.min().normalize(), ts.max().normalize() + pd.Timedelta(hours=23), freq="h")
    cloud_full = cloud.reindex(full_idx)
    for width, name in ((1, "cloud_s1"), (2, "cloud_s2")):
        smooth = cloud_full.groupby(cloud_full.index.normalize()).transform(
            lambda s: s.rolling(2 * width + 1, center=True, min_periods=1).mean()
        )
        X[name] = smooth.reindex(ts).to_numpy()

    daylight = geo["elevation"].to_numpy() > 0
    day_mean = cloud[daylight].groupby(days[daylight]).mean()
    X["cloud_day_mean"] = pd.Series(days).map(day_mean).to_numpy()

    X["elevation"] = geo["elevation"].to_numpy()
    X["azimuth"] = geo["azimuth"].to_numpy()
    aoi = pvlib.irradiance.aoi(TILT, AZIMUTH, geo["apparent_zenith"].to_numpy(), geo["azimuth"].to_numpy())
    X["cos_aoi"] = np.cos(np.radians(np.asarray(aoi)))
    comp = phys.components(geo, fc)
    for col in ("cs_poa", "phys_poa", "cell_temp", "phys_power"):
        X[col] = comp[col].to_numpy()
    X["hour"] = ts.hour
    doy = ts.dayofyear
    X["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    X["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return X[FEATURES]


class FoldPhysics:
    """B2b physics chain whose empirical parameters are fitted on one training fold."""
    def __init__(self):
        self.tl: dict[int, float] | None = None
        self.ab: tuple[float, float] | None = None
        self.loss: float | None = None
        self.calibration_timestamps: pd.DatetimeIndex | None = None

    def _unit_power(self, geo: pd.DataFrame, frame: pd.DataFrame, return_components=False):
        fc, _ = safe_forecasts(frame)
        cs = clearsky(geo, self.tl)
        ghi = cs["ghi"].to_numpy() * cloud_factor(fc["forecast_cloud_cover"].to_numpy(), *self.ab)
        poa = allsky_poa(geo, ghi)
        tcell = pvlib.temperature.faiman(poa, fc["forecast_temp_c"].to_numpy(), fc["forecast_wind_ms"].to_numpy())
        temp_factor = 1 + TEMP_COEFF * (tcell - 25)
        unit = DC_KWP * poa / 1000 * temp_factor
        if return_components:
            return unit, poa, cs["poa"].to_numpy(), tcell
        return unit

    def fit(self, train: pd.DataFrame, geo: pd.DataFrame) -> "FoldPhysics":
        assert train["timestamp"].is_monotonic_increasing
        assert train["timestamp"].equals(pd.Series(geo.index, name="timestamp")) or (
            train["timestamp"].to_numpy() == geo.index.to_numpy()
        ).all()
        self.calibration_timestamps = pd.DatetimeIndex(train["timestamp"])
        fc, _ = safe_forecasts(train)
        # Fit turbidity on this training slice only; fitted_dry excludes Jun-Sep from
        # direct fitting and uses the same production fallback rule for those months.
        exclude = MONSOON_MONTHS if TL_MODE == "fitted_dry" else ()
        self.tl, _ = fit_linke(
            geo,
            train["measured_poa_wm2"].to_numpy(),
            np.ones(len(train), dtype=bool),
            exclude_months=exclude,
        )
        cs_poa = clearsky(geo, self.tl)["poa"].to_numpy()
        measured_poa = train["measured_poa_wm2"].to_numpy(dtype=float)
        kt = measured_poa / np.where(cs_poa > 0, cs_poa, np.nan)
        sel_cloud = (geo["elevation"].to_numpy() > 10) & np.isfinite(kt) & np.isfinite(fc["forecast_cloud_cover"])
        self.ab = fit_cloud_curve(fc["forecast_cloud_cover"].to_numpy()[sel_cloud], np.clip(kt[sel_cloud], 0, 1.2))

        unit = self._unit_power(geo, train)
        y = train["training_target_kw"].to_numpy(dtype=float)
        sel_loss = (
            train["train_valid"].to_numpy()
            & (geo["elevation"].to_numpy() > 15)
            & np.isfinite(y)
            & (y < 9500)
            & np.isfinite(kt)
            & (kt >= 0.8)
            & (fc["forecast_cloud_cover"].to_numpy() <= 0.3)
            & (unit > 0)
        )
        assert int(sel_loss.sum()) > 0, "no training rows available for physics loss fit"
        self.loss = float(y[sel_loss].sum() / unit[sel_loss].sum())
        return self

    def components(self, geo: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
        unit, poa, cs_poa, tcell = self._unit_power(geo, frame, return_components=True)
        power = np.clip(unit * self.loss, 0, AC_KW)
        power[geo["elevation"].to_numpy() <= 0] = 0.0
        return pd.DataFrame(
            {"cs_poa": cs_poa, "phys_poa": poa, "cell_temp": tcell, "phys_power": power},
            index=geo.index,
        )

    def predict(self, geo: pd.DataFrame, frame: pd.DataFrame) -> np.ndarray:
        return self.components(geo, frame)["phys_power"].to_numpy()


def fit_b0(train: pd.DataFrame) -> dict:
    """Training-only month/hour climatology, with the plan's fallback hierarchy."""
    valid = train["train_valid"].to_numpy() & np.isfinite(train["training_target_kw"].to_numpy())
    daylight = train["elevation"].to_numpy() > 0
    use = valid
    y = train["training_target_kw"].to_numpy(dtype=float)
    all_train = pd.DataFrame({
        "month": train["timestamp"].dt.month.to_numpy()[use],
        "hour": train["timestamp"].dt.hour.to_numpy()[use],
        "y": y[use],
    })
    daylight_train = all_train.loc[daylight[use]]
    return {
        "month_hour": all_train.groupby(["month", "hour"])["y"].median().to_dict(),
        "month": daylight_train.groupby("month")["y"].median().to_dict(),
        "hour": all_train.groupby("hour")["y"].median().to_dict(),
        "overall": float(all_train["y"].median()),
    }


def predict_b0(frame: pd.DataFrame, geo: pd.DataFrame, climatology: dict) -> np.ndarray:
    result = np.empty(len(frame), dtype=float)
    for i, ts in enumerate(frame["timestamp"]):
        if geo["elevation"].iloc[i] <= 0:
            result[i] = 0.0
            continue
        key = (int(ts.month), int(ts.hour))
        result[i] = climatology["month_hour"].get(
            key,
            climatology["month"].get(
                int(ts.month),
                climatology["hour"].get(int(ts.hour), climatology["overall"]),
            ),
        )
    return np.clip(result, 0, AC_KW)


def fit_m2(train: pd.DataFrame, geo: pd.DataFrame, phys: FoldPhysics):
    model_inputs = train[["timestamp", *FORECAST_COLS]]
    X = forecast_features(model_inputs, geo, phys)
    y = train["training_target_kw"].to_numpy(dtype=float)
    denominator = X["phys_power"].to_numpy(dtype=float)
    eligible = (
        train["train_valid"].to_numpy()
        & (geo["elevation"].to_numpy() > 0)
        & np.isfinite(y)
        & np.isfinite(denominator)
        & (denominator >= MIN_PHYS_KW)
        & np.isfinite(X.to_numpy(dtype=float)).all(axis=1)
    )
    assert int(eligible.sum()) > 0, "no M2 training rows"
    ratio_target = y[eligible] / denominator[eligible]
    models = []
    for seed in M2_SEEDS:
        params = dict(BASE_PARAMS, num_leaves=M2_LEAVES, seed=seed)
        models.append(
            lgb.train(
                params,
                lgb.Dataset(X.loc[eligible, FEATURES], label=ratio_target),
                num_boost_round=M2_ROUNDS,
            )
        )
    return models, int(eligible.sum())


def predict_m2(models, frame: pd.DataFrame, geo: pd.DataFrame, phys: FoldPhysics) -> np.ndarray:
    model_inputs = frame[["timestamp", *FORECAST_COLS]]
    X = forecast_features(model_inputs, geo, phys)
    raw_ratio = np.mean([model.predict(X[FEATURES]) for model in models], axis=0)
    physics = X["phys_power"].to_numpy(dtype=float)
    pred = np.where(physics >= MIN_PHYS_KW, physics * np.clip(raw_ratio, 0, None), physics)
    pred = np.clip(pred, 0, AC_KW)
    pred[geo["elevation"].to_numpy() <= 0] = 0.0
    return pred


def clean_validation_rows(raw_valid: pd.DataFrame, fold: Fold) -> tuple[pd.DataFrame, dict[str, int]]:
    """Preserve raw target values; only remove invalid/ambiguous validation observations."""
    frame = parse_timestamps(raw_valid)
    frame = frame[frame["timestamp"].between(fold.valid_start, fold.valid_end_inclusive)].copy()
    raw_actual = frame["ac_power_kw"].to_numpy(dtype=float)
    target_sentinel = np.isin(raw_actual, SENTINELS)
    frame["actual_raw_kw"] = raw_actual
    frame["target_sentinel"] = target_sentinel
    frame = replace_sentinels(frame)
    counts = {"in_period": int(len(frame))}
    # A raw reading from the documented MW logging interval is not a kW actual. Do not
    # rescale validation actuals: exclude the known invalid interval instead.
    unit_bad = frame["timestamp"].between(MW_START, MW_END)
    counts["known_mw_unit_issue"] = int(unit_bad.sum())
    frame = frame.loc[~unit_bad].copy()

    # Keep an exact duplicate only when its actual is identical; conflicting raw
    # targets cannot be reduced to one truthful validation actual without alteration.
    dup = frame["timestamp"].duplicated(keep=False)
    counts["duplicate_timestamp_hours"] = int(frame.loc[dup, "timestamp"].nunique())
    conflicting_times = set(
        ts for ts, group in frame.loc[dup].groupby("timestamp")
        if group["ac_power_kw"].nunique(dropna=False) > 1
    )
    counts["conflicting_duplicate_hours"] = int(len(conflicting_times))
    conflict_mask = frame["timestamp"].isin(conflicting_times)
    counts["conflicting_duplicate_rows"] = int(conflict_mask.sum())
    nonconflicting = frame.loc[~conflict_mask].copy()
    counts["identical_duplicate_surplus_rows"] = int(nonconflicting["timestamp"].duplicated(keep="first").sum())
    frame = nonconflicting.drop_duplicates("timestamp", keep="first").copy()
    assert frame["timestamp"].is_unique

    # Do not rewrite actuals: keep the value exactly as read (apart from parsing),
    # including nighttime negative standby readings.
    forecast_view, imputed = safe_forecasts(frame[["timestamp", *FORECAST_COLS]])
    for col in FORECAST_COLS:
        frame[col] = forecast_view[col].to_numpy()
    counts.update({f"imputed_{k}": v for k, v in imputed.items()})
    geo = geometry(frame["timestamp"])
    valid = valid_state_mask(frame, geo, frame["actual_raw_kw"].to_numpy(dtype=float))
    target_sentinel = frame["target_sentinel"].to_numpy(dtype=bool)
    valid &= ~target_sentinel
    forecast_ok = all_required_forecasts_finite(frame)
    valid &= forecast_ok
    counts["unusable_forecast_rows"] = int((~forecast_ok).sum())
    state_target_valid = valid_state_mask(frame, geo, frame["actual_raw_kw"].to_numpy(dtype=float))
    counts["invalid_target_or_state_rows"] = int((~state_target_valid | target_sentinel).sum())
    status = frame["status"].astype("string").str.strip().str.upper()
    poa = frame["measured_poa_wm2"]
    module_temp = frame["measured_module_temp_c"]
    expected = (10.8 * poa * (1 + TEMP_COEFF * (module_temp - 25))).clip(upper=AC_KW)
    ratio = frame["actual_raw_kw"] / expected.where(expected > 0)
    counts["exclude_target_nonfinite_or_sentinel_rows"] = int(
        (~np.isfinite(frame["actual_raw_kw"].to_numpy(dtype=float)) | target_sentinel).sum()
    )
    counts["exclude_stop_rows"] = int(status.eq("STOP").fillna(False).sum())
    counts["exclude_partial_rows"] = int(status.eq("PARTIAL").fillna(False).sum())
    counts["exclude_daylight_standby_rows"] = int((
        status.eq("STANDBY").fillna(False)
        & (geo["elevation"].to_numpy() > 0)
        & (poa.to_numpy() > 300)
    ).sum())
    counts["exclude_missing_status_low_output_rows"] = int((
        status.isna().to_numpy()
        & (poa.to_numpy() > 300)
        & (ratio.to_numpy() < 0.5)
    ).sum())
    counts["exclude_power_poa_ratio_spike_rows"] = int(((poa > 100) & (ratio > 1.5)).fillna(False).sum())
    counts["exclude_power_above_10100_rows"] = int((frame["actual_raw_kw"] > 10_100).sum())
    counts["exclude_frozen_power_rows"] = int((
        frame["timestamp"].between(POWER_FLAT_START, POWER_FLAT_END)
        & np.isclose(frame["actual_raw_kw"].to_numpy(dtype=float), POWER_FLAT_VALUE)
    ).sum())
    frame["is_valid"] = valid
    frame["is_daylight"] = geo["elevation"].to_numpy() > 0
    frame["elevation"] = geo["elevation"].to_numpy()
    return frame.reset_index(drop=True), counts


def metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    err = pred - actual
    return {
        "n_rows": int(len(actual)),
        "mae_kw": float(np.mean(np.abs(err))) if len(actual) else np.nan,
        "rmse_kw": float(np.sqrt(np.mean(err ** 2))) if len(actual) else np.nan,
        "mean_bias_kw": float(np.mean(err)) if len(actual) else np.nan,
        "mae_pct_capacity": float(100 * np.mean(np.abs(err)) / AC_KW) if len(actual) else np.nan,
    }


def validate_fold(raw_train: pd.DataFrame, raw_all: pd.DataFrame, fold: Fold):
    train_input = raw_train.loc[
        raw_train["timestamp_parsed"] < fold.train_end_exclusive
    ].drop(columns="timestamp_parsed").copy()
    train = clean_training_rows(train_input, fold)
    valid_raw = raw_all[raw_all["timestamp_parsed"].between(fold.valid_start, fold.valid_end_inclusive)].copy()
    valid, valid_counts = clean_validation_rows(valid_raw, fold)
    assert train["timestamp"].max() < valid["timestamp"].min()
    train_ts = set(pd.DatetimeIndex(train["timestamp"]))
    valid_ts = set(pd.DatetimeIndex(valid["timestamp"]))
    assert train_ts.isdisjoint(valid_ts)
    assert train["timestamp"].max() < fold.valid_start
    assert valid["timestamp"].min() >= fold.valid_start
    assert valid["timestamp"].max() <= fold.valid_end_inclusive


    train_geo = geometry(train["timestamp"])
    val_geo = geometry(valid["timestamp"])
    assert train_geo.index.is_unique and val_geo.index.is_unique
    assert train_geo.index.max() < val_geo.index.min()
    phys = FoldPhysics().fit(train, train_geo)
    assert set(phys.calibration_timestamps).issubset(train_ts)
    assert set(phys.calibration_timestamps).isdisjoint(valid_ts)

    b0 = fit_b0(train)
    model_valid = valid[["timestamp", *FORECAST_COLS]]
    b0_predictions = predict_b0(model_valid, val_geo, b0)
    b2b_predictions = phys.predict(val_geo, model_valid)
    m2_models, m2_n_train = fit_m2(train, train_geo, phys)
    m2_predictions = predict_m2(m2_models, model_valid, val_geo, phys)
    bl50_predictions = np.clip(0.5 * b2b_predictions + 0.5 * m2_predictions, 0, AC_KW)
    daylight = val_geo["elevation"].to_numpy() > 0
    for predictions in (b0_predictions, b2b_predictions, m2_predictions, bl50_predictions):
        assert np.isfinite(predictions).all()
        assert ((predictions >= 0) & (predictions <= AC_KW)).all()
        assert (predictions[~daylight] == 0).all()

    models = {
        "B0": b0_predictions,
        "B2b": b2b_predictions,
        "M2": m2_predictions,
        "BL50": bl50_predictions,
    }
    scored = valid["is_valid"].to_numpy(dtype=bool)
    actual = valid["actual_raw_kw"].to_numpy(dtype=float)
    prediction_rows = []
    metric_rows = []
    for model_name, pred in models.items():
        assert np.isfinite(pred[scored]).all()
        for population, mask in (("ALL-VALID", scored), ("DAYLIGHT-VALID", scored & daylight)):
            row_metrics = metrics(actual[mask], pred[mask])
            metric_rows.append({"fold": fold.name, "model": model_name, "population": population, **row_metrics})
        for i in np.flatnonzero(scored):
            prediction_rows.append({
                "timestamp": valid["timestamp"].iloc[i],
                "fold": fold.name,
                "model": model_name,
                "actual_ac_power_kw": actual[i],
                "predicted_ac_power_kw": float(pred[i]),
                "is_daylight": bool(daylight[i]),
                "is_valid": True,
            })


    p = pd.DataFrame(prediction_rows)
    assert not p.duplicated(["fold", "model", "timestamp"]).any()
    assert p.groupby(["fold", "timestamp"])["actual_ac_power_kw"].nunique().max() <= 1
    metadata = {
        "fold": fold.name,
        "train_start": train["timestamp"].min().isoformat(),
        "train_end": train["timestamp"].max().isoformat(),
        "valid_start": valid["timestamp"].min().isoformat(),
        "valid_end": valid["timestamp"].max().isoformat(),
        "training_rows_after_dedup": int(len(train)),
        "m2_training_rows": int(m2_n_train),
        "validation_rows_after_quality_dedup": int(len(valid)),
        "scored_all_valid_rows": int(scored.sum()),
        "scored_daylight_valid_rows": int((scored & daylight).sum()),
        **valid_counts,
        "calibration_rows": int(len(phys.calibration_timestamps)),
        "physics_tl_mode": TL_MODE,
        "physics_tl": json.dumps(phys.tl, sort_keys=True),
        "physics_cloud_a": phys.ab[0],
        "physics_cloud_b": phys.ab[1],
        "physics_loss_factor": phys.loss,
        "m2_leaves": M2_LEAVES,
        "m2_rounds": M2_ROUNDS,
        "m2_seeds": ",".join(map(str, M2_SEEDS)),
    }
    return metric_rows, prediction_rows, metadata


def main():
    raw = pd.read_csv(DATA / "train.csv", dtype={"timestamp": str, "status": str})
    raw["timestamp_parsed"] = pd.to_datetime(
        raw["timestamp"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
    ).fillna(pd.to_datetime(raw["timestamp"], format="%d-%m-%Y %H:%M", errors="coerce"))
    assert raw["timestamp_parsed"].notna().all()

    all_metrics, all_predictions, all_metadata = [], [], []
    for fold in FOLDS:
        print(f"[{fold.name}] train < {fold.valid_start} ; validate {fold.valid_start} .. {fold.valid_end_inclusive}")
        fold_metrics, fold_predictions, meta = validate_fold(raw, raw, fold)
        all_metrics.extend(fold_metrics)
        all_predictions.extend(fold_predictions)
        all_metadata.append(meta)
        print(
            f"  rows all/daylight={meta['scored_all_valid_rows']}/{meta['scored_daylight_valid_rows']}; "
            f"M2 train={meta['m2_training_rows']}; excluded unit-window={meta['known_mw_unit_issue']}; "
            f"bad-forecast rows={meta['unusable_forecast_rows']}"
        )

    pred_df = pd.DataFrame(all_predictions)
    fold_df = pd.DataFrame(all_metrics)
    assert not pred_df.duplicated(["fold", "model", "timestamp"]).any()
    assert pred_df["predicted_ac_power_kw"].map(np.isfinite).all()

    # Construct pooled populations explicitly so no fold-level percentage averaging occurs.
    pooled_rows = []
    for model_name, group in pred_df.groupby("model", sort=False):
        for population, mask in (
            ("ALL-VALID", group["is_valid"].to_numpy(dtype=bool)),
            ("DAYLIGHT-VALID", group["is_valid"].to_numpy(dtype=bool) & group["is_daylight"].to_numpy(dtype=bool)),
        ):
            y = group.loc[mask, "actual_ac_power_kw"].to_numpy(dtype=float)
            p = group.loc[mask, "predicted_ac_power_kw"].to_numpy(dtype=float)
            pooled_rows.append({"model": model_name, "population": population, **metrics(y, p)})
    summary_df = pd.DataFrame(pooled_rows)

    RESULTS.mkdir(parents=True, exist_ok=True)
    pred_df = pred_df[[
        "timestamp", "fold", "model", "actual_ac_power_kw", "predicted_ac_power_kw", "is_daylight", "is_valid"
    ]].sort_values(["fold", "timestamp", "model"], kind="stable")
    fold_df = fold_df[[
        "fold", "model", "population", "n_rows", "mae_kw", "rmse_kw", "mean_bias_kw", "mae_pct_capacity"
    ]]
    summary_df = summary_df[[
        "model", "population", "n_rows", "mae_kw", "rmse_kw", "mean_bias_kw", "mae_pct_capacity"
    ]]
    pred_df.to_csv(RESULTS / "validation_predictions.csv", index=False, lineterminator="\n")
    fold_df.to_csv(RESULTS / "fold_results.csv", index=False, lineterminator="\n")
    summary_df.to_csv(RESULTS / "model_summary.csv", index=False, lineterminator="\n")
    pd.DataFrame(all_metadata).to_csv(RESULTS / "fold_audit.csv", index=False, lineterminator="\n")

    print("\nPOOLED RESULTS")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nFOLD RESULTS")
    print(fold_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nWrote isolated experiment outputs to {RESULTS}")


if __name__ == "__main__":
    main()
