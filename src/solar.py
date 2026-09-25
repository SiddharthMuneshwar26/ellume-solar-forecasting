import numpy as np
import pandas as pd
import pvlib

LAT, LON, ALT = 14.10, 77.28, 680
TILT, AZIMUTH, ALBEDO = 15, 180, 0.20
TZ = "Asia/Kolkata"
DC_KWP, AC_KW = 12_500, 10_000
TEMP_COEFF = -0.0035
KASTEN_CZEPLAK = (0.75, 3.4)   # GHI = GHI_cs * (1 - a * C**b)
FORECAST_COLS = ["forecast_cloud_cover", "forecast_temp_c", "forecast_wind_ms", "forecast_humidity_pct"]
TL_GRID = np.round(np.arange(1.5, 8.01, 0.05), 2)
MONSOON_MONTHS = (6, 7, 8, 9)


def _midpoints(timestamps):
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps))
    return (ts + pd.Timedelta(minutes=30)).tz_localize(TZ)


def clearsky_poa(timestamps):
    """Ineichen clear-sky (pvlib default Linke turbidity) GHI and POA, plus elevation/azimuth.

    Used by the audit and cleaning scripts for diagnostics.
    """
    geo = geometry(timestamps)
    cs = clearsky(geo, tl=None)
    return pd.DataFrame({"cs_ghi": cs["ghi"], "cs_poa": cs["poa"], "elevation": geo["elevation"],
                         "azimuth": geo["azimuth"]}, index=geo.index)

def geometry(timestamps):
    """Everything that depends only on the timestamp, computed once and reused."""
    mid = _midpoints(timestamps)
    sp = pvlib.solarposition.get_solarposition(mid, LAT, LON, altitude=ALT)
    am_rel = pvlib.atmosphere.get_relative_airmass(sp["apparent_zenith"])
    am_abs = pvlib.atmosphere.get_absolute_airmass(am_rel, pvlib.atmosphere.alt2pres(ALT))
    out = pd.DataFrame({
        "apparent_zenith": sp["apparent_zenith"].values,
        "zenith": sp["zenith"].values,
        "azimuth": sp["azimuth"].values,
        "elevation": sp["apparent_elevation"].values,
        "airmass_abs": am_abs.values,
        "dni_extra": pvlib.irradiance.get_extra_radiation(mid).values,
        "doy": mid.dayofyear.values,
        "month": mid.month.values,
    }, index=pd.DatetimeIndex(pd.to_datetime(timestamps)))
    return out


def default_linke(geo):
    """pvlib's climatological Linke turbidity (Remund et al. lookup), per row."""
    mid = _midpoints(geo.index)
    return pvlib.clearsky.lookup_linke_turbidity(mid, LAT, LON).values


def default_linke_by_month():
    """pvlib's default TL for each calendar month (independent of which rows are in training)."""
    days = pd.date_range("2023-01-15 12:00", periods=12, freq="MS", tz=TZ) + pd.Timedelta(days=14)
    vals = pvlib.clearsky.lookup_linke_turbidity(days, LAT, LON, interp_turbidity=False).values
    return {int(d.month): float(v) for d, v in zip(days, vals)}


def transpose(geo, ghi, dni, dhi):
    """Hay-Davies transposition to the fixed 15 deg south-facing plane."""
    poa = pvlib.irradiance.get_total_irradiance(
        TILT, AZIMUTH, geo["apparent_zenith"], geo["azimuth"], dni, ghi, dhi,
        dni_extra=geo["dni_extra"], albedo=ALBEDO, model="haydavies")
    return poa["poa_global"].fillna(0).clip(lower=0).values


def clearsky(geo, tl=None):
    """Ineichen clear sky. `tl`: None -> pvlib lookup; dict {month: TL}; or per-row array."""
    if tl is None:
        tl_row = default_linke(geo)
    elif isinstance(tl, dict):
        tl_row = geo["month"].map(tl).values
    else:
        tl_row = np.asarray(tl)
    cs = pvlib.clearsky.ineichen(geo["apparent_zenith"], geo["airmass_abs"], tl_row,
                                 altitude=ALT, dni_extra=geo["dni_extra"])
    cs = cs.fillna(0)
    return pd.DataFrame({"ghi": cs["ghi"].values, "dni": cs["dni"].values, "dhi": cs["dhi"].values,
                         "poa": transpose(geo, cs["ghi"], cs["dni"], cs["dhi"])}, index=geo.index)


def allsky_poa(geo, ghi):
    """Split an all-sky GHI into DNI/DHI with Erbs, then transpose to POA."""
    ghi = pd.Series(np.asarray(ghi, float), index=geo.index)
    erbs = pvlib.irradiance.erbs(ghi, geo["zenith"], geo["doy"])
    return transpose(geo, ghi, pd.Series(erbs["dni"], index=geo.index).fillna(0),
                     pd.Series(erbs["dhi"], index=geo.index).fillna(0))


def cloud_factor(cloud, a, b):
    return 1 - a * np.clip(cloud, 0, 1) ** b


def temp_factor(poa, temp_air, wind):
    """Faiman cell temperature (pvlib default u0=25, u1=6.84) and the -0.35 %/degC power factor."""
    t_cell = pvlib.temperature.faiman(poa, temp_air, wind)
    return 1 + TEMP_COEFF * (t_cell - 25)


def prepare_forecast(df, verbose=False):
    """Fill gaps in the forecast columns, then return a copy of the DataFrame with only the forecast columns."""
    out = df.copy()
    ts = pd.DatetimeIndex(out["timestamp"])
    day, hour = ts.normalize(), ts.hour
    for c in FORECAST_COLS:
        s = pd.Series(out[c].where(out[c] != -999).values, index=ts)
        n0 = int(s.isna().sum())
        if n0 == 0:
            continue
        s = s.groupby(day).transform(lambda x: x.interpolate(method="time", limit_area="inside"))
        n1 = int(s.isna().sum())
        if n1:
            lookup = s.groupby([day, hour]).first()
            for i in np.where(s.isna())[0]:
                vals = [lookup.get((day[i] + pd.Timedelta(days=k), hour[i]), np.nan) for k in (-1, 1)]
                vals = [v for v in vals if pd.notna(v)]
                if vals:
                    s.iloc[i] = float(np.mean(vals))
        n2 = int(s.isna().sum())
        if n2:
            s = s.groupby(day).transform(lambda x: x.ffill().bfill())
        if verbose:
            print(f"  forecast gaps {c}: {n0} -> same-day interpolation {n0 - n1}, "
                  f"adjacent-day same hour {n1 - n2}, nearest in day {n2 - int(s.isna().sum())}")
        out[c] = s.values
    return out

def fit_linke(geo, measured_poa, mask, top_frac=0.10, min_days=3, exclude_months=()):
    """Fit monthly Linke turbidity on the clearest days of the training data.
    Returns a dict {month: TL} and some info for logging.
    """
    mask = np.asarray(mask) & ~np.isin(geo["month"].values, list(exclude_months))
    g = geo[mask]
    m = pd.Series(np.asarray(measured_poa)[mask], index=g.index)
    cs_def = clearsky(g, None)["poa"]
    ok = m.notna() & (g["elevation"] > 0)
    day = g.index.normalize()
    daily = pd.DataFrame({"meas": m.where(ok), "cs": cs_def.where(ok), "n": ok}).groupby(day).sum()
    daily = daily[daily["n"] >= 8]
    daily["k"] = daily["meas"] / daily["cs"]
    daily["month"] = daily.index.month
    clear_days = []
    for _, d in daily.groupby("month"):
        n = max(min_days, int(np.ceil(top_frac * len(d))))
        clear_days += list(d.nlargest(n, "k").index)
    sel = ok & day.isin(clear_days) & (g["elevation"] > 20)
    gs, ms = g[sel], m[sel].values
    sse = {}
    for tl in TL_GRID:
        pred = clearsky(gs, np.full(len(gs), tl))["poa"].values
        sse[tl] = pd.Series((pred - ms) ** 2).groupby(gs["month"].values).sum()
    sse = pd.DataFrame(sse)
    fitted = {int(mo): float(sse.columns[np.argmin(r.values)]) for mo, r in sse.iterrows()}

    ref = default_linke_by_month()
    ratio = np.mean([fitted[mo] / ref[mo] for mo in fitted])
    tl = {mo: fitted.get(mo, float(ref[mo] * ratio)) for mo in range(1, 13)}
    info = {"fitted_months": sorted(fitted), "n_clear_hours": int(sel.sum()), "fallback_ratio": ratio}
    return tl, info


def fit_cloud_curve(cloud, kt):
    """Fit kt = 1 - a * C**b by grid search (least squares)."""
    c = np.clip(np.asarray(cloud, float), 0, 1)
    k = np.asarray(kt, float)
    A = np.arange(0.0, 1.001, 0.01)
    B = np.arange(0.5, 8.01, 0.1)
    best = (np.inf, None, None)
    cb = c[None, :] ** B[:, None]
    for a in A:
        err = ((1 - a * cb) - k[None, :]) ** 2
        s = err.sum(axis=1)
        j = int(np.argmin(s))
        if s[j] < best[0]:
            best = (s[j], a, B[j])
    return float(best[1]), float(best[2])

class PhysicsModel:
    """Forecast-driven physics chain for the plant, with fitted turbidity, cloud-curve and loss factor."""

    def __init__(self, tl="fitted", cloud="fitted"):
        self.tl_mode, self.cloud_mode = tl, cloud
        self.tl, self.ab, self.loss, self.info = None, None, None, {}

    def _unit_power(self, geo, fc, return_tcell=False):
        """Power per unit loss_factor (before clipping) and POA."""
        cs = clearsky(geo, self.tl)
        if self.cloud_mode == "none":
            poa = cs["poa"].values
        else:
            ghi = cs["ghi"].values * cloud_factor(fc["forecast_cloud_cover"].values, *self.ab)
            poa = allsky_poa(geo, ghi)
        t_cell = pvlib.temperature.faiman(poa, fc["forecast_temp_c"].values, fc["forecast_wind_ms"].values)
        tf = 1 + TEMP_COEFF * (t_cell - 25)
        if return_tcell:
            return DC_KWP * poa / 1000 * tf, poa, cs["poa"].values, np.asarray(t_cell)
        return DC_KWP * poa / 1000 * tf, poa, cs["poa"].values

    def components(self, geo, df):
        """Intermediate physics quantities for use as ML features (forecast + timestamp only)."""
        fc = prepare_forecast(df)
        unit, poa, cs_poa, t_cell = self._unit_power(geo, fc, return_tcell=True)
        p = np.clip(unit * self.loss, 0, AC_KW)
        p[geo["elevation"].values < 0] = 0.0
        return pd.DataFrame({"cs_poa": cs_poa, "phys_poa": poa, "cell_temp": t_cell, "phys_power": p},
                            index=geo.index)

    def fit(self, geo, df):
        """geo/df: TRAINING rows of the fold only (same index order)."""
        fc = prepare_forecast(df)
        ok = df["train_ok"].values

        if self.tl_mode in ("fitted", "fitted_dry"):
            excl = MONSOON_MONTHS if self.tl_mode == "fitted_dry" else ()
            self.tl, self.info["tl"] = fit_linke(geo, df["measured_poa_wm2"].values, np.ones(len(df), bool),
                                                 exclude_months=excl)
        else:
            self.tl = None
        cs_poa = clearsky(geo, self.tl)["poa"].values
        kt = df["measured_poa_wm2"].values / np.where(cs_poa > 0, cs_poa, np.nan)

        if self.cloud_mode == "kc":
            self.ab = KASTEN_CZEPLAK
        elif self.cloud_mode == "fitted":
            sel = (geo["elevation"].values > 10) & np.isfinite(kt)
            self.ab = fit_cloud_curve(fc["forecast_cloud_cover"].values[sel], np.clip(kt[sel], 0, 1.2))

        unit, _, _ = self._unit_power(geo, fc)
        y = df["ac_power_clean"].values
        sel = (ok & (geo["elevation"].values > 15) & np.isfinite(y) & (y < 9500) & np.isfinite(kt)
               & (kt >= 0.8) & (fc["forecast_cloud_cover"].values <= 0.3) & (unit > 0))
        self.loss = float(y[sel].sum() / unit[sel].sum())
        self.info["n_loss_hours"] = int(sel.sum())
        return self

    def predict(self, geo, df):
        fc = prepare_forecast(df)
        unit, _, _ = self._unit_power(geo, fc)
        p = np.clip(unit * self.loss, 0, AC_KW)
        p[geo["elevation"].values < 0] = 0.0
        return p

    def params(self):
        d = {"tl_mode": self.tl_mode, "cloud_mode": self.cloud_mode, "loss_factor": round(self.loss, 4)}
        if self.ab is not None:
            d["cloud_a"], d["cloud_b"] = round(self.ab[0], 3), round(self.ab[1], 2)
        if isinstance(self.tl, dict):
            d["tl_by_month"] = {k: round(v, 2) for k, v in self.tl.items()}
        d.update({k: v for k, v in self.info.items() if k != "tl"})
        if "tl" in self.info:
            d["tl_fitted_months"] = self.info["tl"]["fitted_months"]
        return d
