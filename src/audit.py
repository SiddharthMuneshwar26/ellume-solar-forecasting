"""Data audit. Read-only - nothing here modifies or cleans the data.

Loads data/train.csv and data/test.csv, runs quality checks and writes tables (CSV),
plots (PNG) and a plain-text summary to reports/. Interpretation lives in NOTES.md.

Run:  .venv\\Scripts\\python.exe -m src.audit
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.solar import AC_KW, clearsky_poa

ROOT = Path(__file__).resolve().parents[1]
DATA, REPORTS = ROOT / "data", ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

FORECAST_COLS = ["forecast_cloud_cover", "forecast_temp_c", "forecast_wind_ms", "forecast_humidity_pct"]
SENTINELS = [-999.0, 9999.0]
DAY_EL = 10       # deg; "daylight" for flatline / sanity checks
MIDDAY_EL = 40    # deg; "midday" for module-temperature and ratio checks

_summary = []


def say(*parts):
    line = " ".join(str(p) for p in parts)
    print(line)
    _summary.append(line)


def section(title):
    say("")
    say("=" * 78)
    say(title)
    say("=" * 78)


def save_table(df, name):
    df.to_csv(REPORTS / name)
    say(f"  -> reports/{name}")


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(REPORTS / name, dpi=110)
    plt.close(fig)
    say(f"  -> reports/{name}")


def runs(mask):
    """Group consecutive True values of a boolean Series -> DataFrame(start, end, length)."""
    mask = mask.fillna(False).astype(bool)
    grp = (mask != mask.shift()).cumsum()
    out = []
    for _, g in mask[mask].groupby(grp[mask]):
        out.append((g.index[0], g.index[-1], len(g)))
    return pd.DataFrame(out, columns=["start", "end", "length"])
# Loading
def parse_timestamps(raw):
    iso = pd.to_datetime(raw, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    dmy = pd.to_datetime(raw, format="%d-%m-%Y %H:%M", errors="coerce")
    return iso.fillna(dmy), iso.notna(), dmy.notna() & iso.isna()


def load(path):
    df = pd.read_csv(path, dtype={"status": str})
    df["ts"], is_iso, is_dmy = parse_timestamps(df["timestamp"])
    df["ts_format"] = np.where(is_iso, "ISO", np.where(is_dmy, "DD-MM-YYYY HH:MM", "UNPARSED"))
    return df
# 1. Timestamps
def audit_timestamps(df, name, start, end):
    section(f"1. TIMESTAMPS - {name}")
    say(f"rows: {len(df)}")
    say("timestamp string formats:", df["ts_format"].value_counts().to_dict())
    if (df["ts_format"] == "DD-MM-YYYY HH:MM").any():
        ex = df.loc[df["ts_format"] == "DD-MM-YYYY HH:MM", "timestamp"].head(3).tolist()
        say("  examples of non-ISO strings:", ex)
        # Day-first is unambiguous if any day > 12 appears in the first field
        first = df.loc[df["ts_format"] == "DD-MM-YYYY HH:MM", "timestamp"].str[:2].astype(int)
        say(f"  first field > 12 in {int((first > 12).sum())} rows -> format is day-first")
    say("unparsed timestamps:", int(df["ts"].isna().sum()))

    back = df["ts"].diff() < pd.Timedelta(0)
    say(f"out-of-order rows (ts earlier than previous row): {int(back.sum())}")
    if back.any():
        for i in np.where(back)[0]:
            say(f"  row {i}: {df['ts'].iloc[i - 1]} -> {df['ts'].iloc[i]}")

    dup_mask = df["ts"].duplicated(keep=False)
    n_extra = int(df["ts"].duplicated().sum())
    say(f"duplicated timestamps: {dup_mask.sum()} rows over {df.loc[dup_mask, 'ts'].nunique()} "
        f"distinct hours ({n_extra} surplus rows)")
    if dup_mask.any():
        value_cols = [c for c in df.columns if c not in ("timestamp", "ts", "ts_format")]
        d = df[dup_mask].copy()
        conflict = d.groupby("ts")[value_cols].nunique(dropna=False).gt(1)
        n_conf = int(conflict.any(axis=1).sum())
        say(f"  duplicate hours whose rows DISAGREE on some value: {n_conf}")
        say("  columns that disagree (count of hours):", conflict.sum()[conflict.sum() > 0].to_dict())
        mixed_fmt = d.groupby("ts")["ts_format"].nunique().gt(1).sum()
        say(f"  duplicate hours where one copy is ISO and the other DD-MM: {int(mixed_fmt)}")
        save_table(d.sort_values("ts").set_index("ts"), f"{name}_duplicates.csv")

    full = pd.date_range(start, end, freq="h")
    say(f"expected hours {start} .. {end}: {len(full)}; distinct present: {df['ts'].nunique()}")
    missing = full.difference(pd.DatetimeIndex(df["ts"].dropna().unique()))
    extra = pd.DatetimeIndex(df["ts"].dropna().unique()).difference(full)
    say(f"missing hours: {len(missing)}; hours outside expected range: {len(extra)}")
    if len(missing):
        m = pd.Series(True, index=missing).reindex(full, fill_value=False)
        gaps = runs(m)
        say("  gap length distribution:", gaps["length"].value_counts().sort_index().to_dict())
        say("  gaps longer than 1 h:")
        for _, r in gaps[gaps["length"] > 1].iterrows():
            say(f"    {r.start} .. {r.end}  ({r.length} h)")
        mh = pd.Series(missing.hour).value_counts().sort_index()
        say("  missing hours by hour-of-day:", mh.to_dict())
        save_table(gaps, f"{name}_gaps.csv")
    return missing
# 2. Status
def audit_status(tr):
    section("2. STATUS")
    raw = tr["status"].fillna("<NaN>").map(repr).value_counts()
    say("raw values (repr, so whitespace is visible):")
    for k, v in raw.items():
        say(f"  {k:<14} {v}")
    norm = tr["status"].str.strip().str.upper()
    say("normalised (strip + upper):", norm.value_counts(dropna=False).to_dict())
    t = tr.assign(status_norm=norm.fillna("<NaN>"))
    tab = t.groupby("status_norm").agg(
        n=("ac_power_kw", "size"),
        day_share=("elevation", lambda e: (e > 0).mean()),
        mean_kw=("ac_power_kw", "mean"),
        median_kw=("ac_power_kw", "median"),
        mean_poa=("measured_poa_wm2", "mean"),
        power_nan=("ac_power_kw", lambda p: p.isna().sum()),
    )
    say(tab.round(2).to_string())
    save_table(tab, "status_summary.csv")
    # Status vs daylight consistency
    day = t["elevation"] > DAY_EL
    say(f"STANDBY while sun > {DAY_EL} deg: {int(((t.status_norm == 'STANDBY') & day).sum())}")
    say(f"RUN while sun < -5 deg: {int(((t.status_norm == 'RUN') & (t.elevation < -5)).sum())}")
    stop = t[t.status_norm.isin(["STOP", "PARTIAL"])]
    say("STOP / PARTIAL by month:")
    say(stop.groupby([stop.ts.dt.to_period("M"), "status_norm"]).size().unstack(fill_value=0).to_string())
    st = t[t.status_norm == "STOP"]
    say(f"STOP rows with daytime power > 100 kW: {int(((st.ac_power_kw > 100) & (st.elevation > DAY_EL)).sum())} "
        f"of {int((st.elevation > DAY_EL).sum())} daytime STOP rows")
    # Stop events as runs
    stop_runs = runs(pd.Series((t.status_norm == "STOP").values, index=t.ts))
    say(f"STOP episodes: {len(stop_runs)}; longest: {stop_runs['length'].max() if len(stop_runs) else 0} h")
    save_table(stop_runs, "status_stop_runs.csv")
    na = t[t.status_norm == "<NaN>"]
    say(f"NaN status rows: {len(na)}; of which power NaN: {int(na.ac_power_kw.isna().sum())}; "
        f"daytime: {int((na.elevation > DAY_EL).sum())}")
# 3. ac_power_kw
def audit_power(tr):
    section("3. AC_POWER_KW")
    p = tr["ac_power_kw"]
    say(p.describe().round(2).to_string())
    say(f"NaN: {int(p.isna().sum())}")
    for s in SENTINELS:
        say(f"exact sentinel {s}: {int((p == s).sum())}")
    pv = p.where(~p.isin(SENTINELS))

    # Negatives
    neg = pv < 0
    night = tr["elevation"] < 0
    say(f"negative (non-sentinel): {int(neg.sum())}  night: {int((neg & night).sum())}  "
        f"day(el>0): {int((neg & ~night).sum())}  day(el>{DAY_EL}): {int((neg & (tr.elevation > DAY_EL)).sum())}")
    say(f"  negative range: {pv[neg].min():.2f} .. {pv[neg].max():.2f} kW; "
        f"night-time median {pv[neg & night].median():.2f} kW (inverter/transformer standby draw)")
    dn = tr[neg & (tr.elevation > DAY_EL)]
    if len(dn):
        say("  daytime negatives (el>10):")
        say(dn[["ts", "elevation", "measured_poa_wm2", "status", "ac_power_kw"]]
            .to_string(index=False, float_format="%.2f"))

    # > AC rating
    over = tr[pv > AC_KW]
    say(f"> {AC_KW} kW: {len(over)}")
    say(f"  10000-10100 kW (marginal overshoot at clipping): {int(((pv > AC_KW) & (pv <= 10100)).sum())}")
    say(f"  > 10100 kW (impossible spikes): {int((pv > 10100).sum())}")
    big = tr[pv > 10100].copy()
    if len(big):
        # compare against neighbours to see if spike is isolated
        big["prev_kw"] = tr.ac_power_kw.shift(1).loc[big.index]
        big["next_kw"] = tr.ac_power_kw.shift(-1).loc[big.index]
        say(big[["ts", "measured_poa_wm2", "status", "prev_kw", "ac_power_kw", "next_kw"]]
            .to_string(index=False, float_format="%.1f"))
    save_table(over.set_index("ts")[["measured_poa_wm2", "status", "ac_power_kw"]], "power_over_rating.csv")

    # Night-time
    nt = tr[night]
    say(f"night rows (sun el<0 at mid-hour): {len(nt)}")
    say(f"  |power|<=0.01: {int((nt.ac_power_kw.abs() <= 0.01).sum())}; negative: {int((pv[night] < 0).sum())}; "
        f"positive>1 kW: {int((pv[night] > 1).sum())}; >50 kW: {int((pv[night] > 50).sum())}; "
        f">500 kW: {int((pv[night] > 500).sum())}")
    np_ = tr[night & (pv > 50)].copy()
    np_["dawn_dusk"] = np.where(np_.elevation > -8, "twilight (el>-8)", "deep night")
    say("  night >50 kW by type:", np_["dawn_dusk"].value_counts().to_dict())
    save_table(np_.set_index("ts")[["elevation", "measured_poa_wm2", "status", "ac_power_kw", "dawn_dusk"]],
               "power_night_positive.csv")

    # Flatlines: same value for 3+ consecutive hours (hourly-contiguous), any time; report daylight separately
    s = tr.set_index("ts")["ac_power_kw"]
    cont = s.index.to_series().diff().eq(pd.Timedelta("1h"))
    same = s.eq(s.shift()) & cont & s.notna() & (s != 0)
    grp = (~same).cumsum()
    flat_rows = []
    for _, block in s.groupby(grp):
        if len(block) >= 3 and block.iloc[0] != 0 and block.nunique() == 1 and block.notna().all():
            el = tr.set_index("ts").loc[block.index, "elevation"]
            flat_rows.append((block.index[0], block.index[-1], len(block), block.iloc[0],
                              int((el > DAY_EL).sum()), int((el < 0).sum())))
    flat = pd.DataFrame(flat_rows, columns=["start", "end", "hours", "value_kw", "daylight_hours", "night_hours"])
    say(f"flatline runs (identical non-zero value, >=3 consecutive hours): {len(flat)}")
    if len(flat):
        say(flat.to_string(index=False))
    save_table(flat, "power_flatlines.csv")

    # Ceiling: look at the upper tail per month for a pile-up of values
    day = tr[(tr.elevation > DAY_EL) & (pv > 0)].copy()
    day["m"] = day.ts.dt.to_period("M")
    rows = []
    for m, g in day.groupby("m"):
        v = g.ac_power_kw[(g.ac_power_kw > 20) & (g.ac_power_kw <= 10100)]  # kW regime, no spikes
        if v.empty:
            continue
        top = v.max()
        rows.append(dict(month=str(m), max_kw=top, p99=v.quantile(.99),
                         n_within_0p5pct_of_max=int((v >= top * 0.995).sum()),
                         n_ge_9900=int((v >= 9900).sum()),
                         n_9500_9900=int(((v >= 9500) & (v < 9900)).sum())))
    ceil = pd.DataFrame(rows).set_index("month")
    say("upper tail by month (kW-regime daylight values, spikes >10100 excluded):")
    say(ceil.round(1).to_string())
    save_table(ceil, "power_ceiling_by_month.csv")
    # Daily max identical across days is a ceiling signature
    dmax = day.groupby(day.ts.dt.date).ac_power_kw.max()
    dmax = dmax[(dmax > 1000) & (dmax <= 10100)]
    vc = dmax.round(0).value_counts()
    say("most common daily maxima (rounded to 1 kW):", vc.head(8).to_dict())

    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].hist(pv[(pv > 7000) & (pv < 10200)], bins=160)
    ax[0].set_title("ac_power_kw histogram, 7000-10200 kW")
    ax[0].axvline(AC_KW, color="r", lw=0.8)
    ax[1].plot(dmax.index, dmax.values, ".", ms=3)
    ax[1].axhline(AC_KW, color="r", lw=0.8)
    ax[1].set_title("daily max ac_power_kw (kW regime)")
    savefig(fig, "power_upper_tail.png")

    # Unit switch: kW values ~1000x too small on days with good irradiance
    dd = tr[(tr.elevation > 0)].copy()
    dd["date"] = dd.ts.dt.date
    status = dd.status.str.strip().str.upper()
    dd = dd[status != "STOP"]  # stopped days also have max power < 50 kW but are negative, not MW
    daily = dd.groupby("date").agg(max_kw=("ac_power_kw", "max"), max_poa=("measured_poa_wm2",
                                   lambda x: x[~x.isin(SENTINELS)].max()),
                                   n=("ac_power_kw", "size"))
    daily["kw_per_poa"] = daily.max_kw / daily.max_poa
    sus = daily[(daily.max_poa > 300) & (daily.max_kw < 50) & (daily.max_kw > 0)]
    say(f"days with max POA > 300 W/m2 but 0 < max power < 50 kW, status not STOP (candidate MW-unit days): {len(sus)}")
    if len(sus):
        r = runs(pd.Series(True, index=pd.to_datetime(sus.index)).reindex(
            pd.date_range(daily.index.min(), daily.index.max()), fill_value=False).rename(None))
        say("  contiguous date blocks:")
        for _, b in r.iterrows():
            say(f"    {b.start.date()} .. {b.end.date()}  ({b.length} days)")
        med = sus.kw_per_poa.median()
        ok = daily[(daily.max_poa > 300) & (daily.max_kw >= 50)].kw_per_poa.median()
        say(f"  median (daily max power / daily max POA): suspect days {med:.5f}, normal days {ok:.3f} "
            f"-> ratio {ok / med:.0f}x (1000x => MW)")
    save_table(daily, "power_daily_max.csv")
    # exact hourly boundaries of the MW regime: daylight hours where power/POA is ~1000x too small
    r_h = tr.ac_power_kw / tr.measured_poa_wm2.where(tr.measured_poa_wm2.between(100, 1500))
    mw = r_h.between(0.001, 0.05)
    if mw.any():
        say(f"  hourly: {int(mw.sum())} hours with 0.001 < power/POA < 0.05 (kW regime is ~10); "
            f"first {tr.ts[mw].min()}, last {tr.ts[mw].max()}")
        say(f"  kW-regime hours (power/POA 1..20) with POA>=100 inside that window: "
            f"{int((r_h.between(1, 20) & tr.ts.between(tr.ts[mw].min(), tr.ts[mw].max())).sum())}")
        save_table(tr[mw].set_index("ts")[["measured_poa_wm2", "status", "ac_power_kw"]], "power_mw_unit_hours.csv")

    # Ceiling below 10 MW? Compare power with a simple expected power from measured POA.
    # 10.8 kW per W/m2 is the median temperature-corrected ratio from section 6.
    stn = tr.status.str.strip().str.upper()
    tcorr = 1 - 0.0035 * (tr.measured_module_temp_c - 25)
    exp_kw = 10.8 * tr.measured_poa_wm2 * tcorr
    hi = tr[(stn == "RUN") & exp_kw.between(8500, 20000) & pv.between(1000, 10100)].copy()
    hi["expected_kw"] = exp_kw
    hi["ratio"] = hi.ac_power_kw / hi.expected_kw
    hi["bin"] = pd.cut(hi.expected_kw, [8500, 9000, 9500, 10000, 10500, 11000, 20000])
    tab = hi.groupby("bin", observed=True).ac_power_kw.describe()[["count", "25%", "50%", "75%", "max"]]
    say("power vs expected power (RUN hours) - a cap below 10 MW would flatten the upper bins:")
    say(tab.round(0).to_string())
    tab2 = hi.groupby(hi.ts.dt.to_period("M")).agg(n=("expected_kw", "size"), power_max=("ac_power_kw", "max"),
                                                   actual_over_expected=("ratio", "median"))
    say("  by month (hours with expected >= 8.5 MW):")
    say(tab2.round(3).to_string())
    save_table(tab2, "power_ceiling_vs_expected.csv")

    # Unit switch at hourly level inside otherwise-normal days (partial-day switch)
    hr = tr[(tr.measured_poa_wm2 > 300) & ~tr.measured_poa_wm2.isin(SENTINELS)]
    tiny = hr[(hr.ac_power_kw > 0) & (hr.ac_power_kw < 30)]
    say(f"hours with POA>300 and 0<power<30 kW: {len(tiny)}; "
        f"months: {tiny.ts.dt.to_period('M').value_counts().sort_index().to_dict()}")
    zero_day = hr[(hr.ac_power_kw.abs() < 1)]
    say(f"hours with POA>300 and |power|<1 kW (outage/curtailment): {len(zero_day)}; "
        f"status: {zero_day.status.str.strip().str.upper().value_counts(dropna=False).to_dict()}")

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.semilogy(tr.ts, tr.ac_power_kw.clip(lower=0.1), ",", alpha=0.6)
    ax.set_title("ac_power_kw over time (log scale; MW-unit periods show as a band ~1000x lower)")
    savefig(fig, "power_timeseries_log.png")
# 4. Diurnal profile / time shift
def audit_diurnal(tr):
    section("4. DIURNAL PROFILE BY MONTH (hour-beginning slots)")
    ok = tr[tr.ac_power_kw.between(-100, 10100)].copy()  # drops sentinels and >10.1 MW spikes
    ok["m"] = ok.ts.dt.to_period("M").astype(str)
    ok["h"] = ok.ts.dt.hour
    # exclude MW-unit days so they do not drag the monthly profile down
    daymax = ok.groupby(ok.ts.dt.date).ac_power_kw.transform("max")
    ok = ok[daymax > 100]
    prof = ok.pivot_table(index="h", columns="m", values="ac_power_kw", aggfunc="mean")
    save_table(prof.round(1), "diurnal_power_by_month.csv")
    cs = tr.assign(m=tr.ts.dt.to_period("M").astype(str), h=tr.ts.dt.hour) \
        .pivot_table(index="h", columns="m", values="cs_poa", aggfunc="mean")
    poa = tr[~tr.measured_poa_wm2.isin(SENTINELS)].assign(m=lambda d: d.ts.dt.to_period("M").astype(str),
                                                           h=lambda d: d.ts.dt.hour) \
        .pivot_table(index="h", columns="m", values="measured_poa_wm2", aggfunc="mean")

    def centroid(col):
        w = col.clip(lower=0).fillna(0)
        return float(((w.index + 0.5) * w).sum() / w.sum())

    rows = []
    for m in prof.columns:
        rows.append(dict(month=m, peak_slot=int(prof[m].idxmax()),
                         power_centroid_h=centroid(prof[m]),
                         measured_poa_centroid_h=centroid(poa[m]) if m in poa else np.nan,
                         clearsky_poa_centroid_h=centroid(cs[m])))
    t = pd.DataFrame(rows).set_index("month")
    t["power_minus_clearsky_min"] = (t.power_centroid_h - t.clearsky_poa_centroid_h) * 60
    t["poa_minus_clearsky_min"] = (t.measured_poa_centroid_h - t.clearsky_poa_centroid_h) * 60
    t["power_minus_poa_min"] = (t.power_centroid_h - t.measured_poa_centroid_h) * 60
    say("peak slot and energy-weighted centroid (hours, slot midpoint = h+0.5) vs clear-sky POA:")
    say(t.round(2).to_string())
    save_table(t, "diurnal_shift_by_month.csv")
    flagged = t[(t.peak_slot != 12) | (t.power_minus_clearsky_min.abs() > 20)]
    say("months flagged (peak slot != 12 or |centroid shift| > 20 min):",
        flagged.index.tolist() if len(flagged) else "none")

    # Per-day shift to locate exact boundaries of any shifted period
    d = tr[tr.ac_power_kw.between(0, 10100) & (tr.cs_poa > 0)].copy()
    d["date"] = d.ts.dt.date
    d["hm"] = d.ts.dt.hour + 0.5

    def daily_centroid(g, col):
        w = g[col].clip(lower=0)
        return (g.hm * w).sum() / w.sum() if w.sum() > 0 else np.nan

    dayc = pd.DataFrame({
        "power": d.groupby("date").apply(lambda g: daily_centroid(g, "ac_power_kw"), include_groups=False),
        "poa": d[~d.measured_poa_wm2.isin(SENTINELS)].groupby("date")
            .apply(lambda g: daily_centroid(g, "measured_poa_wm2"), include_groups=False),
        "clearsky": d.groupby("date").apply(lambda g: daily_centroid(g, "cs_poa"), include_groups=False),
        "daily_kwh": d.groupby("date").ac_power_kw.sum(),
    })
    dayc = dayc[dayc.daily_kwh > 5000]
    dayc["power_shift_min"] = (dayc.power - dayc.clearsky) * 60
    dayc["poa_shift_min"] = (dayc.poa - dayc.clearsky) * 60
    dayc["power_vs_poa_min"] = (dayc.power - dayc.poa) * 60
    save_table(dayc, "diurnal_shift_by_day.csv")
    say(f"daily power-centroid offset vs clear-sky: median {dayc.power_shift_min.median():.1f} min, "
        f"IQR {dayc.power_shift_min.quantile(.25):.1f}..{dayc.power_shift_min.quantile(.75):.1f} min")
    odd = dayc[(dayc.power_shift_min.abs() > 20) | (dayc.power_vs_poa_min.abs() > 15)]
    say("single days with |power-clearsky| > 20 min or |power-POA| > 15 min:")
    say(odd.round(1).to_string() if len(odd) else "  none")
    roll = (dayc[["power_shift_min", "poa_shift_min", "power_vs_poa_min"]]
            .rolling(7, center=True, min_periods=3).median())
    big = roll[roll.power_vs_poa_min.abs() > 30]
    if len(big):
        b = runs(pd.Series(True, index=pd.to_datetime(big.index)).reindex(
            pd.to_datetime(roll.index), fill_value=False))
        say("periods where 7-day median power centroid differs from POA centroid by > 30 min:")
        for _, r in b.iterrows():
            seg = dayc.loc[r.start.date():r.end.date()]
            say(f"  {r.start.date()} .. {r.end.date()} ({r.length} days w/ data): "
                f"median power-vs-POA {seg.power_vs_poa_min.median():.0f} min, "
                f"power-vs-clearsky {seg.power_shift_min.median():.0f} min")
    else:
        say("no multi-day period with power centroid > 30 min away from POA centroid")

    fig, axes = plt.subplots(3, 6, figsize=(20, 9), sharex=True, sharey=True)
    for ax, m in zip(axes.flat, prof.columns):
        ax.plot(prof.index, prof[m] / prof[m].max(), label="power")
        if m in poa:
            ax.plot(poa.index, poa[m] / poa[m].max(), label="meas POA", alpha=0.7)
        ax.plot(cs.index, cs[m] / cs[m].max(), "k--", lw=0.8, label="clear-sky POA")
        ax.axvline(12, color="grey", lw=0.5)
        ax.set_title(f"{m}  peak={int(prof[m].idxmax())}")
    axes.flat[0].legend(fontsize=7)
    savefig(fig, "diurnal_profiles_by_month.png")

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(pd.to_datetime(dayc.index), dayc.power_shift_min, ".", ms=3, label="power - clearsky (daily)")
    ax.plot(pd.to_datetime(roll.index), roll.power_shift_min, lw=1.2, label="power - clearsky (7d median)")
    ax.plot(pd.to_datetime(roll.index), roll.poa_shift_min, lw=1.2, label="meas POA - clearsky (7d median)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("centroid offset [min]")
    ax.legend()
    ax.set_title("Daily energy-centroid offset relative to clear-sky POA")
    savefig(fig, "diurnal_shift_by_day.png")
# 5. Sensors
def audit_sensors(tr):
    section("5. SENSOR SANITY")
    for c in ["measured_poa_wm2", "measured_module_temp_c", "measured_ambient_temp_c", "rainfall_mm"]:
        v = tr[c]
        say(f"{c}: NaN {int(v.isna().sum())}, -999 {int((v == -999).sum())}, 9999 {int((v == 9999).sum())}, "
            f"range(non-sentinel) {v[~v.isin(SENTINELS)].min():.2f} .. {v[~v.isin(SENTINELS)].max():.2f}")
    poa = tr.measured_poa_wm2.where(~tr.measured_poa_wm2.isin(SENTINELS))
    night = tr.elevation < -2
    say(f"POA at night (el<-2): n={int(night.sum())}, ==0: {int((poa[night] == 0).sum())}, "
        f"!=0: {int((poa[night].fillna(0) != 0).sum())}, >5: {int((poa[night] > 5).sum())}, "
        f"<0: {int((poa[night] < 0).sum())}")
    no = tr[night & (poa.abs() > 5)]
    say("  night POA > 5 W/m2 values:", no.measured_poa_wm2.round(1).value_counts().head(10).to_dict())
    nm = tr[night].assign(m=tr.ts.dt.to_period("M")).groupby("m").measured_poa_wm2.agg(
        lambda x: x[~x.isin(SENTINELS)].median())
    say("  median night POA by month:", nm.round(2).to_dict())
    say(f"negative POA any time: {int((poa < 0).sum())}")

    # Flatlines in POA / temps during daylight
    s_idx = tr.set_index("ts")
    cont = s_idx.index.to_series().diff().eq(pd.Timedelta("1h"))
    for c in ["measured_poa_wm2", "measured_module_temp_c", "measured_ambient_temp_c"]:
        s = s_idx[c]
        same = s.eq(s.shift()) & cont
        grp = (~same).cumsum()
        rr = []
        for _, block in s.groupby(grp):
            if len(block) >= 3 and block.notna().all() and not (c == "measured_poa_wm2" and block.iloc[0] == 0):
                el = s_idx.loc[block.index, "elevation"]
                rr.append((block.index[0], block.index[-1], len(block), block.iloc[0], int((el > DAY_EL).sum())))
        f = pd.DataFrame(rr, columns=["start", "end", "hours", "value", "daylight_hours"])
        say(f"{c} flatlines (>=3 h identical{', non-zero' if 'poa' in c else ''}): {len(f)}")
        if len(f):
            say(f.to_string(index=False))
        save_table(f, f"sensor_flatlines_{c}.csv")

    # POA vs clear-sky
    k = poa / tr.cs_poa
    day = tr.elevation > DAY_EL
    say(f"daylight POA > 1.3 x clear-sky POA: {int((day & (k > 1.3)).sum())}; > 1.5x: {int((day & (k > 1.5)).sum())}")
    say(f"daylight POA == 0 with clear-sky > 300: {int((day & (poa == 0) & (tr.cs_poa > 300)).sum())}")

    # Module vs ambient at midday
    mid = (tr.elevation > MIDDAY_EL) & (poa > 600)
    dT = tr.measured_module_temp_c - tr.measured_ambient_temp_c
    dT = dT.where(~tr.measured_module_temp_c.isin(SENTINELS) & ~tr.measured_ambient_temp_c.isin(SENTINELS))
    say(f"midday (el>{MIDDAY_EL}, POA>600) rows: {int(mid.sum())}; module-ambient median {dT[mid].median():.1f} C")
    bad = mid & (dT < 0)
    say(f"  module temp BELOW ambient at midday: {int(bad.sum())}; below by >5 C: {int((mid & (dT < -5)).sum())}")
    if bad.any():
        say("  by month:", tr[bad].ts.dt.to_period("M").value_counts().sort_index().astype(int).to_dict())
        save_table(tr[bad].set_index("ts")[["elevation", "measured_poa_wm2", "measured_module_temp_c",
                                           "measured_ambient_temp_c", "ac_power_kw", "status"]],
                   "sensor_module_below_ambient.csv")
    # a drifting module sensor would show as a trend in the monthly midday module-ambient difference
    mdT = pd.DataFrame({"dT": dT[mid], "m": tr.ts[mid].dt.to_period("M")}).groupby("m").dT.median()
    say("  monthly median module-ambient at midday:", mdT.round(1).to_dict())
    # ambient sensor vs forecast temp
    diff = (tr.measured_ambient_temp_c - tr.forecast_temp_c).where(~tr.measured_ambient_temp_c.isin(SENTINELS))
    say(f"measured ambient - forecast temp: mean {diff.mean():.2f}, std {diff.std():.2f}, "
        f"|diff|>10 C: {int((diff.abs() > 10).sum())}")

    # Rain
    r = tr.rainfall_mm
    say(f"rainfall: hours >0: {int((r > 0).sum())}, max {r.max():.1f} mm, negative {int((r < 0).sum())}")
    say("  monthly total mm:", r.groupby(tr.ts.dt.to_period("M")).sum().round(0).to_dict())

    fig, ax = plt.subplots(1, 3, figsize=(17, 4))
    ax[0].plot(tr.ts[night], poa[night], ",")
    ax[0].set_title("POA at night (el < -2)")
    ax[1].plot(tr.ts[mid], dT[mid], ",")
    ax[1].axhline(0, color="r", lw=0.6)
    ax[1].set_title("module - ambient at midday (POA>600)")
    ax[2].scatter(tr.cs_poa[day], poa[day], s=1, alpha=0.3)
    ax[2].plot([0, 1200], [0, 1200], "r", lw=0.6)
    ax[2].set_xlabel("clear-sky POA")
    ax[2].set_ylabel("measured POA")
    ax[2].set_title("measured vs clear-sky POA (daylight)")
    savefig(fig, "sensor_sanity.png")
# 6. Performance ratio / soiling / degradation
def audit_ratio(tr):
    section("6. AC POWER / MEASURED POA RATIO (soiling, degradation)")
    status = tr.status.str.strip().str.upper()
    poa = tr.measured_poa_wm2
    p = tr.ac_power_kw
    # Filter: running, good irradiance, below clipping, sane values, not MW-unit
    f = (status == "RUN") & poa.between(400, 1300) & p.between(1000, 9000)
    d = tr[f].copy()
    d["ratio"] = d.ac_power_kw / d.measured_poa_wm2  # kW per W/m2
    # temperature-correct to 25 C module temp for a fairer comparison
    tcorr = 1 - 0.0035 * (d.measured_module_temp_c - 25)
    d["ratio_tc"] = d.ratio / tcorr.where(~d.measured_module_temp_c.isin(SENTINELS))
    d["date"] = d.ts.dt.date
    say(f"hours used: {len(d)} (RUN, 400<=POA<=1300, 1000<=P<=9000)")
    say(f"ratio kW/(W/m2): median {d.ratio.median():.3f}; temp-corrected median {d.ratio_tc.median():.3f} "
        f"(nameplate 12.5 kWp -> 12.5 kW per W/m2 at STC; implied PR {d.ratio_tc.median() / 12.5:.2f})")
    daily = d.groupby("date").agg(ratio=("ratio", "median"), ratio_tc=("ratio_tc", "median"), n=("ratio", "size"))
    daily = daily[daily.n >= 3]
    daily.index = pd.to_datetime(daily.index)
    rain = tr.groupby(tr.ts.dt.date).rainfall_mm.sum()
    rain.index = pd.to_datetime(rain.index)
    daily["rain_mm"] = rain.reindex(daily.index)
    save_table(daily, "ratio_daily.csv")
    monthly = d.groupby(d.ts.dt.to_period("M")).ratio_tc.median()
    say("monthly median temp-corrected ratio:", monthly.round(3).to_dict())

    # Linear trend (degradation) on temp-corrected daily ratio
    x = (daily.index - daily.index[0]).days.values / 365.25
    y = daily.ratio_tc.values
    ok = np.isfinite(y)
    slope, icpt = np.polyfit(x[ok], y[ok], 1)
    say(f"linear trend of daily temp-corrected ratio: {slope:+.3f} per year = {100 * slope / icpt:+.2f} %/yr")
    # Same-season year-on-year comparison (Jan-Jun 2024 vs Jan-Jun 2025)
    for mth in range(1, 7):
        a = monthly.get(pd.Period(f"2024-{mth:02d}", "M"))
        b = monthly.get(pd.Period(f"2025-{mth:02d}", "M"))
        if a is not None and b is not None:
            say(f"  month {mth:02d}: 2024 {a:.3f} -> 2025 {b:.3f}  ({100 * (b / a - 1):+.1f} %)")

    # Sawtooth: slow decline between rain events, step up after rain
    daily["roll"] = daily.ratio_tc.rolling(5, center=True, min_periods=3).median()
    jumps = daily.roll.diff()
    big_up = daily[jumps > 0.25]
    say(f"days where 5-day median ratio jumps up by > 0.25: {len(big_up)}")
    if len(big_up):
        r7 = rain.rolling(7).sum()
        for dt_ in big_up.index[:40]:
            say(f"  {dt_.date()}  jump {jumps.loc[dt_]:+.2f}  rain in prior 7 days {r7.get(dt_, np.nan):.1f} mm")
    # slope between rains during dry spells
    dry = daily[daily.rain_mm.fillna(0) == 0]
    if len(dry) > 30:
        seg = (daily.rain_mm.fillna(0) > 2).cumsum().reindex(dry.index)
        slopes = []
        for _, blk in dry.groupby(seg):
            if len(blk) >= 10:
                xx = (blk.index - blk.index[0]).days.values
                sl = np.polyfit(xx, blk.ratio_tc.values, 1)[0]
                slopes.append((blk.index[0].date(), blk.index[-1].date(), len(blk), sl, 100 * sl / blk.ratio_tc.mean()))
        sl = pd.DataFrame(slopes, columns=["start", "end", "days", "slope_per_day", "pct_per_day"])
        say("dry-spell (no rain > 2 mm) segments >= 10 days, ratio slope:")
        say(sl.round(4).to_string(index=False) if len(sl) else "  none")
        save_table(sl, "ratio_dry_spell_slopes.csv")

    fig, ax = plt.subplots(2, 1, figsize=(15, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax[0].plot(daily.index, daily.ratio_tc, ".", ms=3, alpha=0.6, label="daily median (temp-corrected)")
    ax[0].plot(daily.index, daily.roll, lw=1.2, label="5-day median")
    ax[0].plot(daily.index, icpt + slope * x, "k--", lw=0.8, label=f"trend {100 * slope / icpt:+.2f}%/yr")
    ax[0].set_ylabel("kW per W/m2")
    ax[0].legend()
    ax[0].set_title("AC power / measured POA (RUN, POA 400-1300, P 1-9 MW)")
    ax[1].bar(rain.index, rain.values, width=1)
    ax[1].set_ylabel("rain mm/day")
    savefig(fig, "ratio_over_time.png")
# 7. Forecast columns
def audit_forecasts(tr, te):
    section("7. FORECAST COLUMNS")
    limits = {"forecast_cloud_cover": (0, 1), "forecast_temp_c": (-5, 50),
              "forecast_wind_ms": (0, 40), "forecast_humidity_pct": (0, 100)}
    rows = []
    for name, df in [("train", tr), ("test", te)]:
        for c in FORECAST_COLS:
            v_raw = df[c]
            v = v_raw.where(~v_raw.isin(SENTINELS))
            lo, hi = limits[c]
            decimals = v.dropna().map(lambda x: len(repr(float(x)).split(".")[1].rstrip("0"))).max()
            rows.append(dict(file=name, col=c, n=len(v), nulls=int(v_raw.isna().sum()),
                             sentinel_m999=int((v_raw == -999).sum()), min=v.min(), max=v.max(),
                             mean=v.mean(), std=v.std(), below_phys=int((v < lo).sum()), above_phys=int((v > hi).sum()),
                             n_unique=v.nunique(), max_decimals=decimals))
    t = pd.DataFrame(rows).set_index(["file", "col"])
    say("(min/max/mean/std computed with -999 sentinels excluded)")
    say(t.round(3).to_string())
    save_table(t, "forecast_ranges.csv")
    for name, df in [("train", tr), ("test", te)]:
        bad = df[df[FORECAST_COLS].isna().any(axis=1) | df[FORECAST_COLS].isin(SENTINELS).any(axis=1)]
        say(f"{name}: rows with a NaN or -999 in a forecast column: {len(bad)}")
        if len(bad):
            say(bad.set_index("ts")[FORECAST_COLS].to_string())
            save_table(bad.set_index("ts")[FORECAST_COLS], f"forecast_bad_values_{name}.csv")

    # Everything below works on sentinel-masked copies.
    tr = tr.copy()
    te = te.copy()
    for df in (tr, te):
        df[FORECAST_COLS] = df[FORECAST_COLS].mask(df[FORECAST_COLS].isin(SENTINELS))

    # Interpolation: fraction of points where the 2nd difference ~ 0 (linear interpolation leaves exactly
    # straight segments), and whether changes concentrate on specific hours (e.g. every 3 h).
    say("interpolation check (on de-duplicated, hourly-contiguous series):")
    for name, df in [("train", tr), ("test", te)]:
        s = df.set_index("ts").sort_index()
        s = s[~s.index.duplicated()]
        s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="h"))
        for c in FORECAST_COLS:
            v = s[c]
            step = {"forecast_cloud_cover": 0.001, "forecast_temp_c": 0.1,
                    "forecast_wind_ms": 0.01, "forecast_humidity_pct": 1}[c]
            d1 = v.diff()
            d2 = d1.diff()
            lin = (d2.abs() <= step * 1.01) & d1.notna() & d2.notna()
            unch = d1.abs() < 1e-9
            by_hour = d1.abs().groupby(v.index.hour).mean()
            ratio = by_hour.max() / max(by_hour.min(), 1e-9)
            say(f"  {name:5s} {c:24s} 2nd-diff~0: {lin.mean():6.1%}   unchanged vs prev hour: {unch.mean():6.1%}   "
                f"mean|d1| by hour max/min: {ratio:4.1f} (peak hour {by_hour.idxmax()})")
    # Linear interpolation from 3-hourly data would make the 2nd diff ~0 at 2 of every 3 hours, i.e. a
    # strong hour-mod-3 pattern. Show the share of "straight" points by hour-of-day for train.
    s = tr.set_index("ts").sort_index()
    s = s[~s.index.duplicated()].reindex(pd.date_range(tr.ts.min(), tr.ts.max(), freq="h"))
    lin_by_hour = {}
    for c, step in [("forecast_cloud_cover", 0.001), ("forecast_temp_c", 0.1),
                    ("forecast_wind_ms", 0.01), ("forecast_humidity_pct", 1)]:
        d2 = s[c].diff().diff().shift(-1)  # curvature centred on hour t
        lin_by_hour[c] = (d2.abs() <= step * 1.01).groupby(s.index.hour).mean()
    lin_by_hour = pd.DataFrame(lin_by_hour)
    say("share of hours lying on a straight line with neighbours, by hour-of-day (train):")
    say(lin_by_hour.T.round(2).to_string())
    save_table(lin_by_hour.round(3), "forecast_linear_share_by_hour.csv")

    # Where do forecast changes jump? A day-ahead product issued once per day can show a step at the
    # boundary between issue cycles (midnight). Check |d1| at 00:00 vs other hours.
    s = tr.set_index("ts").sort_index()
    s = s[~s.index.duplicated()]
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="h"))
    jump = pd.DataFrame({c: s[c].diff().abs().groupby(s.index.hour).mean() for c in FORECAST_COLS})
    save_table(jump.round(4), "forecast_mean_abs_hourly_change_by_hour.csv")
    say("mean |hour-to-hour change| by hour-of-day (train; row h = change from h-1 to h):")
    say(jump.round(3).to_string())

    # Alignment of forecasts with measurements (lag with max correlation)
    say("lag (h) maximising correlation between forecast and on-site measurement (positive = forecast lags):")
    meas_amb = s.measured_ambient_temp_c.where(~s.measured_ambient_temp_c.isin(SENTINELS))
    cs = tr.set_index("ts")["cs_poa"]
    cs = cs[~cs.index.duplicated()].reindex(s.index)
    kt = (s.measured_poa_wm2.where(~s.measured_poa_wm2.isin(SENTINELS)) / cs).where(cs > 200)
    for lab, a, b in [("temp vs ambient", s.forecast_temp_c, meas_amb),
                      ("cloud vs (1 - POA/clearsky)", s.forecast_cloud_cover, 1 - kt.clip(0, 1.2))]:
        cc = {lag: a.shift(lag).corr(b) for lag in range(-4, 5)}
        best = max(cc, key=lambda k: cc[k])
        say(f"  {lab:30s} best lag {best:+d} h (r={cc[best]:.3f}); r at lag 0 = {cc[0]:.3f}")
    # monthly lag for temperature to detect periods of misalignment
    mm = []
    for m, idx in s.groupby(s.index.to_period("M")).groups.items():
        a = s.forecast_temp_c.loc[idx]
        b = meas_amb.loc[idx]
        cc = {lag: a.shift(lag).corr(b) for lag in range(-4, 5)}
        mm.append((str(m), max(cc, key=lambda k: cc[k]), cc[0]))
    say("  monthly best lag forecast_temp vs measured ambient:",
        {m: l for m, l, _ in mm})

    # Cloud forecast skill vs measured clearness (daylight)
    day = kt.notna()
    say(f"  corr(forecast cloud, measured clearness) daylight: "
        f"{s.forecast_cloud_cover[day].corr(kt[day]):.3f}")

    # Train vs test distribution: same calendar months
    jj = tr[tr.ts.dt.month.isin([7, 8])]
    comp = pd.DataFrame({"train Jul-Aug 2024 mean": jj[FORECAST_COLS].mean(),
                         "test Jul-Aug 2025 mean": te[FORECAST_COLS].mean(),
                         "train Jul-Aug 2024 std": jj[FORECAST_COLS].std(),
                         "test Jul-Aug 2025 std": te[FORECAST_COLS].std()})
    say("train Jul-Aug 2024 vs test Jul-Aug 2025:")
    say(comp.round(3).to_string())
    save_table(comp, "forecast_train_vs_test.csv")

    fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
    for ax, c in zip(axes, FORECAST_COLS):
        ax.plot(tr.ts, tr[c], ",", label="train")
        ax.plot(te.ts, te[c], ",", color="C1", label="test")
        ax.set_ylabel(c.replace("forecast_", ""))
    axes[0].set_title("Forecast columns over time (train blue, test orange)")
    savefig(fig, "forecast_timeseries.png")

    # zoom a week to eyeball smoothness
    wk = s.loc["2024-07-01":"2024-07-07"]
    fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
    for ax, c in zip(axes, FORECAST_COLS):
        ax.plot(wk.index, wk[c], ".-", ms=4)
        ax.set_ylabel(c.replace("forecast_", ""))
    axes[0].set_title("Forecast columns, one week, hourly points")
    savefig(fig, "forecast_week_zoom.png")


def main():
    tr_raw = load(DATA / "train.csv")
    te_raw = load(DATA / "test.csv")

    audit_timestamps(tr_raw, "train", "2024-01-01 00:00", "2025-06-30 23:00")
    audit_timestamps(te_raw, "test", "2025-07-01 00:00", "2025-08-31 23:00")

    # For the value checks: sort by time and keep the first copy of each duplicated hour.
    # The actual de-duplication rule is in src/clean.py.
    tr = tr_raw.sort_values("ts", kind="stable").drop_duplicates("ts", keep="first").reset_index(drop=True)
    te = te_raw.sort_values("ts", kind="stable").drop_duplicates("ts", keep="first").reset_index(drop=True)
    geo = clearsky_poa(tr["ts"])
    tr["elevation"] = geo["elevation"].values
    tr["cs_poa"] = geo["cs_poa"].values

    audit_status(tr)
    audit_power(tr)
    audit_diurnal(tr)
    audit_sensors(tr)
    audit_ratio(tr)
    audit_forecasts(tr, te)

    (REPORTS / "audit_summary.txt").write_text("\n".join(_summary), encoding="utf-8")
    print("\nwrote reports/audit_summary.txt")


if __name__ == "__main__":
    main()
