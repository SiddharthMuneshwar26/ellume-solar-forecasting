"""Cleaning rules for data/train.csv -> data/train_clean.parquet.

One function per rule. Each function says why the rule exists and prints how many rows it touched.
Rows are never deleted: bad values become NaN and unusable hours get flags (`train_ok` = False,
`excl_*` reason columns). The only rows removed are the surplus copies of duplicated hours (rule 2),
and those are counted explicitly.

Run:  .venv\\Scripts\\python.exe -m src.clean
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.solar import AC_KW, TEMP_COEFF, clearsky_poa

ROOT = Path(__file__).resolve().parents[1]
DATA, REPORTS = ROOT / "data", ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

KW_PER_WM2 = 10.8
SENTINELS = [-999.0, 9999.0]
NUMERIC_COLS = ["forecast_cloud_cover", "forecast_temp_c", "forecast_wind_ms", "forecast_humidity_pct",
                "measured_poa_wm2", "measured_module_temp_c", "measured_ambient_temp_c", "rainfall_mm",
                "ac_power_kw"]
MW_START, MW_END = pd.Timestamp("2024-07-14 06:00"), pd.Timestamp("2024-08-08 18:00")
FLAT_POWER = (pd.Timestamp("2024-09-18 00:00"), pd.Timestamp("2024-09-19 23:00"), 2187.44)
FLAT_POA = (pd.Timestamp("2024-08-25 00:00"), pd.Timestamp("2024-08-27 23:00"), 412.6)


def log(rule, msg):
    print(f"[{rule}] {msg}")


def expected_power(df):
    """Expected AC power from measured POA and module temperature, capped at the inverter rating.

    Used only for cleaning / diagnosis, never as a model feature (measured data is not available at forecast time).
    """
    tcorr = 1 + TEMP_COEFF * (df["measured_module_temp_c"] - 25)
    return (KW_PER_WM2 * df["measured_poa_wm2"] * tcorr).clip(upper=AC_KW)

def parse_and_sort(df):


    iso = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    dmy = pd.to_datetime(df["timestamp"], format="%d-%m-%Y %H:%M", errors="coerce")
    df = df.copy()
    df["timestamp"] = iso.fillna(dmy)
    n_bad = int(df["timestamp"].isna().sum())
    assert n_bad == 0, f"{n_bad} unparseable timestamps"
    out_of_order = int((df["timestamp"].diff() < pd.Timedelta(0)).sum())
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    log(1, f"parsed {int(iso.notna().sum())} ISO + {int((iso.isna() & dmy.notna()).sum())} DD-MM-YYYY timestamps; "
           f"{out_of_order} ordering break(s) fixed by sorting")
    return df

def replace_sentinels(df):


    df = df.copy()
    for c in NUMERIC_COLS:
        m = df[c].isin(SENTINELS)
        if m.any():
            log(3, f"{c}: {int(m.sum())} sentinel(s) -> NaN")
        df.loc[m, c] = np.nan
    return df

def resolve_duplicates(df):




    dup = df["timestamp"].duplicated(keep=False)
    value_cols = [c for c in df.columns if c != "timestamp"]
    n_hours = df.loc[dup, "timestamp"].nunique()
    elev = clearsky_poa(df.loc[dup, "timestamp"])["elevation"]
    others = [c for c in value_cols if c != "ac_power_kw"]
    keep_idx, n_agree, n_pick, n_mean, n_night, n_nan = [], 0, 0, 0, 0, 0
    conflicts = []
    for ts, g in df[dup].groupby("timestamp"):
        assert g[others].nunique(dropna=False).max() <= 1, f"duplicates at {ts} disagree outside ac_power_kw"
        if g["ac_power_kw"].nunique(dropna=False) <= 1:
            keep_idx.append(g.index[0])
            n_agree += 1
            continue
        exp = expected_power(g.iloc[[0]]).iloc[0]
        ok = (g["ac_power_kw"] - exp).abs() <= 0.10 * exp if pd.notna(exp) else pd.Series(False, index=g.index)
        vals = g["ac_power_kw"].round(2).tolist()
        exp_s = None if pd.isna(exp) else round(exp, 1)
        keep_idx.append(ok.idxmax() if ok.sum() == 1 else g.index[0])
        if ok.sum() == 1:
            n_pick += 1
            action = f"kept {g.loc[ok.idxmax(), 'ac_power_kw']:.2f} (one within 10 %)"
        elif ok.sum() == 2:
            df.loc[g.index[0], "ac_power_kw"] = g["ac_power_kw"].mean()
            n_mean += 1
            action = f"mean {g['ac_power_kw'].mean():.2f} (both within 10 %)"
        elif elev.loc[ts].iloc[0] < 0:
            df.loc[g.index[0], "ac_power_kw"] = 0.0
            n_night += 1
            action = "0 (night)"
        else:
            df.loc[g.index[0], "ac_power_kw"] = np.nan
            n_nan += 1
            action = "NaN (neither within 10 %, daytime)"
        conflicts.append((ts, vals, exp_s, action))
    drop = df.index[dup].difference(keep_idx)
    out = df.drop(index=drop).reset_index(drop=True)
    log(2, f"{n_hours} duplicated hours: {n_agree} identical copies, {n_pick} one-copy-within-10%, "
           f"{n_mean} mean of both, {n_night} night -> 0, {n_nan} set to NaN; "
           f"{len(drop)} surplus rows removed ({len(df)} -> {len(out)})")
    for ts, vals, exp, action in conflicts:
        log(2, f"    {ts}  copies {vals}  expected {exp}  -> {action}")
    return out

def fix_mw_window(df):



    df = df.copy()
    win = df["timestamp"].between(MW_START, MW_END)
    pos = win & (df["ac_power_kw"] > 0)
    assert not (pos & (df["ac_power_kw"] > 50)).any(), "kW-sized positive value inside the MW window"
    df.loc[pos, "ac_power_kw"] *= 1000
    ratio = df["ac_power_kw"] / df["measured_poa_wm2"].where(df["measured_poa_wm2"].between(400, 1300))
    log(4, f"{int(pos.sum())} positive values in window x1000; median power/POA inside {ratio[win].median():.2f}, "
           f"outside {ratio[~win].median():.2f} kW per W/m2")
    return df

def fix_over_rating(df):



    df = df.copy()
    spike = df["ac_power_kw"] > 10_100
    over = df["ac_power_kw"].between(AC_KW, 10_100, inclusive="right")
    df.loc[spike, "ac_power_kw"] = np.nan
    df.loc[over, "ac_power_kw"] = AC_KW
    log(5, f"{int(spike.sum())} spikes > 10,100 kW -> NaN; {int(over.sum())} values in (10,000, 10,100] -> 10,000")
    return df

def fix_flatlines(df):



    df = df.copy()
    s, e, v = FLAT_POWER
    m = df["timestamp"].between(s, e) & np.isclose(df["ac_power_kw"], v)
    df.loc[m, "ac_power_kw"] = np.nan
    log(6, f"power flatline {v} kW {s}..{e}: {int(m.sum())} rows -> NaN")
    s, e, v = FLAT_POA
    m = df["timestamp"].between(s, e) & np.isclose(df["measured_poa_wm2"], v)
    df.loc[m, "measured_poa_wm2"] = np.nan
    log(6, f"POA flatline {v} W/m2 {s}..{e}: {int(m.sum())} rows -> measured_poa_wm2 NaN (power kept)")
    return df

def zero_night_negatives(df):


    df = df.copy()
    m = (df["elevation"] < 0) & (df["ac_power_kw"] < 0)
    df.loc[m, "ac_power_kw"] = 0.0
    log(7, f"{int(m.sum())} night-time negative values -> 0")
    return df

RATIO_SPIKE, RATIO_MIN_POA = 1.5, 100


def fix_ratio_spikes(df):


    df = df.copy()
    exp = expected_power(df)
    ratio = df["ac_power_kw"] / exp
    poa = df["measured_poa_wm2"]
    t = df["timestamp"]
    prev_ok = t.diff().eq(pd.Timedelta("1h"))
    next_ok = t.diff(-1).eq(pd.Timedelta("-1h"))
    nb = pd.DataFrame({
        "prev_kw": df["ac_power_kw"].shift(1).where(prev_ok), "prev_poa": poa.shift(1).where(prev_ok),
        "prev_ratio": ratio.shift(1).where(prev_ok),
        "next_kw": df["ac_power_kw"].shift(-1).where(next_ok), "next_poa": poa.shift(-1).where(next_ok),
        "next_ratio": ratio.shift(-1).where(next_ok),
    })
    hit = (poa > RATIO_MIN_POA) & (ratio > RATIO_SPIKE)
    isolated = hit & ~(nb["prev_ratio"] > RATIO_SPIKE) & ~(nb["next_ratio"] > RATIO_SPIKE)
    rep = pd.concat([df[["timestamp", "status", "elevation", "measured_poa_wm2", "measured_module_temp_c"]],
                     exp.rename("expected_kw"), df["ac_power_kw"], ratio.rename("ratio"), nb], axis=1)[hit]
    rep = rep.assign(isolated=isolated[hit])
    rep.to_csv(REPORTS / "ratio_spikes.csv", index=False)
    n200 = int((hit & (poa > 200)).sum())
    log(11, f"power / expected > {RATIO_SPIKE} with POA > {RATIO_MIN_POA}: {int(hit.sum())} rows "
            f"({int(isolated.sum())} isolated single hours; {n200} would remain with POA > 200) "
            f"-> ac_power_kw NaN; listed in reports/ratio_spikes.csv")
    df.loc[hit, "ac_power_kw"] = np.nan
    return df

def build_flags(df):



    df = df.copy()
    raw = df["status"]
    df["status"] = raw.str.strip().str.upper()
    log(8, f"status normalised: {int((raw.notna() & (raw != df['status'])).sum())} values changed; "
           f"counts {df['status'].value_counts(dropna=False).to_dict()}")
    df["expected_kw"] = expected_power(df)
    act_exp = df["ac_power_kw"] / df["expected_kw"]
    poa300 = df["measured_poa_wm2"] > 300
    low = poa300 & (act_exp < 0.5)

    df["excl_stop"] = df["status"].eq("STOP")
    df["excl_partial"] = df["status"].eq("PARTIAL")
    df["excl_power_nan"] = df["ac_power_kw"].isna()
    df["excl_day_standby"] = df["status"].eq("STANDBY") & (df["elevation"] > 0) & poa300
    df["excl_nan_status_low"] = df["status"].isna() & low
    excl = [c for c in df.columns if c.startswith("excl_")]
    df["train_ok"] = ~df[excl].any(axis=1)
    for c in excl:
        log(8, f"{c}: {int(df[c].sum())}")

    # RUN hours producing < 50 % of expected with decent irradiance: unreported curtailment/outage, or a
    # local cloud over the pyranometer. Flagged for review, but kept in training.
    df["suspect_outage"] = df["status"].eq("RUN") & low
    cols = ["timestamp", "status", "measured_poa_wm2", "measured_module_temp_c", "expected_kw", "ac_power_kw"]
    so = df.loc[df["suspect_outage"], cols].assign(actual_over_expected=act_exp[df["suspect_outage"]])
    so.to_csv(REPORTS / "suspect_outage.csv", index=False)
    log(8, f"suspect_outage (RUN, POA>300, actual/expected<0.5): {len(so)} "
           f"-> reports/suspect_outage.csv (not excluded)")
    return df

def add_soiling(df):




    df = df.copy()
    tcorr = 1 + TEMP_COEFF * (df["measured_module_temp_c"] - 25)
    use = (df["train_ok"] & df["status"].eq("RUN") & df["measured_poa_wm2"].between(400, 1300)
           & df["ac_power_kw"].between(1000, 9000))
    r = (df["ac_power_kw"] / (df["measured_poa_wm2"] * tcorr))[use]
    date = df["timestamp"].dt.normalize()
    daily = r.groupby(date[use]).agg(["median", "size"])
    daily = daily.loc[daily["size"] >= 3, "median"]
    baseline = daily[daily.index.month.isin([6, 7, 8, 9, 10])].median()

    smooth = (daily / baseline).rolling(7, center=True, min_periods=3).median()
    all_days = pd.date_range(date.min(), date.max(), freq="D")
    factor = smooth.reindex(all_days).interpolate(method="time").ffill().bfill().clip(upper=1.0)
    df["soiling_factor"] = date.map(factor).values


    n_over = int((df["ac_power_kw"] / df["soiling_factor"] > AC_KW).sum())
    df["ac_power_clean"] = (df["ac_power_kw"] / df["soiling_factor"]).clip(upper=AC_KW)
    log(9, f"{len(daily)} usable days; clean baseline (Jun-Oct median) {baseline:.3f} kW per W/m2; "
           f"soiling_factor range {factor.min():.3f}..{factor.max():.3f}, "
           f"{int((factor < 1).sum())} of {len(factor)} days < 1")
    log(9, f"ac_power_kw / soiling_factor > 10,000 kW on {n_over} rows -> ac_power_clean capped at 10,000")

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(daily.index, daily / baseline, ".", ms=3, alpha=0.5, label="daily ratio / baseline")
    ax.plot(factor.index, factor.values, lw=1.4, label="soiling_factor (7-day median, capped at 1)")
    ax.axhline(1, color="k", lw=0.5)
    ax.set_ylim(0.85, 1.1)
    ax.set_ylabel("relative performance")
    ax.set_title(f"Soiling factor (clean baseline = Jun-Oct median {baseline:.2f} kW per W/m2)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "soiling_factor.png", dpi=110)
    plt.close(fig)
    return df


def summary(df, n_raw):
    excl = [c for c in df.columns if c.startswith("excl_")]
    rows = [("raw rows in train.csv", n_raw), ("rows after duplicate resolution", len(df)),
            ("missing hours (left missing, rule 10)", 13_128 - len(df)),
            ("train_ok = True", int(df["train_ok"].sum())), ("train_ok = False", int((~df["train_ok"]).sum()))]
    rows += [(f"  {c}", int(df[c].sum())) for c in excl]
    rows += [("  rows with >1 reason", int((df[excl].sum(axis=1) > 1).sum())),
             ("suspect_outage (kept)", int(df["suspect_outage"].sum())),
             ("train_ok & daylight (el>0)", int((df["train_ok"] & (df["elevation"] > 0)).sum()))]
    t = pd.DataFrame(rows, columns=["item", "count"])
    print("\n" + t.to_string(index=False))
    return t


def main():
    return run_cleaning()


def run_cleaning():
    """All cleaning rules on data/train.csv; writes data/train_clean.parquet and returns the cleaned frame."""
    raw = pd.read_csv(DATA / "train.csv", dtype={"status": str})
    n_raw = len(raw)
    df = parse_and_sort(raw)
    df = replace_sentinels(df)
    df = resolve_duplicates(df)
    df["ac_power_raw"] = df["ac_power_kw"]
    geo = clearsky_poa(df["timestamp"])
    df["elevation"] = geo["elevation"].values
    df["cs_poa"] = geo["cs_poa"].values
    df = fix_mw_window(df)
    df = fix_over_rating(df)
    df = fix_flatlines(df)
    df = zero_night_negatives(df)
    df = fix_ratio_spikes(df)
    df = build_flags(df)
    df = add_soiling(df)
    # Rule 10: the 186 missing hours stay missing; nothing is reindexed or imputed.
    assert df["timestamp"].is_unique and df["timestamp"].is_monotonic_increasing
    summary(df, n_raw).to_csv(REPORTS / "clean_summary.csv", index=False)
    df.to_parquet(DATA / "train_clean.parquet", index=False)
    print(f"\nwrote data/train_clean.parquet ({len(df)} rows, {df.shape[1]} columns)")
    return df


if __name__ == "__main__":
    main()
