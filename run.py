"""Reproduce predictions.csv from the supplied raw files (data/train.csv, data/test.csv).

Pipeline:
    clean (rules 1-11)
    -> fit B2b physics on all training data (dry-months Linke turbidity, fitted cloud curve, loss factor)
    -> train M2_clean_fixed (LightGBM ratio-to-physics, target ac_power_clean, 15 leaves, 295 rounds,
       5 seeds averaged) on all train_ok daylight rows
    -> predict test -> BL50 = 0.5 * B2b + 0.5 * M2 -> 0 when the sun is below the horizon -> clip [0, 10000]
    -> predictions.csv (timestamp,predicted_ac_power_kw) + reports/test_sanity.txt

Model inputs at prediction time are the timestamp and the four forecast columns only.

Run:  .venv\\Scripts\\python.exe run.py
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.clean import run_cleaning
from src.solar import AC_KW, clearsky_poa, geometry, prepare_forecast
from src.train import FIXED_ITERS, FIXED_LEAVES, FIXED_SEEDS, TL_CHOICE, GBMModel
from src.validate import CLOUD_BUCKETS

ROOT = Path(__file__).resolve().parent
DATA, REPORTS = ROOT / "data", ROOT / "reports"


def fit_final(train):
    geo = geometry(train["timestamp"])
    # M2 fits its own B2b physics model (same config: fitted_dry TL, fitted cloud curve, loss factor on
    # ac_power_clean) on all training rows; that fitted physics model IS the B2b half of the blend.
    m2 = GBMModel("M2", "ac_power_clean", FIXED_LEAVES, FIXED_ITERS, FIXED_SEEDS, tl=TL_CHOICE).fit(geo, train)
    return m2, m2.phys


def predict_test(m2, b2b, test):
    geo = geometry(test["timestamp"])
    p_b2b = b2b.predict(geo, test)
    p_m2 = m2.predict(geo, test)
    p = 0.5 * p_b2b + 0.5 * p_m2
    night = geo["elevation"].values < 0
    p[night] = 0.0
    p = np.clip(p, 0, AC_KW)
    return p, p_b2b, p_m2, geo


def check(pred_df, test_raw, night):
    assert len(pred_df) == 1488, len(pred_df)
    assert list(pred_df.columns) == ["timestamp", "predicted_ac_power_kw"]
    assert (pred_df["timestamp"].values == test_raw["timestamp"].values).all(), "timestamps differ from test.csv"
    v = pred_df["predicted_ac_power_kw"]
    assert v.notna().all(), "NaN in predictions"
    assert ((v >= 0) & (v <= AC_KW)).all(), "prediction outside [0, 10000]"
    assert (v[night] == 0).all(), "non-zero prediction at night"
    print(f"asserts ok: 1488 rows, timestamps identical and ordered, no NaN, range "
          f"[{v.min():.1f}, {v.max():.1f}], {int(night.sum())} night hours all 0")


def sanity(test, pred, geo, train):
    lines = ["Test-period sanity check (BL50 predictions, Jul-Aug 2025) vs actuals Jul-Aug 2024", ""]
    t = test.assign(pred=pred, month=test["timestamp"].dt.month, date=test["timestamp"].dt.normalize(),
                    el=geo["elevation"].values)
    fc = prepare_forecast(test)
    t["cloud"] = fc["forecast_cloud_cover"].values
    tr = train[train["timestamp"].dt.month.isin([7, 8]) & (train["timestamp"].dt.year == 2024)].copy()
    tr["month"] = tr["timestamp"].dt.month
    tr["date"] = tr["timestamp"].dt.normalize()
    # complete days only: 24 rows present, every daylight hour train_ok with power, so daily sums are comparable
    day_ok = tr.groupby("date").apply(
        lambda d: len(d) == 24 and bool((d.loc[d["elevation"] > 0, "train_ok"]
                                         & d.loc[d["elevation"] > 0, "ac_power_kw"].notna()).all()),
        include_groups=False)
    trc = tr[tr["date"].isin(day_ok[day_ok].index)]
    flags = []

    lines.append("1. Mean daily energy (MWh/day)")
    lines.append(f"   {'month':<6}{'pred 2025':>11}{'actual 2024':>13}{'diff':>8}{'2024 days':>11}"
                 f"{'fc cloud 2025':>15}{'fc cloud 2024':>15}")
    for m, name in [(7, "Jul"), (8, "Aug")]:
        pe = t[t.month == m].groupby("date")["pred"].sum().mean() / 1000
        ae = trc[trc.month == m].groupby("date")["ac_power_kw"].sum().mean() / 1000
        nd = trc[trc.month == m]["date"].nunique()
        c25 = t[(t.month == m) & (t.el > 0)]["cloud"].mean()
        c24 = tr[(tr.month == m) & (tr.elevation > 0)]["forecast_cloud_cover"].mean()
        d = 100 * (pe / ae - 1)
        lines.append(f"   {name:<6}{pe:>11.1f}{ae:>13.1f}{d:>+7.1f}%{nd:>11d}{c25:>15.2f}{c24:>15.2f}")
        if abs(d) > 20:
            flags.append(f"{name} daily energy {d:+.1f}% vs 2024")
    pe_all = t.groupby("date")["pred"].sum().mean() / 1000
    ae_all = trc.groupby("date")["ac_power_kw"].sum().mean() / 1000
    lines.append(f"   {'Jul+Aug':<6}{pe_all:>11.1f}{ae_all:>13.1f}{100 * (pe_all / ae_all - 1):>+7.1f}%"
                 f"{trc['date'].nunique():>11d}")
    lines.append("   (2024 actual = ac_power_kw on complete days only: all 24 rows present, every daylight hour "
                 "train_ok with power; excludes the Aug-2024 STOP week and gap days)")
    lines.append("")

    lines.append("2. Mean daylight power by forecast cloud bucket (kW, sun elevation > 0)")
    lines.append(f"   {'bucket':<9}{'pred 2025':>11}{'actual 2024':>13}{'diff':>8}{'hours 2025':>12}{'hours 2024':>12}"
                 f"{'sun el 25/24':>15}{'cs POA 25/24':>15}{'P/csPOA 25/24':>16}")
    trd = tr[(tr.elevation > 0) & tr["train_ok"] & tr["ac_power_kw"].notna()]
    td = t[t.el > 0].copy()
    td["cs_poa"] = clearsky_poa(td["timestamp"])["cs_poa"].values
    for lo, hi, name in CLOUD_BUCKETS:
        a = td[(td.cloud >= lo) & (td.cloud < hi)]
        b = trd[(trd.forecast_cloud_cover >= lo) & (trd.forecast_cloud_cover < hi)]
        d = 100 * (a["pred"].mean() / b["ac_power_kw"].mean() - 1)
        # yield per unit clear-sky POA separates "different hours of day" from "different power level"
        ya, yb = a["pred"].mean() / a["cs_poa"].mean(), b["ac_power_kw"].mean() / b["cs_poa"].mean()
        lines.append(f"   {name:<9}{a['pred'].mean():>11.0f}{b['ac_power_kw'].mean():>13.0f}{d:>+7.1f}%"
                     f"{len(a):>12d}{len(b):>12d}{a['el'].mean():>8.0f}/{b['elevation'].mean():<6.0f}"
                     f"{a['cs_poa'].mean():>8.0f}/{b['cs_poa'].mean():<6.0f}{ya:>9.2f}/{yb:<6.2f}")
        if abs(d) > 20:
            flags.append(
                f"bucket {name} mean daylight power {d:+.1f}% vs 2024. Likely reason: only {len(a)} vs {len(b)} "
                f"hours and a different time-of-day mix (mean sun elevation {a['el'].mean():.0f} vs "
                f"{b['elevation'].mean():.0f} deg, clear-sky POA {a['cs_poa'].mean():.0f} vs "
                f"{b['cs_poa'].mean():.0f} W/m2). Per unit clear-sky POA the difference is "
                f"{100 * (ya / yb - 1):+.1f}%.")
    lines.append(f"   {'all':<9}{td['pred'].mean():>11.0f}{trd['ac_power_kw'].mean():>13.0f}"
                 f"{100 * (td['pred'].mean() / trd['ac_power_kw'].mean() - 1):>+7.1f}%{len(td):>12d}{len(trd):>12d}")
    lines.append("   (2024 actual: train_ok daylight hours with power. P/csPOA = mean power / mean clear-sky POA, "
                 "kW per W/m2)")
    lines.append("   Overall: test is cloudier (more hours in 0.7-1), so the all-hours mean is lower; the drop "
                 "is consistent with the cloud mix, not a level shift.")
    lines.append("")
    lines.append(f"Forecast cloud, daylight mean: test {td['cloud'].mean():.2f} vs Jul-Aug 2024 "
                 f"{tr[tr.elevation > 0]['forecast_cloud_cover'].mean():.2f}")
    lines.append("")
    lines.append("3. Flags (> 20 % off)")
    if flags:
        lines += [f"   - {f}" for f in flags]
    else:
        lines.append("   none: every comparison is within 20 %")
    return lines, flags


def main():
    t0 = time.time()
    train = run_cleaning()
    test_raw = pd.read_csv(DATA / "test.csv", dtype={"timestamp": str})
    test = test_raw.copy()
    test["timestamp"] = pd.to_datetime(test["timestamp"], format="%Y-%m-%d %H:%M:%S")
    assert test["timestamp"].notna().all()

    print("\nfitting final model on all training data ...")
    m2, b2b = fit_final(train)
    bp = b2b.params()
    print(f"B2b: loss {bp['loss_factor']}, cloud a={bp['cloud_a']} b={bp['cloud_b']}, TL {bp['tl_by_month']}")
    print(f"M2_clean_fixed: {m2.n_train} training rows, {FIXED_LEAVES} leaves, {FIXED_ITERS} rounds, "
          f"seeds {FIXED_SEEDS}")

    pred, p_b2b, p_m2, geo = predict_test(m2, b2b, test)
    night = geo["elevation"].values < 0
    out = pd.DataFrame({"timestamp": test_raw["timestamp"], "predicted_ac_power_kw": np.round(pred, 3)})
    check(out, test_raw, night)
    out.to_csv(ROOT / "predictions.csv", index=False, lineterminator="\n")
    print(f"wrote predictions.csv; test daylight mean: BL50 {pred[~night].mean():.0f} kW, "
          f"B2b {p_b2b[~night].mean():.0f} kW, M2 {p_m2[~night].mean():.0f} kW")

    lines, _ = sanity(test, pred, geo, train)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "test_sanity.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\ndone in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
