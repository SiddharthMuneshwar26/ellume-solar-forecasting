from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.solar import AC_KW, DC_KWP, cloud_factor, geometry, prepare_forecast, temp_factor
from src.train import FIXED_ITERS, FIXED_LEAVES, FIXED_SEEDS, MONSOON_FOLDS, TL_CHOICE, GBMModel
from src.validate import CLOUD_BUCKETS, FOLDS, evaluate, scoring_mask

ROOT = Path(__file__).resolve().parents[1]
DATA, REPORTS = ROOT / "data", ROOT / "reports"
OUT_DAYS = REPORTS / "worst_days"
FORECAST_WRONG_THRESHOLD = 0.3   # mean |measured kt - forecast-implied kt| over daylight hours
KT_MIN_ELEVATION = 10            # deg; clearness is noisy at low sun


def bl50_fold(df, geo, fold, tl=TL_CHOICE):
    """Fit B2b and M2_clean_fixed on the fold's training rows; return test rows with predictions."""
    tr_m, te_m = fold.train_mask(df).values, fold.test_mask(df).values
    tr, te = df[tr_m].reset_index(drop=True), df[te_m].reset_index(drop=True)
    m2 = GBMModel("M2", "ac_power_clean", FIXED_LEAVES, FIXED_ITERS, FIXED_SEEDS, tl=tl).fit(geo[tr_m], tr)
    b2b = m2.phys  # the B2b model M2 is built on (same config, same training rows)
    p_m2, X = m2.predict(geo[te_m], te, return_features=True)
    out = te.copy()
    out["B2b"] = b2b.predict(geo[te_m], te)
    out["M2_clean_fixed"] = p_m2
    out["BL50"] = 0.5 * out["B2b"] + 0.5 * out["M2_clean_fixed"]
    out["cloud_s1"] = X["cloud_s1"].values
    out["cs_poa_model"] = X["cs_poa"].values
    # measured clearness vs the clearness the forecast implies through the fitted cloud curve
    ok = (out["elevation"] > KT_MIN_ELEVATION) & (out["cs_poa_model"] > 50)
    out["kt_meas"] = (out["measured_poa_wm2"] / out["cs_poa_model"]).where(ok).clip(0, 1.3)
    out["kt_fcst"] = pd.Series(cloud_factor(out["forecast_cloud_cover"].values, *b2b.ab)).where(ok)
    # "oracle irradiance": the same B2b power chain driven by the MEASURED POA (diagnosis only). If this matches
    # actual power, the day's error came from the irradiance forecast; if not, from the power model.
    poa = out["measured_poa_wm2"].values
    tf = temp_factor(poa, out["forecast_temp_c"].values, out["forecast_wind_ms"].values)
    oracle = np.clip(DC_KWP * poa / 1000 * tf * b2b.loss, 0, AC_KW)
    oracle[out["elevation"].values < 0] = 0.0
    out["oracle_irr"] = oracle
    out["fold"] = fold.name
    return out, b2b


def sensitivity(df, geo):
    print("\n=== 1. Sensitivity (not for selection): BL50 turbidity, MAE % vs ac_power_kw ===")
    rows = []
    for fold in [f for f in FOLDS if f.name in MONSOON_FOLDS]:
        for tl in (TL_CHOICE, "default"):
            te, _ = bl50_fold(df, geo, fold, tl)
            for m in ("B2b", "M2_clean_fixed", "BL50"):
                r = evaluate(te, te[m].values, "ac_power_kw")[0]
                rows.append({"fold": fold.name, "tl": tl, "model": m, "mae_pct": r["mae_pct"],
                             "rmse_pct": r["rmse_pct"], "bias_kw": r["bias_kw"]})
    s = pd.DataFrame(rows)
    t = s.pivot_table(index=["model", "tl"], columns="fold", values="mae_pct")
    t["monsoon mean"] = t[MONSOON_FOLDS].mean(axis=1)
    b = s.pivot_table(index=["model", "tl"], columns="fold", values="bias_kw")
    t["bias mean kW"] = b[MONSOON_FOLDS].mean(axis=1)
    print(t.round(2).to_string())
    s.to_csv(REPORTS / "sensitivity_turbidity.csv", index=False)
    return t


def expected_test_mae(preds):
    print("\n=== 2. Expected test MAE ===")
    te = pd.read_csv(DATA / "test.csv", parse_dates=["timestamp"])
    g = geometry(te["timestamp"])
    c = prepare_forecast(te)["forecast_cloud_cover"].values[g["elevation"].values > 0]
    rows = []
    for lo, hi, name in CLOUD_BUCKETS:
        share_test = float(np.mean((c >= lo) & (c < hi)))
        per_fold, per_fold_bias = [], []
        n_hours = 0
        for f, d in preds.groupby("fold"):
            m = scoring_mask(d, "ac_power_kw")
            cc = d["forecast_cloud_cover"].values[m]
            hb = (cc >= lo) & (cc < hi)
            e = d["BL50"].values[m][hb] - d["ac_power_kw"].values[m][hb]
            per_fold.append(100 * np.abs(e).mean() / AC_KW if hb.any() else np.nan)
            per_fold_bias.append(e.mean() if hb.any() else np.nan)
            n_hours += int(hb.sum())
        allc = np.concatenate([d["forecast_cloud_cover"].values[scoring_mask(d, "ac_power_kw")]
                               for _, d in preds.groupby("fold")])
        rows.append({"bucket": name, "test_share": share_test,
                     "monsoon_folds_share": float(np.mean((allc >= lo) & (allc < hi))),
                     "bl50_mae_pct (mean F1,F2,F8)": np.nanmean(per_fold),
                     "bl50_bias_kw (mean F1,F2,F8)": np.nanmean(per_fold_bias), "n_hours_folds": n_hours})
    t = pd.DataFrame(rows).set_index("bucket")
    exp = float((t["test_share"] * t["bl50_mae_pct (mean F1,F2,F8)"]).sum())
    folds_mix = float((t["monsoon_folds_share"] * t["bl50_mae_pct (mean F1,F2,F8)"]).sum())
    exp_bias = float((t["test_share"] * t["bl50_bias_kw (mean F1,F2,F8)"]).sum())
    folds_bias = float((t["monsoon_folds_share"] * t["bl50_bias_kw (mean F1,F2,F8)"]).sum())
    print(t.round(3).to_string())
    print(f"test daylight hours: {len(c)}, mean forecast cloud {c.mean():.3f}")
    print(f"expected test MAE (test cloud mix): {exp:.2f} % of 10 MW; same bucket MAEs at the folds' own mix: "
          f"{folds_mix:.2f} %")
    print(f"expected test bias (test cloud mix): {exp_bias:+.0f} kW; at the folds' own mix: {folds_bias:+.0f} kW")
    # finer check inside the cloudy bucket: the test set is cloudier within 0.7-1 as well
    fine = []
    for lo, hi in [(0.7, 0.8), (0.8, 0.9), (0.9, 1.0001)]:
        m_all = []
        for _, d in preds.groupby("fold"):
            m = scoring_mask(d, "ac_power_kw")
            cc = d["forecast_cloud_cover"].values[m]
            hb = (cc >= lo) & (cc < hi)
            m_all.append(np.abs(d["BL50"].values[m][hb] - d["ac_power_kw"].values[m][hb]))
        e = np.concatenate(m_all)
        fine.append({"sub-bucket": f"{lo}-{min(hi, 1)}", "test_share": float(np.mean((c >= lo) & (c < hi))),
                     "bl50_mae_pct (pooled)": 100 * e.mean() / AC_KW, "n": len(e)})
    fine = pd.DataFrame(fine)
    print("inside 0.7-1:")
    print(fine.round(3).to_string(index=False))
    t.to_csv(REPORTS / "expected_test_mae.csv")
    return t, exp, folds_mix, fine, c.mean()


def classify_day(d):
    """Pre-stated rule: mean over daylight of |measured clearness - forecast-implied clearness| > 0.3."""
    dl = d["kt_meas"].notna() & d["kt_fcst"].notna()
    diff = (d.loc[dl, "kt_meas"] - d.loc[dl, "kt_fcst"])
    mad = float(diff.abs().mean()) if dl.any() else np.nan
    return mad, float(diff.mean()) if dl.any() else np.nan


def attribute_day(d):
    """Daily energy split: forecast part = BL50 - oracle, model part = oracle - actual (scored hours with POA)."""
    m = scoring_mask(d, "ac_power_kw") & d["measured_poa_wm2"].notna().values
    act, bl, orc = (d[c].values[m].sum() for c in ("ac_power_kw", "BL50", "oracle_irr"))
    return 100 * (bl - orc) / act, 100 * (orc - act) / act


def error_analysis(preds):
    print("\n=== 3. Error analysis, BL50 vs ac_power_kw, F1/F2/F8 ===")
    OUT_DAYS.mkdir(exist_ok=True)
    sc = preds[scoring_mask(preds, "ac_power_kw")].copy()
    sc["err"] = sc["BL50"] - sc["ac_power_kw"]
    sc["date"] = sc["timestamp"].dt.normalize()
    day = sc.groupby(["fold", "date"]).agg(actual_kwh=("ac_power_kw", "sum"), bl50_kwh=("BL50", "sum"),
                                            b2b_kwh=("B2b", "sum"), m2_kwh=("M2_clean_fixed", "sum"),
                                            n_hours=("err", "size"),
                                            mean_fc_cloud=("forecast_cloud_cover", "mean")).reset_index()
    day["err_kwh"] = day["bl50_kwh"] - day["actual_kwh"]
    day["abs_err_kwh"] = day["err_kwh"].abs()
    day["err_pct"] = 100 * day["err_kwh"] / day["actual_kwh"]
    # F1 and F2 share the same test period: keep each date once (the fold with the larger error)
    uniq = day.sort_values("abs_err_kwh", ascending=False).drop_duplicates("date")
    worst = uniq.head(5).copy()
    cls = []
    for _, w in worst.iterrows():
        d = preds[(preds["fold"] == w["fold"]) & (preds["timestamp"].dt.normalize() == w["date"])]
        mad, signed = classify_day(d)
        fc_part, model_part = attribute_day(d)
        cls.append((mad, signed, "forecast wrong" if mad > FORECAST_WRONG_THRESHOLD else "model wrong",
                    fc_part, model_part,
                    "forecast wrong" if abs(fc_part) > abs(model_part) else "model wrong"))
        plot_day(d, w, mad)
    worst["kt_mad"] = [c[0] for c in cls]
    worst["kt_meas_minus_fcst"] = [c[1] for c in cls]
    worst["class"] = [c[2] for c in cls]
    worst["err_pct_forecast_part"] = [c[3] for c in cls]
    worst["err_pct_model_part"] = [c[4] for c in cls]
    worst["class_oracle"] = [c[5] for c in cls]
    other = day[day["date"].isin(worst["date"]) & ~day.set_index(["fold", "date"]).index.isin(
        worst.set_index(["fold", "date"]).index)][["fold", "date", "err_kwh", "err_pct"]]
    cols = ["fold", "date", "n_hours", "mean_fc_cloud", "actual_kwh", "bl50_kwh", "b2b_kwh", "m2_kwh",
            "err_kwh", "err_pct", "kt_mad", "kt_meas_minus_fcst", "class", "err_pct_forecast_part",
            "err_pct_model_part", "class_oracle"]
    print("5 worst days (unique dates, by |daily energy error|):")
    print(worst[cols].to_string(index=False, float_format="%.2f"))
    if len(other):
        print("same dates in the other fold:")
        print(other.to_string(index=False, float_format="%.1f"))
    print("pre-stated rule counts:", worst["class"].value_counts().to_dict())
    print("oracle-irradiance attribution counts:", worst["class_oracle"].value_counts().to_dict())
    # all days (for context): how does the pre-stated rule behave?
    allcls, allkt = [], []
    for (f, dt), d in preds.assign(date=preds["timestamp"].dt.normalize()).groupby(["fold", "date"]):
        allcls.append(classify_day(d)[0])
        allkt.append(d["kt_fcst"].dropna().values)
    allcls, allkt = np.array(allcls), np.concatenate(allkt)
    print(f"context: share of ALL fold-days with kt MAD > {FORECAST_WRONG_THRESHOLD}: "
          f"{np.nanmean(allcls > FORECAST_WRONG_THRESHOLD):.1%} (median {np.nanmedian(allcls):.2f}, "
          f"max {np.nanmax(allcls):.2f}); forecast-implied clearness range {allkt.min():.2f}..{allkt.max():.2f}")

    hrs = sc.assign(abs_err=sc["err"].abs()).sort_values("abs_err", ascending=False).drop_duplicates("timestamp")
    hcols = ["fold", "timestamp", "forecast_cloud_cover", "kt_meas", "kt_fcst", "measured_poa_wm2",
             "ac_power_kw", "B2b", "M2_clean_fixed", "BL50", "oracle_irr", "err"]
    print("3 largest single-hour errors (unique hours):")
    print(hrs.head(3)[hcols].to_string(index=False, float_format="%.2f"))
    worst.to_csv(REPORTS / "worst_days.csv", index=False)
    hrs.head(3)[hcols].to_csv(REPORTS / "worst_hours.csv", index=False)
    day.to_csv(REPORTS / "daily_errors_monsoon_folds.csv", index=False)
    return worst, hrs.head(3)


def plot_day(d, w, mad):
    d = d[(d["elevation"] > -5)]
    t = d["timestamp"] + pd.Timedelta(minutes=30)  # plot at hour mid-point
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    ax[0].plot(t, d["ac_power_kw"].where(d["train_ok"]), "k-o", ms=3, lw=2, label="actual ac_power_kw")
    ax[0].plot(t, d["B2b"], "-", color="C0", label="B2b physics")
    ax[0].plot(t, d["M2_clean_fixed"], "-", color="C2", label="M2_clean_fixed")
    ax[0].plot(t, d["BL50"], "-", color="C3", lw=2, label="BL50 (final)")
    ax[0].plot(t, d["oracle_irr"], ":", color="C1", lw=1.5, label="physics on MEASURED POA (diagnosis)")
    ax[0].set_ylabel("kW")
    ax[0].legend(fontsize=8, loc="upper left")
    ax[0].set_title(f"{w['date']:%Y-%m-%d} ({w['fold']}): actual {w['actual_kwh'] / 1000:.1f} MWh, "
                    f"BL50 {w['bl50_kwh'] / 1000:.1f} MWh, error {w['err_pct']:+.0f} %")
    ax[1].plot(t, d["cloud_s1"], "-o", ms=3, color="grey", label="forecast cloud (smoothed ±1 h)")
    ax[1].plot(t, d["kt_fcst"], "--", color="C0", label="forecast-implied clearness 1 - a·C^b")
    ax[1].plot(t, d["kt_meas"], "-o", ms=3, color="C1", label="measured clearness POA / clear-sky POA")
    ax[1].set_ylim(0, 1.3)
    ax[1].set_title(f"mean |measured - forecast-implied clearness| = {mad:.2f}", fontsize=9)
    ax[1].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DAYS / f"{w['date']:%Y-%m-%d}_{w['fold']}.png", dpi=100)
    plt.close(fig)


def by_hour(preds):
    print("\n=== 4. BL50 error by hour of day, F1/F2/F8 vs ac_power_kw ===")
    sc = preds[scoring_mask(preds, "ac_power_kw")].copy()
    sc["err"] = sc["BL50"] - sc["ac_power_kw"]
    sc["hour"] = sc["timestamp"].dt.hour
    sc["period"] = pd.cut(sc["hour"], [-1, 9, 13, 23], labels=["morning (<10)", "midday (10-13)",
                                                               "afternoon (14+)"])
    per = sc.groupby("period", observed=True).agg(
        n=("err", "size"), mae_pct=("err", lambda e: 100 * e.abs().mean() / AC_KW),
        bias_kw=("err", "mean"), mean_actual_kw=("ac_power_kw", "mean"))
    per["b2b_bias_kw"] = (sc["B2b"] - sc["ac_power_kw"]).groupby(sc["period"], observed=True).mean()
    per["m2_bias_kw"] = (sc["M2_clean_fixed"] - sc["ac_power_kw"]).groupby(sc["period"], observed=True).mean()
    per["mae_pct_of_mean"] = 100 * (sc["err"].abs().groupby(sc["period"], observed=True).mean()
                                    / per["mean_actual_kw"])
    print(per.round(1).to_string())
    hr = sc.groupby("hour").agg(n=("err", "size"), mae_pct=("err", lambda e: 100 * e.abs().mean() / AC_KW),
                                bias_kw=("err", "mean"), mean_actual_kw=("ac_power_kw", "mean"))
    hr["b2b_bias_kw"] = (sc["B2b"] - sc["ac_power_kw"]).groupby(sc["hour"]).mean()
    hr["m2_bias_kw"] = (sc["M2_clean_fixed"] - sc["ac_power_kw"]).groupby(sc["hour"]).mean()
    per_fold = sc.pivot_table(index="period", columns="fold", values="err", aggfunc="mean", observed=True)
    print(hr.round(1).to_string())
    print("bias kW by period and fold:")
    print(per_fold.round(0).to_string())
    per.to_csv(REPORTS / "error_by_period.csv")
    hr.to_csv(REPORTS / "error_by_hour.csv")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(hr.index - 0.2, hr["bias_kw"], 0.4, label="BL50 bias")
    ax.bar(hr.index + 0.2, hr["mae_pct"] * 100, 0.4, alpha=0.6, label="BL50 MAE (kW)")
    ax.plot(hr.index, hr["b2b_bias_kw"], "C0--", lw=1, label="B2b bias")
    ax.plot(hr.index, hr["m2_bias_kw"], "C2--", lw=1, label="M2_clean_fixed bias")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("hour beginning (IST)")
    ax.set_ylabel("kW")
    ax.set_title("BL50 error by hour, monsoon folds F1/F2/F8 vs ac_power_kw")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORTS / "error_by_hour.png", dpi=110)
    plt.close(fig)
    return per, hr


def main():
    df = pd.read_parquet(DATA / "train_clean.parquet")
    geo = geometry(df["timestamp"])
    sensitivity(df, geo)
    preds = pd.concat([bl50_fold(df, geo, f)[0] for f in FOLDS if f.name in MONSOON_FOLDS], ignore_index=True)
    for f, d in preds.groupby("fold"):
        r = evaluate(d, d["BL50"].values, "ac_power_kw")[0]
        print(f"check {f}: BL50 MAE {r['mae_pct']:.2f} %")
    expected_test_mae(preds)
    error_analysis(preds)
    by_hour(preds)


if __name__ == "__main__":
    main()
