from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.solar import AC_KW

CLOUD_BUCKETS = [(0.0, 0.3, "0-0.3"), (0.3, 0.7, "0.3-0.7"), (0.7, 1.0001, "0.7-1")]


@dataclass(frozen=True)
class Fold:
    name: str
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_end: pd.Timestamp | None
    note: str = ""

    def test_mask(self, df):
        return df["timestamp"].between(self.test_start, self.test_end)

    def train_mask(self, df):
        if self.train_end is None:
            return ~self.test_mask(df)
        return df["timestamp"] < self.train_end


def _month_fold(i, month):
    start = pd.Timestamp(month)
    end = start + pd.offsets.MonthBegin(1) - pd.Timedelta(hours=1)
    return Fold(f"F{i}", start, end, start, f"rolling origin: test {start:%Y-%m}, train = all before")


FOLDS = [
    Fold("F1", pd.Timestamp("2024-07-01"), pd.Timestamp("2024-08-31 23:00"), pd.Timestamp("2024-07-01"),
         "forward monsoon: train <= 2024-06-30 (no monsoon months in training)"),
    Fold("F2", pd.Timestamp("2024-07-01"), pd.Timestamp("2024-08-31 23:00"), None,
         "leave-out monsoon: train = all other data, incl. FUTURE data up to 2025-06 (optimistic)"),
] + [_month_fold(i, f"2025-{m:02d}-01") for i, m in zip(range(3, 9), range(1, 7))]


def scoring_mask(df, target):
    return df["train_ok"].values & (df["elevation"].values > 0) & np.isfinite(df[target].values)


def _metrics(actual, pred, day, cloud_day_bucket=None):
    err = pred - actual
    out = {
        "n_hours": int(len(actual)),
        "mae_pct": 100 * np.mean(np.abs(err)) / AC_KW if len(actual) else np.nan,
        "rmse_pct": 100 * np.sqrt(np.mean(err ** 2)) / AC_KW if len(actual) else np.nan,
        "bias_kw": float(np.mean(err)) if len(actual) else np.nan,

        "mae_pct_of_mean": (100 * np.mean(np.abs(err)) / np.mean(actual)
                            if len(actual) and np.mean(actual) > 0 else np.nan),
    }

    d = pd.DataFrame({"a": actual, "p": pred, "day": day}).groupby("day")[["a", "p"]].sum()
    if cloud_day_bucket is not None:
        d = d[d.index.isin(cloud_day_bucket)]
    out["n_days"] = int(len(d))
    out["daily_energy_err_pct"] = (100 * (d["p"] - d["a"]).abs().sum() / d["a"].sum()
                                   if len(d) and d["a"].sum() > 0 else np.nan)
    return out


def add_skill(res, ref_model="B0", keys=("fold", "target", "bucket")):
    """Add `skill_vs_<ref>` = 1 - MAE / MAE_ref, matched on fold, scoring target and cloud bucket.

    `res` is a long results table with a `model` column and one row per (model, *keys).
    """
    ref = res[res["model"] == ref_model].set_index(list(keys))["mae_pct"].rename("_ref")
    out = res.join(ref, on=list(keys))
    out[f"skill_vs_{ref_model}"] = 1 - out["mae_pct"] / out["_ref"]
    return out.drop(columns="_ref")


def evaluate(df, pred, target):

    m = scoring_mask(df, target)
    a = df[target].values[m]
    p = np.asarray(pred)[m]
    c = df["forecast_cloud_cover"].values[m]
    day = df["timestamp"].dt.normalize().values[m]
    day_cloud = pd.Series(c).groupby(day).mean()
    rows = [{"bucket": "all", **_metrics(a, p, day)}]
    for lo, hi, name in CLOUD_BUCKETS:
        hb = (c >= lo) & (c < hi)
        days_b = day_cloud.index[(day_cloud >= lo) & (day_cloud < hi)]
        r = _metrics(a[hb], p[hb], day[hb])

        r_day = _metrics(a, p, day, cloud_day_bucket=days_b)
        r["n_days"], r["daily_energy_err_pct"] = r_day["n_days"], r_day["daily_energy_err_pct"]
        rows.append({"bucket": name, **r})
    return rows
