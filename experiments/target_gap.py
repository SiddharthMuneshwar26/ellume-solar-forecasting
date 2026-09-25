"""Diagnostic: how much does the soiling-corrected training target move the submitted test predictions?

Fits the production pipeline exactly as run.py does (clean rules 1-11 -> B2b fitted_dry physics -> M2 ratio model,
15 leaves, 295 rounds, seeds 42-46 -> BL50), twice:
  - production: target ac_power_clean (B2b loss factor and M2), as in run.py
  - raw:        target = row-cleaned ac_power_kw (rules 1-11, no soiling correction) for both the loss factor and M2
and compares the two on test daylight hours, overall and by forecast cloud bin.

Diagnostic only. It writes nothing except experiments/results/target_gap.csv: cleaning side files go to a temp
folder, and run.py outputs / predictions.csv are not touched. The production refit is checked against predictions.csv.

Run:  .venv\\Scripts\\python.exe experiments\\target_gap.py
"""
import contextlib
import io
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run
import src.clean as clean
from src.solar import prepare_forecast
from src.train import FIXED_ITERS, FIXED_LEAVES, FIXED_SEEDS, TL_CHOICE, GBMModel

RESULTS = Path(__file__).resolve().parent / "results"
BINS = [(0.0, 0.3, "0-0.3"), (0.3, 0.7, "0.3-0.7"), (0.7, 1.0001, ">=0.7")]


def clean_train():
    """run_cleaning() on a temp copy so no project file is rewritten."""
    tmp = Path(tempfile.mkdtemp())
    shutil.copy(ROOT / "data" / "train.csv", tmp / "train.csv")
    data0, rep0 = clean.DATA, clean.REPORTS
    clean.DATA = clean.REPORTS = tmp
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return clean.run_cleaning()
    finally:
        clean.DATA, clean.REPORTS = data0, rep0
        shutil.rmtree(tmp, ignore_errors=True)


def fit_predict(train, test):
    """Same calls as run.fit_final / run.predict_test; the target comes from train['ac_power_clean']."""
    m2, b2b = run.fit_final(train)
    bl50, p_b2b, p_m2, geo = run.predict_test(m2, b2b, test)
    return {"B2b": p_b2b, "M2": p_m2, "BL50": bl50}, geo, b2b.loss, m2.n_train


def main():
    train = clean_train()
    test_raw = pd.read_csv(ROOT / "data" / "test.csv", dtype={"timestamp": str})
    test = test_raw.assign(timestamp=pd.to_datetime(test_raw["timestamp"], format="%Y-%m-%d %H:%M:%S"))

    prod, geo, loss_prod, n_prod = fit_predict(train, test)
    sub = pd.read_csv(ROOT / "predictions.csv", dtype={"timestamp": str})
    assert (sub["timestamp"].values == test_raw["timestamp"].values).all()
    max_dev = float(np.abs(np.round(prod["BL50"], 3) - sub["predicted_ac_power_kw"].values).max())
    assert max_dev < 1e-6, f"production refit does not reproduce predictions.csv (max dev {max_dev})"

    # Raw variant: the production code reads the target from `ac_power_clean` (loss factor in PhysicsModel.fit,
    # M2 via GBMModel's target). Pointing that column at the row-cleaned ac_power_kw swaps the target and nothing else.
    train_raw = train.copy()
    train_raw["ac_power_clean"] = train_raw["ac_power_kw"]
    raw, _, loss_raw, n_raw = fit_predict(train_raw, test)

    day = geo["elevation"].values > 0
    cloud = prepare_forecast(test)["forecast_cloud_cover"].values
    rows = []
    for model in ("B2b", "M2", "BL50"):
        for lo, hi, name in [(0.0, 1.0001, "all")] + BINS:
            m = day & (cloud >= lo) & (cloud < hi)
            p, r = prod[model][m], raw[model][m]
            diff = r.mean() - p.mean()
            rows.append({"model": model, "cloud_bin": name, "n_hours": int(m.sum()),
                         "mean_production_kw": p.mean(), "mean_raw_target_kw": r.mean(),
                         "mean_diff_kw": diff, "mean_diff_pct_of_production": 100 * diff / p.mean(),
                         "mean_abs_diff_kw": float(np.abs(r - p).mean())})
    out = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS / "target_gap.csv", index=False, lineterminator="\n")

    print(f"production refit reproduces predictions.csv (max |dev| {max_dev:.3g} kW)")
    print(f"B2b loss factor: production (ac_power_clean) {loss_prod:.4f}, raw (ac_power_kw) {loss_raw:.4f}; "
          f"M2 training rows {n_prod} / {n_raw}")
    print(f"test daylight hours: {int(day.sum())}; diff = raw-target minus production\n")
    with pd.option_context("display.width", 200):
        print(out.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\nwrote {RESULTS / 'target_gap.csv'}")


if __name__ == "__main__":
    main()
