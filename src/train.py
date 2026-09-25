import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features import CLOUD_FEATURES, FEATURES, build_features
from src.solar import AC_KW, PhysicsModel, geometry
from src.validate import FOLDS, add_skill, evaluate, scoring_mask

ROOT = Path(__file__).resolve().parents[1]
DATA, REPORTS = ROOT / "data", ROOT / "reports"
TARGETS = ["ac_power_kw", "ac_power_clean"]
MONSOON_FOLDS = ["F1", "F2", "F8"]
DRY_FOLDS = [f"F{i}" for i in range(3, 9)]


class Climatology:


    def fit(self, geo, df):
        d = df[df["train_ok"] & df["ac_power_kw"].notna()]
        self.table = d.groupby([d["timestamp"].dt.month, d["timestamp"].dt.hour])["ac_power_kw"].mean()
        self.months = sorted(self.table.index.get_level_values(0).unique())
        return self

    def predict(self, geo, df):
        m = df["timestamp"].dt.month.values
        h = df["timestamp"].dt.hour.values
        near = {mo: min(self.months, key=lambda x: min(abs(x - mo), 12 - abs(x - mo))) for mo in set(m)}
        p = np.array([self.table.get((near[a], b), 0.0) for a, b in zip(m, h)])
        p = np.clip(p, 0, AC_KW)
        p[geo["elevation"].values < 0] = 0.0
        return p

    def params(self):
        return {"months_in_training": self.months}


MODELS = {
    "B0": lambda: Climatology(),
    "B1": lambda: PhysicsModel(tl="fitted", cloud="none"),
    "B2a": lambda: PhysicsModel(tl="fitted", cloud="kc"),
    "B2b": lambda: PhysicsModel(tl="fitted", cloud="fitted"),
    "B1_defTL": lambda: PhysicsModel(tl="default", cloud="none"),
    "B2a_defTL": lambda: PhysicsModel(tl="default", cloud="kc"),
    "B2b_defTL": lambda: PhysicsModel(tl="default", cloud="fitted"),
    "B2b_dryTL": lambda: PhysicsModel(tl="fitted_dry", cloud="fitted"),
}


def pick_weeks(te):
    """Clearest and cloudiest 7-day windows in the test period by mean daylight forecast cloud.

    Only days with >= 8 scored (train_ok, daylight, non-NaN) hours count, so outage weeks are skipped.
    """
    d = te[te["elevation"] > 0]
    date = d["timestamp"].dt.normalize()
    n_ok = (d["train_ok"] & d["ac_power_kw"].notna()).groupby(date).sum()
    day = d.groupby(date)["forecast_cloud_cover"].mean().where(n_ok >= 8)
    day = day.reindex(pd.date_range(day.index.min(), day.index.max(), freq="D"))
    roll = day.rolling(7, min_periods=6).mean().dropna()
    if roll.empty:
        return None, None
    return roll.idxmin() - pd.Timedelta(days=6), roll.idxmax() - pd.Timedelta(days=6)


def plot_fold(fold, te, preds):
    clear, cloudy = pick_weeks(te)
    fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharey=True)
    for ax, start, lab in [(axes[0], clear, "clearest week"), (axes[1], cloudy, "cloudiest week")]:
        w = te["timestamp"].between(start, start + pd.Timedelta(days=7) - pd.Timedelta(hours=1))
        t = te.loc[w, "timestamp"]
        actual = te.loc[w, "ac_power_kw"].where(te.loc[w, "train_ok"])
        ax.plot(t, actual, "k-", lw=1.5, label="actual ac_power_kw (train_ok)")
        ax.plot(t, preds["B0"][w.values], "-", color="C0", lw=1, label="B0 climatology")
        ax.plot(t, preds["B2b"][w.values], "-", color="C3", lw=1, label="B2b physics + fitted cloud")
        ax2 = ax.twinx()
        ax2.fill_between(t, te.loc[w, "forecast_cloud_cover"], step="mid", alpha=0.12, color="grey")
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("forecast cloud")
        cc = te.loc[w & (te["elevation"] > 0), "forecast_cloud_cover"].mean()
        ax.set_title(f"{fold.name} {lab}: {start:%Y-%m-%d} (mean daylight forecast cloud {cc:.2f})")
        ax.set_ylabel("kW")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORTS / f"baseline_weeks_{fold.name}.png", dpi=100)
    plt.close(fig)


def main():
    df = pd.read_parquet(DATA / "train_clean.parquet")
    geo = geometry(df["timestamp"])
    rows, params = [], {}
    for fold in FOLDS:
        tr_m, te_m = fold.train_mask(df).values, fold.test_mask(df).values
        tr, te = df[tr_m].reset_index(drop=True), df[te_m].reset_index(drop=True)
        g_tr, g_te = geo[tr_m], geo[te_m]
        preds = {}
        for name, make in MODELS.items():
            model = make().fit(g_tr, tr)
            preds[name] = model.predict(g_te, te)
            params[f"{fold.name}/{name}"] = model.params()
            for target in TARGETS:
                for r in evaluate(te, preds[name], target):
                    rows.append({"fold": fold.name, "model": name, "target": target, **r})
        plot_fold(fold, te, preds)
        print(f"{fold.name}: train {int(tr_m.sum())} rows, test {int(te_m.sum())} rows "
              f"({fold.test_start:%Y-%m-%d}..{fold.test_end:%Y-%m-%d})")

    res = add_skill(pd.DataFrame(rows), "B0")
    res.to_csv(REPORTS / "baselines.csv", index=False)
    with open(REPORTS / "baseline_params.json", "w") as f:
        json.dump(params, f, indent=1, default=str)

    pd.set_option("display.width", 200)
    allb = res[res.bucket == "all"]
    for metric in ["mae_pct", "rmse_pct", "bias_kw", "daily_energy_err_pct", "mae_pct_of_mean", "skill_vs_B0"]:
        for target in TARGETS:
            t = allb[allb.target == target].pivot(index="model", columns="fold", values=metric)
            t["F3-F8 mean"] = t[DRY_FOLDS].mean(axis=1)
            print(f"\n{metric} vs {target}")
            print(t.loc[list(MODELS)].round(2).to_string())
    bk = res[(res.target == "ac_power_kw")].groupby(["model", "bucket"])[["mae_pct", "bias_kw"]].mean()
    print("\nmean over folds by cloud bucket (vs ac_power_kw):")
    print(bk.unstack("bucket").loc[list(MODELS)].round(2).to_string())
    soiling_diagnosis(df, res)
    print("\nwrote reports/baselines.csv, reports/baseline_params.json, reports/baseline_weeks_F*.png")


def soiling_diagnosis(df, res):
    """Dry-season folds: is B2's over-prediction of raw power explained by soiling?

    For any model, bias_vs_kw - bias_vs_clean = mean(ac_power_clean - ac_power_kw) on the scored rows (the
    soiling gap). The informative question is which target each model is unbiased against.
    """
    print("\n=== Soiling diagnosis, F3..F8 (bias kW / MAE %) ===")
    gap = {}
    for fold in FOLDS[2:]:
        te = df[fold.test_mask(df).values]
        m = scoring_mask(te, "ac_power_kw")
        gap[fold.name] = {"mean_soiling_factor": te["soiling_factor"].values[m].mean(),
                          "gap_kw (clean - raw)": (te["ac_power_clean"] - te["ac_power_kw"]).values[m].mean()}
    print(pd.DataFrame(gap).round(3).to_string())
    a = res[(res.bucket == "all") & res.fold.isin([f.name for f in FOLDS[2:]])]
    t = a.pivot_table(index="model", columns="target", values=["bias_kw", "mae_pct"], aggfunc="mean")
    t.columns = [f"{m} vs {tg}" for m, tg in t.columns]
    print("mean over F3..F8:")
    print(t.loc[list(MODELS)].round(2).to_string())
    per = a.pivot_table(index="model", columns=["fold", "target"], values="bias_kw")
    print("bias_kw per fold:")
    print(per.loc[list(MODELS)].round(0).to_string())

TL_CHOICE = "fitted_dry"
SEED = 42
GRID_LEAVES = (15, 31)
BASE_PARAMS = dict(objective="regression", learning_rate=0.03, min_child_samples=20, feature_fraction=0.9,
                   bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, seed=SEED, deterministic=True,
                   force_row_wise=True, num_threads=4, verbose=-1)
MIN_PHYS_KW = 200.0


class GBMModel:
    """LightGBM on forecast/physics features, trained on daytime rows only."""

    def __init__(self, kind, target, leaves=None, n_iter=None, seeds=(SEED,), tl=TL_CHOICE):

        self.kind, self.target = kind, target
        self.fixed = leaves is not None and n_iter is not None
        self.leaves, self.n_iter, self.seeds = leaves, n_iter, tuple(seeds)
        self.tl = tl

    def _xy(self, X, df):
        y = df[self.target].values
        phys = X["phys_power"].values
        sel = df["train_ok"].values & (X["elevation"].values > 0) & np.isfinite(y)
        if self.kind == "M1":
            yt = y
        elif self.kind == "M2":
            sel &= phys >= MIN_PHYS_KW
            yt = y / np.where(phys > 0, phys, np.nan)
        else:
            yt = y - phys
        return sel, yt

    def _lgb_params(self, leaves, seed=SEED):
        p = dict(BASE_PARAMS, num_leaves=leaves, seed=seed)
        if self.kind == "M1":
            p["monotone_constraints"] = [-1 if f in CLOUD_FEATURES else 0 for f in FEATURES]
            p["monotone_constraints_method"] = "advanced"
        return p

    def fit(self, geo, df):
        self.phys = PhysicsModel(tl=self.tl, cloud="fitted").fit(geo, df)
        X = build_features(df, geo, self.phys)
        sel, yt = self._xy(X, df)
        self.n_train = int(sel.sum())
        if self.fixed:
            self.models = [lgb.train(self._lgb_params(self.leaves, s), lgb.Dataset(X[sel], yt[sel]),
                                     num_boost_round=self.n_iter) for s in self.seeds]
            return self
        t = df["timestamp"]
        cut = t.min() + 0.8 * (t.max() - t.min())
        inner_tr, inner_va = sel & (t < cut).values, sel & (t >= cut).values
        best = None
        for leaves in GRID_LEAVES:
            m = lgb.train(self._lgb_params(leaves), lgb.Dataset(X[inner_tr], yt[inner_tr]), num_boost_round=3000,
                          valid_sets=[lgb.Dataset(X[inner_va], yt[inner_va])],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            score = m.best_score["valid_0"]["l2"]
            if best is None or score < best[0]:
                best = (score, leaves, max(m.best_iteration, 50))
        _, self.leaves, self.n_iter = best
        self.models = [lgb.train(self._lgb_params(self.leaves), lgb.Dataset(X[sel], yt[sel]),
                                 num_boost_round=self.n_iter)]
        return self

    def predict(self, geo, df, return_features=False):
        X = build_features(df, geo, self.phys)
        raw = np.mean([m.predict(X) for m in self.models], axis=0)
        phys = X["phys_power"].values
        if self.kind == "M1":
            p = raw
        elif self.kind == "M2":
            p = np.where(phys >= MIN_PHYS_KW, phys * np.clip(raw, 0, None), phys)
        else:
            p = phys + raw
        p = np.clip(p, 0, AC_KW)
        p[geo["elevation"].values < 0] = 0.0
        if return_features:
            return p, X
        return p

    def params(self):
        return {"kind": self.kind, "target": self.target, "num_leaves": self.leaves, "n_iter": self.n_iter,
                "fixed": self.fixed, "seeds": list(self.seeds), "n_train": self.n_train,
                "phys": self.phys.params()}


ML_MODELS = {
    "B0": ("ac_power_kw", lambda: Climatology()),
    "B2b": ("ac_power_clean", lambda: PhysicsModel(tl=TL_CHOICE, cloud="fitted")),
}
for _k in ("M1", "M2", "M3"):
    ML_MODELS[f"{_k}_kw"] = ("ac_power_kw", lambda k=_k: GBMModel(k, "ac_power_kw"))
    ML_MODELS[f"{_k}_clean"] = ("ac_power_clean", lambda k=_k: GBMModel(k, "ac_power_clean"))


def run_models():
    df = pd.read_parquet(DATA / "train_clean.parquet")
    geo = geometry(df["timestamp"])
    rows, params = [], {}
    for fold in FOLDS:
        tr_m, te_m = fold.train_mask(df).values, fold.test_mask(df).values
        tr, te = df[tr_m].reset_index(drop=True), df[te_m].reset_index(drop=True)
        for name, (train_target, make) in ML_MODELS.items():
            model = make().fit(geo[tr_m], tr)
            pred = model.predict(geo[te_m], te)
            params[f"{fold.name}/{name}"] = model.params()
            for score_target in TARGETS:
                primary = (score_target == "ac_power_kw") if fold.name in ("F1", "F2") \
                    else (score_target == train_target)
                for r in evaluate(te, pred, score_target):
                    rows.append({"fold": fold.name, "model": name, "train_target": train_target,
                                 "target": score_target, "primary": primary, **r})
        print(f"{fold.name} done: " + ", ".join(
            f"{n} leaves={params[f'{fold.name}/{n}']['num_leaves']} iters={params[f'{fold.name}/{n}']['n_iter']}"
            for n in ML_MODELS if n.startswith("M")))

    res = add_skill(pd.DataFrame(rows), "B0")
    res.to_csv(REPORTS / "model_comparison.csv", index=False)
    with open(REPORTS / "model_params.json", "w") as f:
        json.dump(params, f, indent=1, default=str)

    pd.set_option("display.width", 220)
    order = list(ML_MODELS)
    allb = res[res.bucket == "all"]
    for target in TARGETS:
        for metric in ["mae_pct", "rmse_pct", "bias_kw", "skill_vs_B0", "daily_energy_err_pct"]:
            t = allb[allb.target == target].pivot(index="model", columns="fold", values=metric).loc[order]
            t["F1F2 mean"] = t[["F1", "F2"]].mean(axis=1)
            t["monsoon mean"] = t[MONSOON_FOLDS].mean(axis=1)
            t["F3-F8 mean"] = t[DRY_FOLDS].mean(axis=1)
            print(f"\n{metric} scored vs {target}")
            print(t.round(3 if metric == "skill_vs_B0" else 2).to_string())

    p = allb[allb.primary & allb.fold.isin(DRY_FOLDS)]
    print("\nF3-F8 mean, scored vs each model's own training target:")
    print(p.groupby("model")[["mae_pct", "rmse_pct", "bias_kw", "skill_vs_B0"]].mean().loc[order].round(3).to_string())
    bk = res[res.fold.isin(["F1", "F2"]) & (res.target == "ac_power_kw")]
    print("\nF1/F2 mean by forecast cloud bucket (vs ac_power_kw):")
    print(bk.groupby(["model", "bucket"])[["mae_pct", "bias_kw", "skill_vs_B0"]].mean()
            .unstack("bucket").loc[order].round(2).to_string())
    print("\nwrote reports/model_comparison.csv, reports/model_params.json")



FIXED_LEAVES, FIXED_ITERS, FIXED_SEEDS = 15, 295, (42, 43, 44, 45, 46)
BL_CLOUD_THRESHOLD = 0.3
DECISION_ORDER = ["B2b", "BL50", "BL_cloud", "M2_clean_fixed"]
DECISION_TOL = 0.1


def run_final_choice():
    df = pd.read_parquet(DATA / "train_clean.parquet")
    geo = geometry(df["timestamp"])
    rows, params = [], {}
    for fold in FOLDS:
        tr_m, te_m = fold.train_mask(df).values, fold.test_mask(df).values
        tr, te = df[tr_m].reset_index(drop=True), df[te_m].reset_index(drop=True)
        tuned = GBMModel("M2", "ac_power_clean").fit(geo[tr_m], tr)
        fixed = GBMModel("M2", "ac_power_clean", FIXED_LEAVES, FIXED_ITERS, FIXED_SEEDS).fit(geo[tr_m], tr)
        b2b = fixed.phys
        p_fixed, X = fixed.predict(geo[te_m], te, return_features=True)
        preds = {"B2b": b2b.predict(geo[te_m], te), "M2_clean": tuned.predict(geo[te_m], te),
                 "M2_clean_fixed": p_fixed}
        preds["BL50"] = 0.5 * preds["B2b"] + 0.5 * preds["M2_clean_fixed"]
        w = np.where(X["cloud_s1"].values < BL_CLOUD_THRESHOLD, 0.7, 0.3)
        preds["BL_cloud"] = w * preds["B2b"] + (1 - w) * preds["M2_clean_fixed"]
        params[fold.name] = {"B2b": b2b.params(), "M2_clean_tuned": tuned.params(),
                             "M2_clean_fixed": fixed.params()}
        for name, p in preds.items():
            for target in TARGETS:
                for r in evaluate(te, p, target):
                    rows.append({"fold": fold.name, "model": name, "target": target, **r})
        print(f"{fold.name} done (tuned M2_clean: leaves={tuned.leaves}, iters={tuned.n_iter})")

    res = pd.DataFrame(rows)
    res.to_csv(REPORTS / "final_choice.csv", index=False)
    with open(REPORTS / "final_choice_params.json", "w") as f:
        json.dump(params, f, indent=1, default=str)

    pd.set_option("display.width", 220)
    order = ["B2b", "M2_clean", "M2_clean_fixed", "BL50", "BL_cloud"]
    allb = res[res.bucket == "all"]
    metrics = ["mae_pct", "rmse_pct", "bias_kw", "daily_energy_err_pct"]
    for target in TARGETS:
        for metric in metrics:
            t = allb[allb.target == target].pivot(index="model", columns="fold", values=metric).loc[order]
            t["monsoon mean"] = t[MONSOON_FOLDS].mean(axis=1)
            t["F3-F8 mean"] = t[DRY_FOLDS].mean(axis=1)
            print(f"\n{metric} vs {target}")
            print(t.round(2).to_string())

    summ = pd.concat({
        "monsoon (F1,F2,F8) vs kw": allb[allb.fold.isin(MONSOON_FOLDS) & (allb.target == "ac_power_kw")]
            .groupby("model")[metrics].mean(),
        "F3-F8 vs clean": allb[allb.fold.isin(DRY_FOLDS) & (allb.target == "ac_power_clean")]
            .groupby("model")[metrics].mean(),
    }, axis=1).loc[order]
    print("\n=== Summary ===")
    print(summ.round(2).to_string())
    summ.to_csv(REPORTS / "final_choice_summary.csv")

    bk = res[res.fold.isin(MONSOON_FOLDS) & (res.target == "ac_power_kw")]
    print("\nmonsoon mean (F1,F2,F8) by forecast cloud bucket, vs ac_power_kw:")
    print(bk.groupby(["model", "bucket"])[["mae_pct", "bias_kw", "daily_energy_err_pct"]].mean()
            .unstack("bucket").loc[order].round(2).to_string())
    bk = res[res.fold.isin(DRY_FOLDS) & (res.target == "ac_power_clean")]
    print("\nF3-F8 mean by forecast cloud bucket, vs ac_power_clean:")
    print(bk.groupby(["model", "bucket"])[["mae_pct", "bias_kw"]].mean()
            .unstack("bucket").loc[order].round(2).to_string())


    st = allb[allb.model.isin(["M2_clean", "M2_clean_fixed"])]
    st = st[((st.fold.isin(["F1", "F2"])) & (st.target == "ac_power_kw"))
            | ((~st.fold.isin(["F1", "F2"])) & (st.target == "ac_power_clean"))]
    st = st.pivot(index="fold", columns="model", values="mae_pct")
    st["diff (fixed - tuned)"] = st["M2_clean_fixed"] - st["M2_clean"]
    print("\nstability, MAE % (F1/F2 vs kw, F3-F8 vs clean):")
    print(st.round(3).to_string())


    mon = summ["monsoon (F1,F2,F8) vs kw"]["mae_pct"]
    best = mon.min()
    pick = next(m for m in DECISION_ORDER if mon[m] <= best + DECISION_TOL)
    print(f"\nDecision rule: best monsoon-mean MAE = {best:.3f} ({mon.idxmin()}); "
          f"candidates within {DECISION_TOL}: {[m for m in DECISION_ORDER if mon[m] <= best + DECISION_TOL]}; "
          f"simplest -> {pick}")
    print("\nwrote reports/final_choice.csv, reports/final_choice_summary.csv, reports/final_choice_params.json")


if __name__ == "__main__":
    commands = {"baselines": main, "models": run_models, "final_choice": run_final_choice}
    cmd = sys.argv[1] if len(sys.argv) > 1 else "baselines"
    if cmd not in commands:
        sys.exit(f"unknown command {cmd!r}; use one of: {', '.join(commands)}")
    commands[cmd]()
