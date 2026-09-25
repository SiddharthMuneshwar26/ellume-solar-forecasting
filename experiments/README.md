# Leakage-safe validation experiment

This folder contains an isolated measurement run for the frozen production setup. It does not modify `run.py`, any production source file under `src/`, `predictions.csv`, or `reports/`.

## Run

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe experiments\leakage_safe_validation.py
```

The program reads `data/train.csv` and writes only beneath `experiments/results/`.

## Method

The six date windows are the plan's contiguous forward folds. For each fold, the raw rows are time-filtered before any target cleanup or calibration; assertions enforce `max(training timestamp) < min(validation timestamp)`. Training cleanup applies the documented sentinel, MW-to-kW, over-rating, known flatline, nighttime negative, and POA/power-ratio rules, without any soiling correction or future interpolation. Duplicate training rows use the documented duplicate resolution after the cutoff. Validation targets remain the raw `ac_power_kw` values. Known rows in the fixed historian MW-unit interval are excluded from validation, never rescaled. Conflicting duplicate validation timestamps are excluded as ambiguous; identical-target duplicates are reduced to one unchanged actual. STOP, PARTIAL, daylight STANDBY with high POA, missing-target/sentinel, and unsafe forecast rows are not scored.

Forecast imputation uses time interpolation only among same-day forecast values. No adjacent-day fallback is used. Any row with a required weather input still missing after this step is omitted from the common scoring population for all models. Across validation windows, seven missing wind forecast entries were filled by same-day interpolation; no required forecast entries remained missing and no rows were excluded for unusable forecasts. `fold_audit.csv` records per-field interpolation counts, unresolved forecast gaps, each invalid/outage reason, unit-window rows, and duplicate rows. The validation validity mask also excludes documented >10,100 kW spikes, POA/power-ratio spikes, and the fixed logger flatline while retaining raw values for every scored actual.

The experiment reuses only deterministic geometry/radiative helpers and the training-only `fit_linke` / `fit_cloud_curve` routines from `src.solar`. Its `FoldPhysics` fits turbidity, cloud attenuation, and loss once using only the fold's training rows, then freezes them. Forecast features are built in this folder with same-day-only imputation. M2 uses the production feature list, LightGBM settings, 15 leaves, 295 rounds, and seeds 42–46, with no tuning. B0 follows the specified month/hour median and fallback order. BL50 is an equal blend. All models predict zero for elevation <= 0 and are bounded to [0, 10,000] kW.

## Outputs

- `results/validation_predictions.csv`: long-form scored predictions, one row per fold/timestamp/model; `is_valid` is true for each included row. A timestamp is unique per fold/model, with four model rows per scored timestamp.
- `results/fold_results.csv`: requested fold × model × population metrics.
- `results/model_summary.csv`: pooled metrics over all validation predictions, not a mean of fold percentages.
- `results/fold_audit.csv`: fold boundaries, training/calibration/model row counts, fitted physics settings, forecast gap counts, and exclusions.

`ALL-VALID` includes eligible nighttime rows; `DAYLIGHT-VALID` is its subset with solar elevation > 0. The same actual rows are used for all four models within each population. Metrics are in kW and percent of 10 MW capacity.

## Executed row counts

| Fold | ALL-VALID | DAYLIGHT-VALID | M2 training rows | MW-window rows excluded | Invalid target/state rows | Conflicting duplicate rows excluded |
|---|---:|---:|---:|---:|---:|---:|
| F1 | 724 | 379 | 2,012 | 605 | 139 | 0 |
| F2 | 1,211 | 595 | 2,631 | 0 | 241 | 4 |
| F3 | 1,427 | 680 | 3,169 | 0 | 14 | 0 |
| F4 | 1,281 | 581 | 3,768 | 0 | 74 | 2 |
| F5 | 1,426 | 738 | 4,350 | 0 | 28 | 2 |
| F6 | 1,351 | 723 | 5,011 | 0 | 95 | 4 |

The invalid target/state counts are overlapping row-level rule flags, not additive categories. Identical-target duplicate surplus copies are also removed without changing the retained actual. The outputs contain 7,420 pooled ALL-VALID rows and 3,696 pooled DAYLIGHT-VALID rows.

The experiment was run repeatedly with the frozen seeds/configuration. SHA-256 hashes of `fold_results.csv`, `model_summary.csv`, and `validation_predictions.csv` matched across consecutive runs.
