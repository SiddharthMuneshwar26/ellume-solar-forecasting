# NOTES

Working log of what I found in the data, what I decided and why.

> **Which validation numbers are final.** The reported validation metric is in `## Leakage-safe validation`
> (last section): BL50 pooled daylight MAE **274.43 kW = 2.744 %** of 10 MW over six forward folds.
> The sections `## Physics and baselines` through `## Error analysis` use the original folds in `src/validate.py`.
> That validation is superseded because of leakage. Its numbers (including BL50 3.87 %) are kept as the record of how
> the model was chosen, not as performance estimates.

## Setup

- venv at `.venv`, **Python 3.12.10**.
- `requirements.txt` pins: pandas 3.0.6, numpy 2.5.3, pvlib 0.15.2, lightgbm 4.7.0, scikit-learn 1.9.1, matplotlib 3.11.2, plus pyarrow 25.0.1 (parquet I/O, D5) and ipykernel 7.3.0 (walkthrough notebook only).
- Layout: `src/` (package), `reports/`, `NOTES.md`, `notebooks/walkthrough.ipynb`. `src/solar.py` holds shared geometry helpers (solar position + Ineichen clear-sky POA at hour mid-point, Asia/Kolkata).

## Audit

Script: `src/audit.py` (read-only; run `.venv\Scripts\python.exe -m src.audit`). Full console output is in
`reports/audit_summary.txt`. Tables are `reports/*.csv` and plots are `reports/*.png`.
For the value checks I sorted by time and kept the first copy of each duplicated hour. The actual
de-duplication rule is in `src/clean.py`.

### A1. Timestamps (train)
- **Two string formats.** 12,589 ISO `YYYY-MM-DD HH:MM:SS` and 389 `DD-MM-YYYY HH:MM`. The second format is day-first: 207 rows have a first field > 12. All 12,978 rows parse. Test is 100 % ISO.
- **Ordering.** One break: after `2025-06-30 23:00` the file continues with a **165-row block covering 2024-06-10 00:00 – 2024-06-16 23:00**, appended at the end. None of these hours occur elsewhere, so this is misplaced data, not duplicate data. Sorting fixes it.
- **Duplicates.** 72 rows cover 36 hours, so there are 36 surplus rows. 3 of the pairs are one ISO copy plus one DD-MM copy. In **12 hours the copies disagree, only on `ac_power_kw`**. Differences range from 0.4 to 534 kW (e.g. 2024-01-30 15:00: 6042.7 vs 5666.6; 2025-05-26 15:00: Δ534). Every other column is identical. → `reports/train_duplicates.csv`
- **Gaps.** The expected span is 13,128 h and 12,942 distinct hours are present, so **186 hours are missing**: 12,942 + 36 duplicates = 12,978 rows. The gaps are
  133 single-hour holes spread evenly across hours of day and months, plus 2024-05-12 20–21 (2 h), 2025-02-05 01–03 (3 h), and
  **2025-01-20 00:00 – 2025-01-21 23:00 (48 h, two whole days)**. → `reports/train_gaps.csv`
- Test: 1,488 rows, complete, ordered, no duplicates.

### A2. `status`
- Raw values are messy: mixed case plus leading/trailing spaces (`' RUN '`, `'run'`, `' standby '`, …). After strip+upper they reduce to 5 states:
  RUN 6,338 · STANDBY 5,903 · **NaN 432** · PARTIAL 149 · STOP 120.
- STANDBY never appears with sun > 10°, and RUN never appears with sun < −5°. The status is consistent with daylight.
- **STOP**: 5 episodes. The main one is **2024-08-12 → 2024-08-16** (longest run 49 h; 116 STOP rows in Aug-2024). The plant exports nothing (−2 to −11 kW) while POA is 150–660 W/m². There is also a short one on 2025-05-14. No STOP row has power > 100 kW. These are genuine outages.
- **PARTIAL**: 2024-03-12, 2024-09-08..10, 2024-11-19, 2025-02-27..28, 2025-05-14. Power is ≈ 0.6× (median 0.61) what the POA would give. This looks like partial inverter availability.
- NaN status: 432 rows, 212 of them in daytime. **All 242 NaN-power rows also have NaN status.** The other 190 NaN-status rows have normal-looking power.

### A3. `ac_power_kw`
- NaN 242. **Sentinel −999: 67 rows.** No 9999 in power.
- **Negatives (excluding −999): 3,370**, between −11.5 and −1.5 kW. 3,285 of them are at night, which is normal inverter/transformer standby draw (night median −6.5 kW). Of the 33 with sun > 10°, 31 fall in the Aug-12..16 STOP outage and 2 in the 2025-05-14 STOP. None are "random" daytime negatives.
- **Night-time (sun < 0° at mid-hour, 6,369 rows):** 2,698 exact zero, 3,285 negative, 249 > 1 kW, 113 > 50 kW.
  91 of the >50 kW rows are twilight (elevation −8…0° at mid-hour, i.e. sunrise/sunset falls inside the hour). These are legitimate.
  The 22 deep-night ones are all part of the flatline below.
- **> 10,000 kW: 29 rows.**
  - 16 are marginal overshoot at clipping (10,000.6 – 10,066 kW, clear March/October noons). They are physically plausible as a slight AC overshoot or metering error.
  - **13 are impossible spikes of 11.5 – 26.9 MW**, isolated single hours with normal neighbours (e.g. 2025-02-23 12:00 = 26,906 kW between 9,176 and 9,193). → `reports/power_over_rating.csv`
- **Flatline: one run.** **2024-09-18 00:00 → 2024-09-19 23:00 = 48 h stuck at 2,187.44 kW**, day and night, with status STANDBY at night. Removing daylight hours as "flatline" would miss it, because it spans both day and night. No other ≥ 3 h identical non-zero run exists.
- **Ceiling below 10 MW: none found.** Across all months, power tracks a POA-based expected power (10.8 kW per W/m², temp-corrected) with a monthly median actual/expected of 0.98–1.04. Power only saturates at ≈ 10,000 kW. The histogram has no pile-up below the rating, and daily maxima never repeat more than 3×. Lower monthly maxima in Jun–Aug/Dec come from the season and weather, not a cap. → `power_ceiling_vs_expected.csv`, `power_upper_tail.png`
- **kW → MW unit switch: 2024-07-14 06:00 → 2024-08-08 18:00 (26 days).** Daytime power in this window is ~1000× too small (7–9 at noon; power/POA = 0.0107 vs 10.05 normally, a ratio of 927×). Every daylight hour with POA ≥ 100 in the window is affected, and none outside it. The small **negative night values inside the window are still kW-sized** (−2 … −10), so either only positive exports were rescaled, or night values are kW anyway. Fix: multiply positive values in the window by 1000 (night values: see D1). → `power_mw_unit_hours.csv`, `power_timeseries_log.png`
- 39 hours with POA > 300 W/m² and |power| < 1 kW: 21 are STOP and 18 are STANDBY-labelled. The STANDBY ones have a daytime label mismatch; I checked them in cleaning (`excl_day_standby`).

### A4. Diurnal profile / time alignment
- **Every month peaks in the 12:00 slot** (12:00–13:00, mid 12:30; solar noon ≈ 12:20 IST). → `diurnal_profiles_by_month.png`
- Energy-weighted centroid of power vs Ineichen clear-sky POA (computed at mid-hour): offset **between −2.7 and +0.9 min in every month**. The same holds for measured POA vs clear sky. **No month looks shifted**, and there is no multi-day period where power and POA centroids differ by > 30 min. The timestamps are consistent with hour-beginning IST.
- Individual days deviate by up to ±30 min (e.g. 2024-06-20, 2025-01-03 at −20 min; 2025-06-30 at +31 min). These are weather-driven (afternoon or morning cloud); measured POA moves the same way on the same days. → `diurnal_shift_by_day.csv/png`
- Artefacts visible in the monthly profiles: Aug-2024 POA has a night-time floor from the 412.6 flatline (A5), and Sep-2024 power has a night floor from the 2,187 flatline (A3).

### A5. Sensor sanity
- Sentinels: POA has **−999 ×22 and 9999 ×29**, module temp −999 ×39, ambient −999 ×26. NaN: ~240 each, coinciding with the missing-power rows.
- **Pyranometer night offset**: median night POA = 0.0 in every month. 212 non-zero night values exist, mostly small twilight values (2–13 W/m²), with no negatives. The real problem is a **POA flatline of 412.6 W/m² from 2024-08-25 00:00 to 2024-08-27 23:00** (3 runs, 70 h, day and night). It also produces the spike in the power/POA ratio plot around 2024-08-25.
- 20 daylight hours have POA > 1.3× clear-sky (6 > 1.5×). These are cloud-enhancement candidates or bad values, and the 9999s are excluded from this count.
- **Module temp below ambient at midday (el > 40°, POA > 600): 0 occurrences.** Midday module − ambient has a median of 16 °C, seasonal (10 °C monsoon with wind and cloud, 18–19 °C winter). No temperature flatlines.
- ⚠ **Measured ambient vs forecast temp: mean diff −0.05 °C, std 1.03 °C, r = 0.98.** This is far better than a real day-ahead temperature forecast. Either the "forecast" is very close to an analysis, or the on-site sensor is derived from the same source. Not a problem for modelling, since the sensor is never a model input (D3).
- Rainfall: 326 wet hours, max 29.1 mm/h, no negatives. It is dry Dec–May, with the monsoon Jun–Nov (Jul-2024 252 mm).

### A6. Power / POA ratio: soiling and degradation
- Filter: RUN, POA 400–1300, power 1–9 MW (unclipped), temp-corrected with −0.35 %/°C. The median is 10.8 kW per W/m², i.e. PR ≈ 0.86 vs 12.5 kWp.
- **Clear soiling sawtooth in the dry season.** The ratio decays **≈ 0.08–0.15 %/day (≈ 5–7 % over ~2 months)**, then steps back up by 4–7 % on dates **with no rain**. These are **manual cleaning events at ~2-month intervals**:
  2024-03-01 (10.45→10.92), 2024-05-01 (10.43→11.13), 2025-01-07 (10.46→11.03), 2025-03-09 (10.32→10.92), 2025-05-09 (10.52→10.98).
  During the monsoon (Jun–Oct) the ratio stays flat and high (~11.0–11.2), because rain keeps the panels clean. Soiling resumes from ~Nov. → `ratio_over_time.png`, `ratio_daily.csv`
- **Degradation.** The linear trend is −0.30 %/yr, and same-month year-on-year change is −1.2 … +0.1 % (median ≈ −0.2 %). Both are consistent with normal ~0.3–0.5 %/yr degradation but confounded by where each month sits in the cleaning cycle. The effect is negligible over the test horizon.
- For the test period (Jul–Aug, monsoon), the plant should be in the "clean" state. Training on dry-season hours will slightly under-predict unless soiling is accounted for (e.g. by season or month features, or by giving more weight to monsoon-period data).
- Ratio outliers: the spike around 2024-08-25 comes from the POA flatline, and the lows of ~3.7 in mid-Sep 2024 are PARTIAL days.

### A7. Forecast columns
- Ranges (sentinels excluded) are physical: cloud 0.002–0.996, temp 7.9–40.6 °C, wind 0.49–13.8 m/s, RH 13–100 %. Resolution is 0.001 / 0.1 °C / 0.01 m/s / 1 %.
- **Train: `forecast_wind_ms` = −999 in 13 rows.** No NaNs in train forecasts.
- ⚠ **Test has bad forecast values in 19 rows**: `forecast_temp_c` −999 ×5 and NaN ×4, `forecast_wind_ms` NaN ×6, `forecast_humidity_pct` NaN ×3, `forecast_cloud_cover` NaN ×1 (2025-08-22 06:00). The prediction pipeline **must** handle these (e.g. time-interpolate from neighbouring hours) and still output all 1,488 rows. → `forecast_bad_values_test.csv`
- **Not interpolated from a coarser grid.** Hours lying on a straight line with their neighbours show no hour-mod-3 or hour-mod-6 pattern. They are 1–3 % for cloud and wind. For temp (~18 %) and RH (~30 %) the higher share comes from coarse rounding (0.1 °C / 1 %), is uniform across hours, and the mean |Δ| follows a smooth diurnal cycle. The values look like genuine hourly output.
- **Day-boundary stitching.** The hour-to-hour change at **00:00 is ~2× the typical change** for every variable (cloud 0.125 vs 0.063; RH 5.1 vs 2.5; wind 0.88 vs 0.35). The share of "straight" hours also drops at 23:00/00:00. Each calendar day comes from a different forecast issue, so jumps appear at midnight. This is harmless for solar (night) but means features must not assume continuity across days.
- Cloud forecast is noisy hour-to-hour (mean |Δ| 0.06 at every hour). Its daylight correlation with measured clearness (POA/clear-sky) is only **r = −0.67**, so this is the main source of forecast error.
- Alignment: lag-0 maximises correlation for temp vs ambient (r = 0.98) and cloud vs (1−clearness) in every month. **No timezone offset between the forecast and the plant data.**
- **Train Jul–Aug 2024 vs test Jul–Aug 2025**: mean cloud 0.70 vs **0.80** (test is cloudier), temp 23.4 vs 23.7, wind 6.2 vs 5.8, RH 83.8 vs 84.3. The test months sit in the cloudy tail of the training distribution.

### Implications for cleaning (implemented in `src/clean.py`)
1. Parse both timestamp formats (DD-MM is day-first), sort, and resolve the 36 duplicate hours. 12 of them conflict on power (options: mean, or drop both).
2. Treat −999/9999 as missing in every column. Interpolate test forecast gaps and never drop test rows.
3. Multiply positive power by 1000 for 2024-07-14 06:00 – 2024-08-08 18:00.
4. Remove the 13 spikes > 10.1 MW. Clip marginal overshoot (10,000–10,066) to 10,000, or keep it.
5. Mask the 2,187.44 flatline (2024-09-18/19, 48 h) and the POA 412.6 flatline (2024-08-25..27).
6. Set night negatives to 0 for the target (standby draw isn't forecastable, and the output rule clips to 0 anyway).
7. Exclude STOP, PARTIAL and NaN-status/NaN-power hours from training, since they are availability events that the weather can't predict. Log counts for each.
8. Leave the 186 missing hours missing and do not impute the target.

### Decisions
- **D1. Night values inside the MW window stay as they are; I only rescale positive values.** The negative night values there are already kW-sized (−2 … −10 kW, the same standby draw as elsewhere), so multiplying them by 1000 would create −2 … −10 MW readings.
- **D2. I train and score on normal-availability hours only.** STOP, PARTIAL and missing-status outages are not predictable from a weather forecast; training on them would teach the model noise, and scoring on them would measure outages rather than forecast skill.
- **D3. I treat the forecast columns as given, even though the ambient sensor agrees with the forecast temperature to 1 °C std.** It may mean the "forecast" is close to an analysis; either way the sensor is never a model input, so it cannot leak into predictions.
- **D4. I handle soiling in the target rather than in the features.** The test period (Jul–Aug) is monsoon, when rain keeps the plant clean, so I build a soiling-corrected target (`ac_power_clean`, rule 9) and validate on time-ordered folds that include monsoon months.

## Cleaning

Script: `src/clean.py` (run `.venv\Scripts\python.exe -m src.clean`). It writes `data/train_clean.parquet`,
`reports/clean_summary.csv`, `reports/suspect_outage.csv`, `reports/soiling_factor.png` and `reports/clean_stdout.txt`.
Rows are never deleted. Bad values become NaN and unusable hours get `excl_*` flags and `train_ok = False`.
The only rows removed are the 36 surplus copies of duplicated hours. `ac_power_raw` keeps the post-dedup, pre-fix value.
Expected power (used for cleaning only, never as a feature) is `10.8 × POA × (1 − 0.0035 × (T_module − 25))`, capped at 10,000 kW.

Execution order: 1 → 3 → 2 → 4 … 10. Sentinels are nulled before duplicate resolution so expected power never uses a −999 POA.

| # | Rule | Rows affected |
|---|---|---|
| 1 | Parse ISO + `DD-MM-YYYY HH:MM` (day-first), then sort | 12,589 ISO + 389 DD-MM; 1 ordering break fixed |
| 3 | −999/9999 → NaN, per column only | wind 14, POA 52, module T 39, ambient T 26, power 67 |
| 2 | Duplicates: identical → keep one; power conflict → one copy within 10 % of expected → keep it; both within 10 % → mean; sun below horizon at mid-hour → 0; otherwise NaN | 36 hours: 24 identical, 2 one-copy, 5 mean, 3 night → 0, **2 → NaN** (2024-06-04 07:00, 2024-09-13 06:00); 36 surplus rows removed (12,978 → 12,942) |
| 4 | MW window 2024-07-14 06:00 – 2024-08-08 18:00: assert no positive value > 50, then positive × 1000 | 333 values; median power/POA inside **10.80**, outside 10.10 kW per W/m² ✓ |
| 5 | Power > 10,100 → NaN; (10,000, 10,100] → 10,000 | 13 spikes NaN; 16 clipped |
| 6 | Power flatline 2,187.44 (2024-09-18/19) → NaN; POA flatline 412.6 (2024-08-25..27) → POA NaN only | 48 power; 70 POA |
| 7 | Sun below horizon at mid-hour and power < 0 → 0 | 3,282 |
| 8 | Status strip+upper; flags `excl_stop`, `excl_partial`, `excl_power_nan`, `excl_day_standby` (STANDBY, el > 0, POA > 300), `excl_nan_status_low` (NaN status, POA > 300, actual/expected < 0.5); separate `suspect_outage` (RUN, POA > 300, actual/expected < 0.5), not excluded | 1,219 status strings changed; 120 / 149 / 382 / 0 / 2; suspect_outage **0** |
| 9 | `soiling_factor` = daily median temp-corrected power/POA (train_ok, RUN, POA 400–1300, P 1–9 MW, ≥ 3 h/day) ÷ Jun–Oct median (11.087), 7-day centred rolling median, time-interpolated to all days, capped at 1.0; `ac_power_clean = min(ac_power_kw / soiling_factor, 10000)` | 511 usable days; factor 0.926–1.000; 181 rows capped at 10,000 |
| 10 | 186 missing hours left missing, no target imputation | 186 |
| 11 | *(added post-analysis, runs after rule 7, before rule 8)* `ac_power_kw / expected_kw > 1.5` with measured POA > 100 W/m² → `ac_power_kw` NaN | **10** rows, all isolated single hours; `reports/ratio_spikes.csv` |

Final table:

| item | count |
|---|---|
| rows | 12,942 (+186 missing hours = 13,128) |
| train_ok = True | 12,290 (6,199 of them daylight) |
| train_ok = False | 652 |
| excl_stop | 120 |
| excl_partial | 149 |
| excl_power_nan | 382 (= 242 original NaN + 67 sentinel + 13 spikes > 10.1 MW + 48 flatline + 2 duplicate conflicts + 10 ratio spikes) |
| excl_day_standby | 0 |
| excl_nan_status_low | 2 |
| rows with > 1 reason | 1 |
| suspect_outage (kept) | 0 |

Observations / caveats:
- **Duplicate rule (revised).** At first, both-within-10 % and night conflicts went to NaN (10 hours). Now both-within-10 % uses the mean of the two copies (5 hours) and night conflicts use 0 (3 hours, consistent with rule 7). Only 2 daytime hours, where neither copy is within 10 %, remain NaN.
- **Sentinel count, wind 14 (clean) vs 13 (audit):** 2025-06-26 07:00 is a duplicated hour with −999 wind in *both* copies. Clean counts before de-duplication and the audit counted after, so it is one hour counted twice. The same applies to **POA 52 (clean) vs 51 (audit: −999 ×22 + 9999 ×29)**: 2024-03-02 15:00 is duplicated with −999 in both copies.
- **`excl_day_standby` = 0.** The 18 "STANDBY with POA > 300" hours in the audit were all at night inside the frozen-POA window. Once that POA is nulled, none remain.
- **`suspect_outage` = 0.** Once the MW window, STOP and PARTIAL are handled, no RUN hour with POA > 300 falls below 50 % of expected. There are no hidden outages at this threshold.
- **MW-window ratio check:** 10.80 inside vs 10.10 outside does not mean the rescale is off. The window (mid-Jul to early Aug) is monsoon, when rain keeps the panels clean, so 10.80 is a clean-state value close to the clean baseline of 11.09. The remaining gap is because this check's ratio is not temperature-corrected and includes cloud-edge hours. "Outside" pools all other months, including the soiled dry season, where the ratio sits 5–7 % lower. After ×1000 the window is fully consistent with the rest of the record.
- **Soiling factor** reproduces the sawtooth: it bottoms at 0.93–0.94 before each cleaning (2024-03-01, 05-01, 2025-01-07, 03-09, 05-09) and holds at ≈ 1.0 during Jun–Oct. Mean daylight factor by month: Jul 0.997, Aug 1.000. "439 of 547 days < 1" is inflated because the baseline is a median, so about half the monsoon days sit fractionally below 1.
- `ac_power_clean` is NaN wherever `ac_power_kw` is NaN. The soiling factor is defined for every row.
- **I added rule 11 after the error analysis, as a data-validity fix (D9).** It was not tuned for score.
  - **Trigger:** 2025-06-07 07:00 read 5,325 kW at 192 W/m² POA; the physical maximum is ≈ 2.1 MW. Rule 5 only removes values > 10,100 kW, so impossible values below the rating slipped through.
  - **Inspection first:** `reports/ratio_spikes.csv` lists every row with measured POA > 100 and power / expected > 1.5, with neighbours' power, POA and ratio, sun elevation and status.
    - **10 rows**, ratio 2.0–3.0 (the 99th percentile of the ratio is 1.07), **all isolated single hours**: neither neighbour exceeds 1.5.
    - Sun elevation is 14–62°, so they are not a low-sun artefact. 5 are at 07:00 with 3.6–5.3 MW, when clear sky allows ≈ 2.3 MW.
    - **Neighbour-copy check:** none of the 10 spike values equals, or is within 1 % of, the previous or next hour's power. The closest are 2024-09-02 07:00 (+3.0 % vs 08:00) and 2024-10-23 07:00 (+5.6 % vs 08:00); the rest are 9 % to > 2,000 % away. So this is *not* a simple one-hour timestamp slip, which I had first suspected. The cause is unknown; I treat them as invalid values.
  - **Option chosen: POA > 100** (10 rows). The alternative POA > 200 would catch only 6 and miss the triggering 192 W/m² case.
  - All 10 were RUN / train_ok hours before the rule; they are now `excl_power_nan`.
  - Soiling factor and `ac_power_clean` are unchanged apart from these hours: still 511 usable days, baseline 11.087.

### Decisions (continued)
- **D5. I store the cleaned table as parquet (`pyarrow==25.0.1`).** Parquet keeps dtypes, booleans and timestamps exactly between runs; pyarrow is used for file I/O only, never for modelling.
- **D6. `ac_power_clean = min(ac_power_kw / soiling_factor, 10000)`.** At clipping the inverter caps output; dust only removes DC headroom, so a clean plant still exports at most 10,000 kW (181 rows capped).

## Physics and baselines

Code: `src/validate.py` (folds + metrics), `src/solar.py` (physics chain), `src/train.py` (baselines; run `.venv\Scripts\python.exe -m src.train`).
Outputs: `reports/baselines.csv` (fold × model × target × cloud bucket), `reports/baseline_params.json` (every fitted parameter per fold),
`reports/baselines_stdout.txt`, `reports/baseline_weeks_F1..F8.png` (clearest and cloudiest scored week per fold: actual vs B0 vs B2b).

### Validation design (`src/validate.py`)
> *Superseded as the performance estimate; see `## Leakage-safe validation` for why and for the replacement.*

| Fold | Train | Test | Note |
|---|---|---|---|
| F1 | ≤ 2024-06-30 | 2024-07-01..08-31 | forward monsoon: **no monsoon months in training** (hardest, closest to real operations) |
| F2 | all except test | 2024-07-01..08-31 | leave-out monsoon; uses future data (to 2025-06), so it is optimistic |
| F3..F8 | everything before the month | 2025-01 … 2025-06, one month each | rolling origin |

- **Scoring rows:** train_ok, sun elevation > 0 at mid-hour, target not NaN.
- **Metrics:** MAE and RMSE as % of 10,000 kW; bias in kW (pred − actual); daily energy error = Σ|daily pred energy − daily actual energy| / Σ daily actual energy.
- **Cloud buckets** (0–0.3, 0.3–0.7, 0.7–1): hourly metrics use the hour's forecast cloud cover; daily energy error per bucket uses days whose mean daylight forecast cloud falls in the bucket.
- Every model is fitted on `fold.train_mask` rows only, and applies its own filters (train_ok, …) inside `fit`.
- Test-period caveat for F1/F2: 2024-08-12..16 was a STOP outage, so those days are excluded from scoring (train_ok = False). The week plots only pick weeks with ≥ 8 scored hours on at least 6 of 7 days.

### Physics chain (`src/solar.py`, `PhysicsModel`)
Inputs at predict time are the **timestamp + forecast columns only**. Measured POA/power are used inside `fit()` as calibration targets on training rows; they are never inputs to `predict()`, because measured data is not available at forecast time.
1. **Solar position** at mid-hour (Asia/Kolkata) → apparent zenith, absolute airmass (680 m), extraterrestrial DNI.
2. **Ineichen clear sky** with monthly Linke turbidity (TL):
   - *fitted*: per calendar month, take the top 10 % of training days (at least 3) by daily measured-POA / default-clear-sky-POA. On those days' hours with elevation > 20°, grid-search TL (1.5–8.0, step 0.05) to minimise squared error of transposed clear-sky POA vs measured POA. Months absent from training get pvlib's default TL for that month × the mean fitted/default ratio.
   - *default*: pvlib's Remund climatology lookup (Jan 3.3 … Aug 5.25).
3. **Cloud mapping** on GHI: `GHI = GHI_cs · (1 − a·C^b)`.
   - (a) Kasten-Czeplak a = 0.75, b = 3.4.
   - (b) a, b grid-fitted (least squares) to measured clearness `kt = POA_meas / POA_cs` on training daylight hours (el > 10°).
   - I fit on POA clearness but apply the factor to GHI before decomposition. At 15° tilt the two ratios are close; this is a known approximation.
4. **Erbs decomposition** → DNI/DHI → **Hay-Davies transposition** (tilt 15°, azimuth 180°, albedo 0.2).
5. **Faiman** cell temperature (pvlib default u0 = 25, u1 = 6.84) from forecast temp, forecast wind and estimated POA; temp factor `1 − 0.0035·(T_cell − 25)`.
6. `power = 12500 · POA/1000 · temp_factor · loss_factor`, clipped to [0, 10000], 0 when the sun is below the horizon.
   loss_factor = Σ actual / Σ model over training hours that are train_ok, el > 15°, measured kt ≥ 0.8, forecast cloud ≤ 0.3, and `ac_power_clean` < 9,500 (unclipped). Target: `ac_power_clean`. Each variant fits its own loss factor on the same hour set.
7. Forecast gaps (train wind −999 ×13; test has 19 rows with NaN/−999) are filled by time interpolation within the forecast series (`prepare_forecast`).

> **Superseded (pre-rule-11).** The fitted parameters, results tables and bias notes in this section come from
> `baselines.csv` / `baseline_params.json`, which I produced before cleaning rule 11 and did not re-run. I keep them as
> the record of the baseline decisions. Current B2b (fitted_dry) numbers are in `## Final model choice` → "Re-run after
> cleaning rule 11" and `reports/final_choice*.csv`.

**Fitted parameters** (`baseline_params.json`):
- **Loss factor:** ≈ 0.880–0.889 with fitted TL and 0.832–0.848 with default TL. The loss factor absorbs the difference in clear-sky level, and it is very stable across folds.
- **Fitted TL:** Jan–Jun ≈ 4.4–5.0, well above pvlib's 3.3–4.9. The dry season is hazier than the climatology says.
  - Monsoon months: Jul 5.6, **Aug 8.0 (grid ceiling)**, Sep 6.0. There are no truly clear days in the monsoon, so the "clearest days" still contain cloud, and TL absorbs cloud attenuation. This is a real weakness of the fitted-TL variant exactly where the test period lives. In F1 (no monsoon training), Jul/Aug fall back to 5.8/6.0.
- **Fitted cloud curve (b):** a ≈ 0.56–0.65, b ≈ 4.2–4.7 with fitted TL; a ≈ 0.49–0.53, b ≈ 2.7–3.1 with default TL. Both are weaker than Kasten-Czeplak (0.75): full forecast overcast removes only ~50–65 % of clear-sky irradiance here. The effective attenuation is milder than KC, as expected for a noisy forecast (regression to the mean).

### Results (MAE % of 10 MW, scored daylight hours; full detail in `baselines.csv`)
| Model | F1 kw | F2 kw | F3–F8 mean kw | F3–F8 mean clean |
|---|---|---|---|---|
| B0 climatology | 7.43 | 7.13 | 3.00 | 3.86 |
| B1 clear-sky (fitted TL) | 10.26 | 9.77 | 4.08 | 2.92 |
| B2a KC (fitted TL) | 4.98 | 5.12 | 3.01 | **2.14** |
| B2b fitted cloud (fitted TL) | 4.21 | **4.03** | 3.39 | 2.26 |
| B1_defTL | 9.67 | 9.45 | 3.53 | 3.48 |
| B2a_defTL | 5.24 | 5.32 | 2.91 | 3.04 |
| B2b_defTL | **4.10** | 4.07 | **2.84** | 2.97 |

**Daily energy error:**
- **F1:** B0 14.7 %, B1 23.2 %, B2a 10.6 %, B2b 8.6 %, B2b_defTL 8.3 %.
- **F3–F8 mean vs kw:** B0 5.1 %, B2b_defTL 4.5 %, B2a 4.5 %.

**By forecast cloud bucket** (mean over folds, vs kw, MAE %):
- **Clear hours (C < 0.3):** all physics models ~2.0–2.3 %, vs 5.1 % for climatology.
- **Overcast (C ≥ 0.7):** B1 11.1 %, B2a 7.2 %, B2b 4.2 % (fitted TL) / 5.7 % (default), B0 6.9 %.
- The cloud mapping is where the skill comes from: overcast hours dominate the monsoon error.

**Bias:**
- Physics models are fitted to the clean target. Against `ac_power_kw` in the soiled dry season (F4–F7) they over-predict by +130 … +330 kW; against `ac_power_clean` the bias is ≈ 0 … ±100 kW. The two scores bracket the soiling effect (~3–6 %).
- In **F8 (Jun 2025)** the fitted-TL variants over-predict by +460 (B2b) / +850 (B1) kW. June TL is fitted from June 2024 clear days (4.6), but June 2025 was hazier/cloudier. B2a/B2b_defTL are much better there.
- In **F1**, B0 under-predicts (−418 kW). Jul/Aug borrow the June 2024 climatology, which was cloudier and more outage-hit.

### Which target does the test period resemble?
**Test (Jul–Aug 2025) is monsoon, where soiling_factor ≈ 1** (mean daylight factor Jul 0.997, Aug 1.000). On F1/F2 the scores against `ac_power_kw` and `ac_power_clean` differ by ≤ 0.1 pp for every model. So for the test period the two targets are practically the same, and training on `ac_power_clean` (clean-state plant) is the right choice. The dry-season folds are where the targets diverge; they measure robustness, not test-like skill.

### Takeaways
- **Best simple baseline for the test season:** B2b (physics + fitted cloud curve). F1 MAE 4.1–4.2 %, daily energy error ~8.5 %. Climatology (B0) is clearly worse in the monsoon: 7.4 % MAE, 14.7 % daily energy error.
- **Fitted vs default TL:** there is no consistent winner. Fitted TL helps clean-target scores in the dry season, but it is fragile in cloudy months (Aug TL hits the grid ceiling; F8 bias). A safer option is to fit TL only on dry-season months and keep the climatological seasonal shape for the monsoon, or cap TL.
- The remaining monsoon error is mostly cloud-forecast error (forecast cloud vs measured clearness r = −0.67). An ML residual model on top of B2b (features: cloud, humidity, hour, clear-sky POA, cloud × elevation, daily mean cloud) is the natural next step.

## Models

Code:
- `src/features.py`: features.
- `src/train.py`: `GBMModel`, `run_models`; run with `.venv\Scripts\python.exe -m src.train models`, about 2 min.
- `src/validate.py`: `add_skill` and `mae_pct_of_mean`.

Outputs:
- `reports/model_comparison.csv`
- `reports/model_params.json`
- `reports/models_stdout.txt`
- the baseline rerun: `reports/baselines.csv` and `reports/baselines_stdout.txt`.

> **Superseded (pre-rule-11).** Every table in this section (A–D) comes from `baselines.csv` / `model_comparison.csv`,
> produced before cleaning rule 11 and not re-run. The decisions drawn from them (fitted_dry TL, M2 as the ML model)
> stand; current numbers for B2b and M2 are in `## Final model choice` → "Re-run after cleaning rule 11".

### A. Dry-season diagnosis (baselines, F3..F8)
New metrics in `validate.py`, for all models:
- `skill_vs_B0` = 1 − MAE / MAE_B0, matched on fold, scoring target and cloud bucket.
- `mae_pct_of_mean` = MAE / mean daylight actual power.

Reading the table: for any model, bias vs raw − bias vs clean = mean(clean − raw), the soiling gap on the scored rows. So the question is which target each model is unbiased against.

| fold | soiling factor | gap kW | B2b bias vs clean / raw | B0 bias vs clean / raw |
|---|---|---|---|---|
| F3 Jan-25 | 0.980 | 120 | −103 / +17 | −14 / +106 |
| F4 Feb-25 | 0.947 | 326 | −56 / **+270** | **−312** / +13 |
| F5 Mar-25 | 0.961 | 215 | −33 / **+182** | **−149** / +66 |
| F6 Apr-25 | 0.948 | 277 | +37 / **+315** | **−316** / −38 |
| F7 May-25 | 0.978 | 114 | +217 / +331 | −12 / +102 |
| F8 Jun-25 | 0.995 | 26 | +463 / +489 | −674 / −648 |

(B2b here is the fitted-TL version from the earlier baseline table.)

**Hypothesis confirmed for the soiled months (F4–F6):**
- B2b's loss factor is fitted on `ac_power_clean`, and it is unbiased against clean power (−56 … +37 kW). It over-predicts raw power by about the soiling gap (+182 … +315 kW), and that alone makes its raw-target MAE worse than B0's (3.4 vs 1.6 %).
- B0 is fitted on raw power, which is soiled in exactly these months of the previous year, so it absorbs soiling implicitly. It is unbiased against raw power and under-predicts clean power by 150–316 kW.

**Rejected for F7/F8:** there the soiling factor is ≈ 1, and B2b's over-prediction vs clean (+217 / +463 kW) comes from turbidity/weather (June 2025 hazier than the June 2024 fit), not soiling.

Against the clean target, B2b beats B0 in every dry fold (skill +0.02 … +0.67).

**Turbidity decision → `fitted_dry`.** This is the fitted monthly TL on non-monsoon months; Jun–Sep use pvlib's default × the mean fitted/default ratio.

| B2b variant | F1 / F2 MAE vs kw | F3–F8 mean MAE vs clean |
|---|---|---|
| fitted TL | 4.21 / 4.03 (mean 4.12) | 2.26 |
| default TL | 4.10 / 4.07 (mean 4.09) | 2.97 |
| fitted_dry TL | 4.38 / 4.06 (mean 4.22) | **2.10** |

- **F1/F2:** the three options are tied (0.13 pp spread, smaller than the F1-vs-F2 spread of the same model). The tie rule would pick default, but the clean-target dry folds clearly favour fitting (by 0.7–0.9 pp), so the fitted family wins.
- **Why `fitted_dry` over plain `fitted`:**
  - With plain `fitted`, the final model trained on all data would use the Aug TL fitted from monsoon "clearest days" = **8.0 (grid ceiling)**, and Jul 5.6. These values are **never validated**: in F1 and F2, Jul/Aug are absent from training, so both folds only exercised the fallback path. `fitted_dry` makes the final model use exactly the configuration F1/F2 validated.
  - `fitted_dry` is also best on the clean dry folds, and it fixes the F8 June over-prediction (4.20 vs 5.38 MAE vs kw).
- **Final TL table** (all training data): Jan 4.9, Feb 4.85, Mar 4.7, Apr 4.45, May 4.35, **Jun 5.89, Jul 6.07, Aug 6.31, Sep 5.95** (fallback), Oct 4.9, Nov 4.85, Dec 4.8.

### B. Features (`src/features.py`, forecast + timestamp only)
- **Forecast values:** `cloud`, `temp`, `wind`, `rh`.
- **Cloud context:** `cloud_s1` / `cloud_s2` = centred ±1 h / ±2 h mean **within the same calendar day** (computed on a full hourly grid, so missing rows don't shift the window). `cloud_day_mean` = mean daylight forecast cloud of that calendar day.
- **Geometry:** `elevation`, `azimuth`, `cos_aoi` (15°/180° plane).
- **Physics:** `cs_poa`, `phys_poa`, `cell_temp`, `phys_power`, from B2b with `fitted_dry` TL, fitted per fold on its training rows.
- **Calendar:** `hour`, `doy_sin`, `doy_cos`. No year or trend features.
- A hard assert rejects any `measured_*`, `rainfall`, `status` or `ac_power*` column.
- **Forecast gap filling** (`solar.prepare_forecast`, also used by the physics chain; replaces the earlier cross-day interpolation):
  1. time interpolation within the same calendar day;
  2. then the mean of the same hour on the previous and next day;
  3. then the nearest value within the day.
  - Rows are never dropped.
  - Test's 19 gaps: 18 filled by same-day interpolation and 1 by the adjacent-day rule (`forecast_temp_c` 2025-08-01 00:00).
  - The change has no measurable effect on train baseline scores.

### C. Models
Training rows: train_ok, sun elevation > 0, target not NaN. Night is forced to 0 and predictions are clipped to [0, 10000].
- **M1:** LightGBM, target = power. Monotone decreasing in `cloud`, `cloud_s1`, `cloud_s2`, `cloud_day_mean` ("advanced" method).
- **M2:** target = power / `phys_power`, training hours with `phys_power` < 200 kW skipped. Prediction = `phys_power` × ratio (ratio clipped ≥ 0); for hours with `phys_power` < 200 kW the prediction is `phys_power` itself.
- **M3:** target = power − `phys_power`; prediction = `phys_power` + residual.
- Each model is trained on `ac_power_kw` (`_kw`) and on `ac_power_clean` (`_clean`).
- **Hyper-parameters:** fixed `learning_rate` 0.03, `min_child_samples` 20, `feature_fraction` 0.9, bagging 0.8/1, `lambda_l2` 1, seed 42, deterministic.
  - `num_leaves` ∈ {15, 31} and `n_estimators` (early stopping, patience 100, max 3000) are chosen on the **last 20 % in time of each fold's own training period**, then the model is refit on all training rows.
  - **Nothing is tuned on F1/F2 test data.** The chosen leaves/iterations per fold are in `model_params.json`. F2's inner validation is Mar–Jun 2025, so it stops early (50–170 iterations).

### D. Results
**MAE %** (F1/F2 vs `ac_power_kw`; F3–F8 vs own training target and vs raw):

| model | F1 | F2 | **F1/F2 mean** | F3–F8 vs own target | F3–F8 vs raw |
|---|---|---|---|---|---|
| B0 climatology | 7.43 | 7.13 | 7.28 | 3.00 | 3.00 |
| B2b physics (fitted_dry) | 4.38 | 4.06 | 4.22 | 2.10 | 3.23 |
| M1_kw | 4.57 | 4.64 | 4.61 | 2.19 | 2.19 |
| M1_clean | 4.66 | 4.44 | 4.55 | 1.89 | 2.85 |
| M2_kw | 4.19 | 4.18 | 4.18 | 2.16 | 2.16 |
| **M2_clean** | **4.10** | **4.05** | **4.07** | **1.80** | 2.95 |
| M3_kw | 4.23 | 4.25 | 4.24 | 2.13 | 2.13 |
| M3_clean | 4.33 | 4.12 | 4.23 | 1.82 | 2.96 |

**F1/F2 mean, other metrics (vs raw):**

| model | RMSE % | bias kW | skill vs B0 | daily energy error |
|---|---|---|---|---|
| B0 | 10.20 | −151 | 0 | 13.9 % |
| B2b | 6.28 | +195 | 0.42 | 8.7 % |
| M1_kw | 6.40 | −150 | 0.37 | 8.9 % |
| M2_kw | 5.94 | −98 | 0.43 | 8.4 % |
| **M2_clean** | **5.82** | **−56** | **0.44** | **8.1 %** |
| M3_clean | 5.99 | −100 | 0.42 | 8.5 % |

**F1/F2 by forecast cloud bucket** (MAE % / bias kW):

| model | C < 0.3 | 0.3–0.7 | ≥ 0.7 |
|---|---|---|---|
| B0 | 9.5 / −937 | 8.0 / −681 | 6.7 / +238 |
| B2b | **1.4** / −18 | 3.6 / +264 | 4.8 / +161 |
| M1_kw | 3.2 / −207 | 3.9 / −124 | 5.2 / −164 |
| M2_clean | 1.9 / −86 | **3.4** / −35 | **4.7** / −69 |

**Observations:**
- **ML adds only modest skill over physics in the monsoon.** The best ML model (M2_clean) improves B2b by 0.15 pp MAE (−3.5 %) and 0.46 pp RMSE (−7 %). It removes most of B2b's +195 kW over-prediction, especially in partly-cloudy hours (+264 → −35 kW), and cuts daily energy error 8.7 → 8.1 %. The overcast bucket (≥ 0.7) dominates the monsoon error and barely improves (4.8 → 4.7 %). Error there is set by the forecast cloud's own skill (r = −0.67 vs measured clearness).
- **M1 (plain GBM) is worse than physics in the monsoon** (4.61 vs 4.22), even with monotone cloud constraints. Trees cannot extrapolate to monsoon cloudiness/seasonality that F1 never saw, and even in F2 they under-predict (−150 kW) and are weak in clear hours (3.2 vs 1.4 %). Anchoring on physics (M2/M3) is what makes ML safe in the unseen season.
- **Dry folds:** `_kw` models beat B0 and B2b against raw power (2.1–2.2 vs 3.0 / 3.2), because they learn the average soiling level. `_clean` models are best against the clean target (1.80–1.89). This is the soiling story from section A, and it doesn't matter for Jul–Aug.
- **Differences among M2_clean, M2_kw, M3_clean and B2b on F1/F2 are ~0.1–0.15 pp.** That is of the same order as the F1-vs-F2 spread, so the ranking among them is not strongly significant. Picking the best of several models on F1/F2 also makes its F1/F2 score slightly optimistic.

### Choice: **M2_clean** (LightGBM on the ratio power / B2b-physics, trained on `ac_power_clean`, TL `fitted_dry`)
1. **Best on the monsoon evidence (F1/F2, weighted most).** Lowest MAE (4.07 %), RMSE (5.82 %), daily energy error (8.1 %) and |bias| (56 kW); skill vs B0 0.44. It also wins the 0.3–0.7 and ≥ 0.7 cloud buckets, which is where the cloudier test period (mean forecast cloud 0.80 vs 0.70) will sit.
2. **Best on the dry folds against its own target** (1.80 %, skill 0.53), so it is not a monsoon-only fluke.
3. **The clean target is right for the test period.** Jul–Aug is monsoon with soiling_factor ≈ 1, and F1/F2 scores vs raw and clean differ by ≤ 0.1 pp. Training on `ac_power_clean` removes a dry-season soiling signal the model otherwise has to learn from doy/season, which it would wrongly apply to 2025.
4. **The ratio formulation is physically anchored.** The GBM only rescales a physics forecast, so in unseen conditions it degrades towards B2b rather than towards the training mean (unlike M1).

- **Fallbacks:** M2_kw is near-equivalent (4.18 %). B2b alone (4.22 %) is a strong, fully transparent backup.
- **Final fit plan:** train M2_clean on all train_ok rows with `num_leaves` / `n_estimators` chosen on the last 20 % of the full training period (same rule as in the folds), predict test, clip, and set night to 0.

### Decisions (continued)
- **D7. For hours with `phys_power` < 200 kW (dawn/dusk, heavy overcast) M2 returns plain physics.** The ratio target is unstable when the denominator is small, and these hours carry little energy, so a separate floor or blend is not worth the extra complexity.

## Final model choice

Code: `src/train.py` → `run_final_choice` (run `.venv\Scripts\python.exe -m src.train final_choice`, ~30 s).
Outputs: `reports/final_choice.csv` (fold × model × target × bucket), `reports/final_choice_summary.csv`,
`reports/final_choice_params.json`, `reports/final_choice_stdout.txt`. No new model families.

**Setup:**
- **Monsoon evidence = F1, F2 and F8** (June 2025, monsoon onset). Every summary table now has a "monsoon mean" column. `run_models` prints it too; the values below were back-computed from the saved CSVs.
  - *(Superseded, pre-rule-11)* Monsoon mean MAE vs kw, earlier tables: B0 7.66, B2b(fitted_dry) 4.22, M1_kw 4.67, M1_clean 4.51, M2_kw 4.36, **M2_clean 4.17**, M3_kw 4.37, M3_clean 4.27.
  - Baseline TL variants: B2b fitted 4.54, default 4.06, fitted_dry 4.22.
- **M2_clean_fixed:**
  - `num_leaves` = 15; `n_estimators` = **295**, the median of the M2_clean early-stopping iterations F1..F8 (258, 84, 1381, 705, 1317, 115, 75, 332) from the pre-rule-11 `model_params.json`. No early stopping. The value was fixed then and kept after rule 11 (the post-rule-11 tuned iterations in `final_choice_stdout.txt` differ; they are not used).
  - 5 seeds (42–46), predictions averaged. The seed affects bagging (0.8) and feature_fraction (0.9) sampling.
  - Same features, `fitted_dry` physics and ratio target as M2_clean.
- **Blends** (fixed rules, no weight fitting or search):
  - `BL50 = 0.5·B2b + 0.5·M2_clean_fixed`
  - `BL_cloud = w·B2b + (1−w)·M2_clean_fixed`, with w = 0.7 if `cloud_s1` < 0.3 else 0.3. `cloud_s1` = forecast cloud smoothed ±1 h within the calendar day; I chose ±1 h as "smoothed cloud".

> **Superseded (pre-rule-11):** the stability table, the comparison table, the per-fold BL50 line, the cloud-bucket
> table, the numbers in "Why blending helps", the decision-rule numbers (3.927) and the turbidity trade-off numbers below
> were all computed before cleaning rule 11. Conclusions are unchanged. **Current numbers:** "Re-run after cleaning
> rule 11" below, `reports/final_choice_summary.csv` and `reports/final_choice_stdout.txt`.

**Stability of the fixed config vs the tuned one** *(superseded, pre-rule-11)* (MAE %; F1/F2 vs kw, F3–F8 vs clean):

| fold | F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 |
|---|---|---|---|---|---|---|---|---|
| tuned M2_clean | 4.10 | 4.05 | 1.19 | 1.08 | 1.05 | 1.07 | 1.89 | 4.51 |
| fixed (15 leaves, 295 it, 5 seeds) | 4.21 | 4.21 | 1.11 | 1.09 | 1.02 | 1.09 | 1.76 | 4.46 |
| diff | +0.11 | +0.17 | −0.08 | +0.01 | −0.02 | +0.01 | −0.13 | −0.06 |

The fixed config is **stable**: every fold is within ±0.17 pp, the monsoon mean is 4.23 vs 4.17 and the F3–F8 clean mean 1.75 vs 1.80, with no systematic sign. The tuned version picked between 75 and 1,381 iterations, which is itself a sign that the inner-validation early stopping was noisy. A single fixed config is easier to defend and reproduce.

**Comparison** *(superseded, pre-rule-11)*:

| model | monsoon MAE | monsoon RMSE | monsoon bias kW | monsoon daily energy err | F3–F8 MAE (clean) | F3–F8 RMSE | F3–F8 bias | F3–F8 daily energy err |
|---|---|---|---|---|---|---|---|---|
| B2b | 4.22 | 6.39 | +206 | 8.45 % | 2.10 | 3.55 | +48 | 3.18 % |
| M2_clean (tuned) | 4.17 | 6.01 | −53 | 8.32 % | 1.80 | 3.22 | −1 | 3.01 % |
| M2_clean_fixed | 4.23 | 6.06 | −84 | 8.40 % | **1.75** | **3.16** | −17 | 2.91 % |
| **BL50** | **3.93** | **5.82** | +61 | **7.73 %** | 1.81 | 3.23 | +16 | **2.83 %** |
| BL_cloud | 3.98 | **5.82** | **+3** | 7.84 % | 1.85 | 3.24 | −4 | 2.92 % |

(Monsoon = mean of F1, F2, F8, scored vs `ac_power_kw`. F3–F8 are scored vs `ac_power_clean`.)
Per-fold MAE vs kw: BL50 F1 4.00, F2 3.88, F8 3.90. It is best or tied-best in each of the three monsoon folds.

**Monsoon mean by forecast cloud bucket** *(superseded, pre-rule-11)* (MAE % / bias kW):

| model | C < 0.3 | 0.3–0.7 | ≥ 0.7 |
|---|---|---|---|
| B2b | **1.63** / −72 | 3.71 / +243 | 5.01 / +239 |
| M2_clean_fixed | 1.84 / −108 | 3.90 / −115 | 4.64 / −24 |
| BL50 | 1.70 / −90 | **3.43** / +64 | 4.56 / +108 |
| BL_cloud | 1.77 / −94 | 3.52 / −7 | **4.53** / +55 |

**Why blending helps** *(numbers pre-rule-11; post-rule-11 the same pattern holds: B2b +208 kW, M2 −113 kW, BL50 +47 kW)*:
- **Bias cancellation:** B2b over-predicts in the monsoon (+206 kW) and M2 under-predicts (−84 kW). Averaging brings the bias to +61 kW.
- **Imperfectly correlated errors:** the gain is largest in partly-cloudy hours (3.71 / 3.90 → 3.43).
- **Caveat:** part of the gain depends on B2b and M2 keeping opposite-sign biases in Jul–Aug 2025. If both drift the same way, BL50 loses that part of its advantage but is never worse than the worse of the two.
- In clear hours physics alone is best (1.63), and BL_cloud leans towards it there. It still does not beat BL50 overall.

**Decision rule** (stated before the results; numbers below *superseded, pre-rule-11*; post-rule-11 result in the re-run subsection): pick the simplest model within 0.1 MAE points of the best monsoon mean, in the order B2b < BL50 < BL_cloud < M2_clean_fixed.
- Best monsoon mean: **3.927 (BL50)**. Within 0.1: BL50 (3.93) and BL_cloud (3.98).
- B2b (4.22) and M2_clean_fixed (4.23) are 0.29–0.30 behind, outside the tolerance.
- → **Winner: BL50** = 0.5 × B2b physics (fitted_dry TL, fitted cloud curve) + 0.5 × M2_clean_fixed (LightGBM ratio on physics, 15 leaves, 295 iterations, 5 seeds, target `ac_power_clean`).
- **Supporting evidence (beyond the rule):**
  - best monsoon RMSE (tied) and best monsoon daily energy error (7.73 %);
  - within 0.06 pp of the best dry-fold MAE and best dry daily energy error (2.83 %);
  - it needs no fitted weights.
- **Caveat:** the five candidates were compared on the same folds used to choose between them, so the monsoon score of the winner is slightly optimistic. The rule and tolerance were fixed beforehand to limit this.

**Turbidity trade-off (carried into BL50 through both components)** *(numbers superseded, pre-rule-11; current: B2b fitted_dry 4.17 vs default 4.02, see Sensitivity below)*:
- The `fitted_dry` TL was chosen on dry-fold evidence: B2b F3–F8 clean MAE 2.10 vs 2.97 default.
- It avoids the unvalidated Aug TL = 8.0 that plain `fitted` would use in the final all-data fit.
- **Cost on monsoon evidence (B2b, MAE vs kw):**
  - F1/F2 mean: 4.22 (fitted_dry) vs **4.09** (default).
  - **With F8 counted as monsoon: 4.22 vs 4.06**, a gap of 0.16 pp, which is larger than the 0.1 decision tolerance used above.
- So on the monsoon folds alone, default TL would have been the better physics component. I did not switch; see D8.

### Decisions (continued)
- **D8. I kept `fitted_dry` turbidity, although with F8 counted as monsoon default TL beats it by 0.16 pp on B2b (pre-rule-11; 0.15 pp after).** I chose `fitted_dry` before F8 was reclassified as a monsoon fold; switching afterwards would mean choosing on the evaluation folds. The sensitivity check below confirms the choice is immaterial inside BL50 (post-rule-11: 3.87 vs 3.91).

**Final model (committed):** BL50 = 0.5 × B2b + 0.5 × M2_clean_fixed, turbidity `fitted_dry` (D8).

### Re-run after cleaning rule 11 (D9)
Rule 11 NaNs 10 physically impossible power spikes; see `## Cleaning`. I re-ran the pipeline: clean → final_choice → analysis. **Selection is closed: BL50 stays the final model regardless** (the pre-stated decision rule would also still pick it: best 3.874 = BL50; within 0.1: BL50, BL_cloud).

Monsoon mean (F1, F2, F8), vs `ac_power_kw`:

| model | MAE % before | MAE % after | bias kW before | bias kW after |
|---|---|---|---|---|
| B2b | 4.22 | 4.17 | +206 | +208 |
| M2_clean_fixed | 4.23 | 4.24 | −84 | −113 |
| **BL50** | **3.93** | **3.87** | +61 | +47 |

- BL50 per fold after: F1 3.92, F2 3.86, F8 3.84.
- Two things changed:
  1. The spikes no longer sit in the scored hours. This is why B2b improves with no change to its fit.
  2. M2 no longer learns from them, so its bias moves down by 30 kW.
- The tables above this subsection (comparison, stability, buckets) are the pre-rule-11 numbers and are marked superseded; the conclusions are unchanged.

**Current comparison (post-rule-11, `reports/final_choice_summary.csv`):**

| model | monsoon MAE | RMSE | bias kW | daily energy err | F3–F8 MAE (clean) | RMSE | bias kW | daily energy err |
|---|---|---|---|---|---|---|---|---|
| B2b | 4.17 | 6.27 | +208 | 8.56 % | 2.03 | 3.08 | +50 | 3.15 % |
| M2_clean (tuned) | 4.29 | 6.05 | −103 | 8.80 % | 1.71 | 2.67 | +4 | 2.96 % |
| M2_clean_fixed | 4.24 | 5.93 | −113 | 8.63 % | **1.69** | **2.60** | −14 | 2.89 % |
| **BL50** | **3.87** | **5.63** | +47 | **7.77 %** | 1.74 | 2.68 | +18 | **2.77 %** |
| BL_cloud | 3.95 | 5.64 | **−17** | 7.93 % | 1.79 | 2.71 | −2 | 2.90 % |

Monsoon mean by forecast cloud bucket, post-rule-11 (MAE % / bias kW): B2b 1.65 / −75, 3.66 / +245, 4.98 / +240;
M2_clean_fixed 1.78 / −133, 3.89 / −142, 4.71 / −61; **BL50 1.64 / −104, 3.38 / +51, 4.54 / +90**;
BL_cloud 1.75 / −105, 3.48 / −26, 4.53 / +30.

### Sensitivity: turbidity (NOT used for selection; after rule 11)
From `src/analysis.py` → `reports/sensitivity_turbidity.csv`. MAE % vs `ac_power_kw`:

| model | TL | F1 | F2 | F8 | monsoon mean | bias kW |
|---|---|---|---|---|---|---|
| BL50 | **fitted_dry (final)** | 3.92 | 3.86 | 3.84 | **3.87** | +47 |
| BL50 | default | 3.94 | 3.98 | 3.80 | 3.91 | −27 |
| B2b | fitted_dry | 4.34 | 4.04 | 4.12 | 4.17 | +208 |
| B2b | default | 4.07 | 4.04 | 3.94 | 4.02 | +26 |
| M2_clean_fixed | fitted_dry | 4.13 | 4.28 | 4.32 | 4.24 | −113 |
| M2_clean_fixed | default | 4.13 | 4.13 | 3.91 | 4.06 | −80 |

- Inside the blend, the turbidity choice is **immaterial**: 3.87 vs 3.91, and neither wins every fold.
- Default TL makes B2b less biased, which removes part of the bias cancellation that helps BL50. The final model is therefore robust to the one choice (D8) that was contestable on monsoon evidence.

## Expected test performance
> *Derived from the superseded folds; not re-derived from the leakage-safe validation. Treat it as indicative only.*


**Method:**
- Take BL50's MAE and bias per forecast-cloud bucket on the monsoon folds (mean of F1, F2, F8, after rule 11).
- Weight them by the share of **test** daylight hours (sun elevation > 0 at mid-hour, 806 hours) in each bucket.
- `reports/expected_test_mae.csv`.

| bucket | BL50 MAE % | BL50 bias kW | share of fold hours | share of test hours |
|---|---|---|---|---|
| 0–0.3 | 1.64 | −104 | 4.1 % | 2.7 % |
| 0.3–0.7 | 3.38 | +51 | 43.0 % | 20.5 % |
| 0.7–1 | 4.54 | +90 | 52.8 % | **76.8 %** |

- **Expected test MAE ≈ 4.2 % of 10 MW (4.22 %)**, vs 3.92 % for the same bucket MAEs at the folds' own cloud mix. The test period is much cloudier: mean forecast cloud 0.80, three quarters of daylight hours ≥ 0.7. Its error will be dominated by the overcast bucket.
- **Expected test bias ≈ +77 kW (slight over-prediction)**, vs +65 kW at the folds' mix.
  - That is ≈ 0.8 % of rating, or roughly 2 % of mean monsoon daylight power.
  - The shift comes from the overcast bucket carrying most of the test weight, where BL50 over-predicts by ~90 kW. The only under-predicting bucket (clear, −104 kW) is almost absent from the test period.
  - I apply no correction: bias per bucket is estimated on only three monsoon months, and correcting it would mean fitting on the evaluation folds.
- **Finer check inside 0.7–1:**
  - The test sits in the upper part of that bucket: 0.7–0.8 15 %, 0.8–0.9 27 %, 0.9–1 35 % of test hours.
  - Pooled BL50 MAE in those sub-buckets is 4.0 / 5.1 / 4.2 %; the 0.9–1 sub-bucket has only 150 fold hours.
  - Re-weighting with sub-buckets gives ≈ 4.2 %. **Expected range ≈ 4.0–4.5 %** MAE (≈ 400–450 kW mean absolute hourly daylight error).
- **Daily energy error** on the monsoon folds is 7.8 %; expect roughly 8–9 % on a cloudier test.
- **Why the real score could be worse:**
  1. The folds are scored on train_ok hours only. If Jul–Aug 2025 contains outages/curtailment (the Aug-2024 STOP week), the test score against raw export will include errors no weather model can avoid.
  2. The bucket MAEs come from only three monsoon months (two of them the same Jul–Aug 2024 period).
  3. Heavy-overcast skill depends on the cloud forecast itself, whose skill in 2025 is unknown.
- **Why it could be better:** absolute error scales with power, and overcast hours produce less power.

## Error analysis

Code: `src/analysis.py` (run `.venv\Scripts\python.exe -m src.analysis`). Numbers are after cleaning rule 11.
Outputs:
- `reports/analysis_stdout.txt`
- `reports/worst_days.csv`, `reports/worst_hours.csv`, `reports/daily_errors_monsoon_folds.csv`
- figures in `reports/worst_days/*.png`
Scope: BL50 predictions on F1, F2 and F8, scored vs `ac_power_kw`, train_ok daylight hours. F1 and F2 share the same test dates, so each date is counted once (the fold with the larger error). Measured POA is used here for diagnosis only.

**5 worst days** (by |daily energy error|):

| date | fold | mean fc cloud | actual MWh | BL50 MWh | error | kt meas − kt fcst | forecast / model part of error |
|---|---|---|---|---|---|---|---|
| 2025-06-23 | F8 | 0.63 | 44.9 | 60.4 | **+35 %** | −0.19 | +31 % / +4 % |
| 2024-08-09 | F1 | 0.83 | 39.3 | 53.0 | +35 % | −0.15 | +33 % / +1 % |
| 2024-08-20 | F1 | 0.67 | 44.4 | 58.0 | +31 % | −0.15 | +27 % / +3 % |
| 2025-06-22 | F8 | 0.78 | 41.6 | 51.9 | +25 % | −0.16 | +22 % / +2 % |
| 2024-08-21 | F1 | 0.90 | 37.5 | 47.3 | +26 % | −0.06 | +23 % / +3 % |

(F2 errors on the same dates: 2024-08-09 +30 %, 08-20 +27 %, 08-21 +23 %.)

**All five are over-predictions:** the day was darker than the forecast implied. There are no big under-prediction days.

**Classification:**
- **The pre-stated 0.3 rule was uninformative.** The rule: "forecast wrong" if the mean daylight |measured clearness − forecast-implied clearness| > 0.3.
  - Result: **0 forecast wrong, 5 model wrong.**
  - No fold-day in F1/F2/F8 reaches 0.3 (median 0.11, max 0.22). The fitted cloud curve 1 − a·C^b **compresses forecast-implied clearness to 0.47–1.0**, so the forecast cannot "disagree" by 0.3 on average by construction.
  - I recorded the rule's result as stated rather than re-tune the threshold after seeing it.
- **Measured-POA attribution classes all 5 worst days as forecast misses.**
  - Method: drive the same B2b physics chain with the **measured** POA instead of the forecast ("oracle irradiance", dotted line in the figures). If the oracle matches actual, the power model is right and the miss is in the irradiance forecast.
  - On all five days the oracle is within +1 … +4 % of actual energy, while the forecast part is +22 … +33 %.
  - → **5 of 5 "forecast wrong".** The plant model (loss, temperature, transposition) is not the problem on bad days; the forecast of how much sun reaches the panels is.

**3 largest single-hour errors** (after rule 11):

| hour | fc cloud | measured kt | actual kW | BL50 kW | physics on measured POA | error |
|---|---|---|---|---|---|---|
| 2025-06-23 10:00 (F8) | 0.64 | 0.57 | 4,643 | 7,126 | 4,610 | +2,483 |
| 2024-08-20 12:00 (F1) | 0.71 | – (POA missing) | 6,101 | 8,425 | – | +2,323 |
| 2024-08-20 11:00 (F1) | 0.85 | 0.46 | 4,519 | 6,719 | 4,528 | +2,200 |

- All three are "darker than forecast" hours on worst days. Where POA exists, physics on measured POA hits actual almost exactly.
- **Before rule 11**, the largest hourly error was 2025-06-07 07:00 (−3,203 kW): actual 5,325 kW at 192 W/m², a data error. It is now NaN (cleaning rule 11, D9).

**Walkthrough day: 2025-06-23 (F8, June 2025, +35 %)** (`reports/worst_days/2025-06-23_F8.png`)
- The forecast said "partly cloudy" all day: daylight cloud 0.49–0.82, mostly 0.55–0.70.
- Through the fitted cloud curve, which attenuates weakly at mid cloud values, that means an almost clear day (implied clearness 0.88–0.98 at every hour except 08:00, 0.76). BL50 predicted 60 MWh, a near-clear profile peaking at ≈ 8.5 MW.
- In reality the morning was overcast (measured clearness 0.53–0.57 for the 08:00–10:00 hours) and midday only 0.6–0.75, so the plant made 44.9 MWh.
- Driving the same physics with the measured POA reproduces actual power hour by hour (+4 %). The plant, the loss factor and the temperature model behaved as expected; the miss is the cloud forecast.
- Both components failed together (B2b +40 %, M2 +29 %), because both only see forecast cloud. A blend can't fix an input that is wrong.
- The lesson: "partly cloudy" monsoon forecasts carry the most downside risk. Fixing that needs a better cloud forecast (or ensemble spread), not a better power model.

### Error by hour of day (BL50, F1/F2/F8 vs `ac_power_kw`, after rule 11)
`reports/error_by_hour.csv`, `reports/error_by_period.csv`, `reports/error_by_hour.png`

| period | hours | MAE % of 10 MW | MAE % of mean actual | BL50 bias kW | B2b bias | M2 bias |
|---|---|---|---|---|---|---|
| morning (hour < 10) | 547 | 2.5 | 9.3 | +29 | +114 | −57 |
| midday (10–13) | 558 | 6.4 | 8.7 | **+84** | +358 | −190 |
| afternoon (≥ 14) | 689 | 2.9 | 8.9 | +19 | +146 | −108 |

**Patterns:**
- **Over-prediction peaks at midday and scales with irradiance.**
  - B2b is +340 … +400 kW at hours 10–14 (≈ 5 % of mean actual), a multiplicative optimism consistent with the worst-day story: mid-value forecast cloud is mapped to too much sun.
  - M2 corrects in the opposite direction, most strongly at 12–16 (−140 … −230 kW).
  - BL50 keeps a residual bias of +55 … +123 kW at hours 9–14. By fold, the midday bias is F1 +163, F8 +188 and F2 −45.
- **Relative error is flat through the day** (8.7–9.3 % of mean actual power). The large midday absolute MAE is just where the energy is. There is no timing/shift problem: the diurnal shape is right, only the amplitude on cloudy days is wrong.
- **Mild morning > afternoon asymmetry in B2b:** hour 9 +268 kW (5.1 % of mean actual) vs hour 15 +201 kW (3.9 %). BL50 carries a trace of it (+103 kW at hour 9, +12 kW at hour 15).
- **Implication:** BL50 is slightly optimistic around midday in the monsoon, by ≈ 1–1.5 % of mean power. This is consistent with the +77 kW expected test bias. I apply no correction: it would have to be fitted on the same folds used to measure it.

### Decisions (continued)
- **D9. I added cleaning rule 11 (power / expected > 1.5 with measured POA > 100 → NaN) after the error analysis.** The largest hourly error, 2025-06-07 07:00, was a 5,325 kW reading at 192 W/m², far above what the irradiance allows, and rule 5 only caught values above the rating. I inspected all candidates first (`reports/ratio_spikes.csv`): 10 isolated hours, none a copy of a neighbour. The threshold was set on physical grounds, not tuned for score; see `## Cleaning`.

## Leakage-safe validation

Code: `experiments/leakage_safe_validation.py` (run `.venv\Scripts\python.exe experiments\leakage_safe_validation.py`); method
details in `experiments/README.md`. Outputs in `experiments/results/`: `validation_predictions.csv`, `fold_results.csv`,
`model_summary.csv`, `fold_audit.csv`. This is the **final reported validation**. It replaces the original folds as the
performance estimate. It does not change the model.

### Why the original validation was superseded
The original validation (`src/validate.py`, folds F1–F8; BL50 monsoon mean 3.87 %) had temporal/data leakage:
- **Cleaning ran on the whole training file before the folds were cut.** The soiling correction in `src/clean.py` uses a
  centred 7-day rolling median, a Jun–Oct baseline over all data, and interpolation/back-fill. So a fold's training
  target `ac_power_clean` depended on data from after its validation window.
- **F2 trained on data after its validation window** (leave-out monsoon; already flagged above as optimistic).
- **Forecast gap filling** (`prepare_forecast`) can fall back to the same hour on the *next* day's forecast.
- **Scoring was against cleaned targets** (MW window rescaled, and `ac_power_clean` on the dry folds), not raw actuals.
- The model choice (BL50) and M2's 295 rounds were also selected on these same folds.

### Six chronological folds
Each fold trains on all rows from 2024-01-01 up to the cutoff and validates on the next two months:

| Fold | Train up to | Validate | Scored hours (all / daylight) | M2 training rows |
|---|---|---|---|---:|
| F1 | 2024-06-30 23:00 | 2024-07-01 .. 2024-08-31 | 724 / 379 | 2,012 |
| F2 | 2024-08-31 23:00 | 2024-09-01 .. 2024-10-31 | 1,211 / 595 | 2,631 |
| F3 | 2024-10-31 23:00 | 2024-11-01 .. 2024-12-31 | 1,427 / 680 | 3,169 |
| F4 | 2024-12-31 23:00 | 2025-01-01 .. 2025-02-28 | 1,281 / 581 | 3,768 |
| F5 | 2025-02-28 23:00 | 2025-03-01 .. 2025-04-30 | 1,426 / 738 | 4,350 |
| F6 | 2025-04-30 23:00 | 2025-05-01 .. 2025-06-30 | 1,351 / 723 | 5,011 |

### Leakage safeguards
- Raw rows are cut at the fold boundary **before** any target cleanup, calibration or fitting. The script asserts
  `max(training timestamp) < min(validation timestamp)` and that training and validation timestamps are disjoint.
- Everything fitted is fitted on the fold's training rows only, then frozen: Linke turbidity (`fitted_dry`), the cloud
  curve `a, b`, the loss factor, B0 climatology and the LightGBM models.
- Training cleanup applies only row-local rules: sentinels, MW → kW, over-rating, the known flatlines, night negatives,
  POA/power ratio spikes, and the duplicate rule. There is **no soiling normalisation**, because it needs future data.
- Forecast gaps are filled by time interpolation **within the same calendar day** only, with no adjacent-day fallback.
  7 wind forecast entries were filled this way, and no rows were dropped for unusable forecasts.
- Validation actuals stay **raw**. They are never rescaled or rewritten. Invalid hours are excluded instead of repaired:
  - the MW-unit window and conflicting duplicate timestamps
  - STOP, PARTIAL, and daylight STANDBY with POA > 300
  - missing status with low output, POA/power spikes, readings > 10,100 kW, and the frozen logger
- No tuning: the production configuration is re-used unchanged. All four models are scored on the same rows, and pooled
  metrics are computed over all prediction rows, not averaged over fold percentages.

### Result (pooled over the six folds, daylight hours, 3,696 rows; `model_summary.csv`)

| Model | MAE kW | MAE % of 10 MW | RMSE kW | Bias kW |
|---|---:|---:|---:|---:|
| B0 (climatology) | 476.19 | 4.762 | 791.62 | +179.57 |
| B2b | 287.14 | 2.871 | 430.17 | +88.64 |
| M2 | 281.62 | 2.816 | 422.42 | +48.90 |
| **BL50 (final)** | **274.43** | **2.744** | 412.11 | +68.77 |

- **Final reported metric: BL50 pooled daylight MAE 274.43 kW = 2.744 %.** Including night hours (all predicted 0)
  it would be 139.24 kW / 1.392 %. That is why daylight MAE is the main metric.
- BL50 per fold (daylight MAE %, `fold_results.csv`): F1 4.310, F2 3.511, F3 3.202, F4 1.593, F5 1.409, F6 3.150.
- Lowest per fold (among B2b/M2/BL50): BL50 in F1 and F6, M2 in F2, F4 and F5, B2b in F3. BL50 is never the worst of the
  three. Its pooled gain over M2 alone is small (≈ 7 kW).
- Worst validation day by BL50 daylight MAE: 2025-06-23 (F6). All three models over-predicted because the forecast-implied
  irradiance was well above the measured irradiance. This is shown in `notebooks/walkthrough.ipynb`, section 10.

### Test period untouched
The experiment reads only `data/train.csv`, and every validation window ends by 2025-06-30 23:00. The test period
**2025-07-01 through 2025-08-31** was not used for any fitting, selection or scoring. The experiment does not modify
`predictions.csv`, `run.py`, `src/` or `reports/`.

### Leakage-safe experiment vs the frozen production pipeline
**Shared by both:**
- geometry and radiative helpers, `fit_linke` / `fit_cloud_curve` from `src.solar`, and `fitted_dry` turbidity
- the `src.features.FEATURES` list, LightGBM parameters, 15 leaves, 295 rounds and seeds 42–46
- the ratio target with a 200 kW physics floor, the 50/50 blend, and clipping to [0, 10,000] kW

**Different:**

| | Production (`run.py`) | Leakage-safe experiment |
|---|---|---|
| Cleaning | `src/clean.py` rules 1–11 on the full training file | fold-local, row-local rules on each training slice |
| Training target | soiling-corrected `ac_power_clean` | cleaned `ac_power_kw`, no soiling correction |
| Forecast gap fill | same-day interpolation, then adjacent-day same hour, then nearest in day | same-day interpolation only |
| Fit data | all training data (2024-01 .. 2025-06) | each fold's training slice |
| Night rule | 0 when elevation < 0 | 0 when elevation ≤ 0 |

For the monsoon test months the soiling factor is ≈ 1 (see `## Cleaning`), so the target difference should matter little
there. It has not been measured separately.

### Limitations
- **Re-measurement, not a new model.** BL50, 15 leaves and 295 rounds (the median of early-stopping iterations on the
  original folds) were all chosen on the superseded validation. The six folds re-measure that frozen setup without
  tuning. They are not a fully untouched hold-out for model selection.
- **Season mix.** The pooled 2.744 % includes the easier dry-season folds (F4 and F5 are below 1.6 %). The monsoon folds
  are higher (F1 Jul–Aug 4.31 %, F2 Sep–Oct 3.51 %), so Jul–Aug 2025 should be expected to be worse than the pooled figure.
- The earlier expected-test estimate (≈ 4.2 %, `## Expected test performance`) comes from the superseded folds and was not re-derived.
