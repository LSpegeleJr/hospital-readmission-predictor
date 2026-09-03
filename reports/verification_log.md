# Data Verification Log

This file documents every empirical check performed against the real dataset during
development — not assumptions or general knowledge, but actual results confirmed by
running code against `data/diabetic_data.csv`. Each entry includes what was checked,
why, and the actual result.

**All checks below are reproducible by running `src/verify_pipeline.py`** — this is
the single script that regenerates every number in this log. Re-run it any time the
data or pipeline logic changes to confirm these results still hold.

---

## 1. CMS HRRP condition coverage (all ages)

**Why:** verify whether the 6 CMS HRRP-tracked conditions are identifiable in this
dataset, and to what extent, before deciding on dataset scope.

**Method:** `src/check_hrrp_conditions.py` — matched ICD-9 prefixes against `diag_1`,
`diag_2`, `diag_3` for each of the 4 diagnosis-based conditions.

**Result:**

| Condition | Encounters | % of dataset |
|---|---|---|
| AMI (heart attack) | 4,356 | 4.3% |
| Heart Failure | 17,464 | 17.2% |
| Pneumonia | 6,147 | 6.0% |
| COPD | 10,984 | 10.8% |
| CABG | Not identifiable — procedure code, not diagnosis | n/a |
| Hip/Knee Replacement | Not identifiable — procedure code, not diagnosis | n/a |

**Conclusion:** 4 of 6 HRRP conditions are identifiable as comorbidities in this
diabetes-based dataset. CABG and Hip/Knee Replacement cannot be checked — confirmed by
inspecting the full column list of `diabetic_data.csv` directly (no procedure-code
field exists, only `num_procedures`, a count).

---

## 2. Medicare-age (60+) proxy subgroup

**Why:** CMS HRRP applies to Medicare beneficiaries (age 65+ eligibility). This
dataset only records age in 10-year brackets, so an exact 65+ cutoff isn't possible.
Decision made to use `[60-70)` and above as a "Medicare-age proxy," accepting this
includes some patients aged 60-64.

**Method:** `src/check_hrrp_conditions.py`, extended with an age-bracket filter.

**Result:**

| Condition | All ages | Medicare-age proxy (60+) |
|---|---|---|
| AMI | 4,356 | 3,261 |
| Heart Failure | 17,464 | 14,476 |
| Pneumonia | 6,147 | 4,854 |
| COPD | 10,984 | 8,893 |

---

## 3. Patient encounter distribution (data leakage risk check)

**Why:** determine whether the same patient can appear in multiple rows/encounters,
which would create a data leakage risk if rows were randomly split into train/test
without accounting for patient identity.

**Method:** `.nunique()` and `.value_counts()` on `patient_nbr`, run against the raw
CSV (before any cleaning).

**Result:**

- Total encounters (rows): 101,766
- Unique patients: 71,518
- Encounters per patient (average): 1.42
- Patients with more than 1 encounter: 16,773 (~23% of all unique patients)
- Max encounters for a single patient: 40

**Conclusion:** confirmed real and non-trivial leakage risk. `patient_nbr` was removed
from `DROP_COLS` (previously planned for removal) and `GroupShuffleSplit` was used
instead of plain `train_test_split`, ensuring all of one patient's encounters stay
within the same split (train or test), never divided across both.

---

## 4. Train/test split integrity check (post-`GroupShuffleSplit`)

**Why:** confirm the `GroupShuffleSplit` fix actually eliminates patient-level leakage
between train and test sets, rather than assuming it works.

**Method:** after running `train_test_split_df()`, compared the sets of `patient_nbr`
values present in `X_train` vs. `X_test`.

**Result:**

- Patients in train: 56,351
- Patients in test: 14,088
- Overlapping patients: 0 ✅
- Total unique patients (post-cleaning): 70,439
- Train + Test patients: 70,439 ✅ (fully reconciled, no missing patients)

**Note:** the drop from 71,518 unique patients (raw data) to 70,439 (post-cleaning) is
expected — `load_and_clean()` removes encounters where the patient died or was
discharged to hospice, and ~1,079 patients' only encounter(s) in the raw data were
exclusively of this type, removing them from the dataset entirely once cleaned.

---

## 5. Post-cleaning consistency check

**Why:** sanity-check that row count and unique patient count after `load_and_clean()`
are internally consistent with each other, cross-validating against numbers already
established independently in Check #4.

**Method:** `check_cleaning_consistency()` in `src/verify_pipeline.py` — reports row
count and unique patient count directly from the cleaned DataFrame.

**Result:**

- Rows after cleaning: 100,114
- Unique patients after cleaning: 70,439

**Conclusion:** matches exactly with numbers independently reported in Check #4
(70,439 unique patients, post-cleaning) — confirms consistency across two separately-
computed checks, with no discrepancy.

---

## 6. HRRP feature column verification (`add_hrrp_features()`)

**Why:** confirm the engineered feature columns (`has_ami`, `has_heart_failure`,
`has_pneumonia`, `has_copd`, `is_60_and_over`) produced by `add_hrrp_features()`
compute correctly, by comparing their counts against the independently-verified
numbers from Check #1 and Check #2 (which compute these same conditions a different
way, directly from `HCP` and `MEDICARE_AGE_BRACKETS`, without using
`add_hrrp_features()`).

**Method:** `check_hrrp_feature_columns()` in `src/verify_pipeline.py` — sums each
new boolean column on the cleaned + feature-enriched DataFrame
(`add_hrrp_features(load_and_clean(...))`).

**Result:**

| Column | Check #1/#2 (raw data) | Check #6 (cleaned + featured) | Difference |
|---|---|---|---|
| has_ami | 4,356 | 4,167 | -189 |
| has_heart_failure | 17,464 | 17,095 | -369 |
| has_pneumonia | 6,147 | 5,919 | -228 |
| has_copd | 10,984 | 10,792 | -192 |
| is_60_and_over | 68,541 | 67,121 | -1,420 |

**Conclusion:** every column dropped somewhat after cleaning, which is expected, not
a bug — `load_and_clean()` removes encounters where the patient died or was
discharged to hospice, and both serious HRRP-tracked conditions and older age are
plausibly over-represented among those removed rows.

This was checked quantitatively rather than just assumed: the overall row count
dropped by 1,652 / 101,766 ≈ 1.6% during cleaning, while AMI's count dropped by
189 / 4,356 ≈ 4.3% — a meaningfully larger drop rate than the dataset overall,
supporting the hypothesis that AMI patients are disproportionately represented among
the death/hospice rows removed. The same logic applies to `is_60_and_over`: older
patients (60+) carry higher baseline mortality risk, so it's expected that this
subgroup would shrink by a larger proportion (1,420 / 68,541 ≈ 2.1%) than the overall
dataset's row-count shrinkage (≈1.6%) once death/hospice encounters are removed.

All five columns' post-cleaning counts are internally consistent with this
explanation, with no unexplained discrepancies — confirming `add_hrrp_features()`
is computing each flag correctly against the same logic already verified in
Checks #1 and #2.

---

## 7. Cross-validation fold integrity (`StratifiedGroupKFold`)

**Why:** confirm that `get_cv_folds()`'s use of `StratifiedGroupKFold` actually
delivers on both guarantees it was chosen for — (1) no patient split across a
fold's train/validation portions (the "Group" guarantee), and (2) stable class
balance across all folds (the "Stratified" guarantee) — rather than just trusting
the tool's documentation.

**Method:** `check_cv_fold_integrity()` in `src/verify_pipeline.py` — for each of
the 5 folds produced by `get_cv_folds(X_train, y_train)`, checks patient overlap
between that fold's train/validation portions (same set-intersection technique as
Check #4), and reports the validation portion's readmission rate.

**Result:**

| Fold | Train rows | Val rows | Patient overlap | Val readmission rate |
|---|---|---|---|---|
| 1 | 64,033 | 16,006 | 0 | 0.113 |
| 2 | 64,027 | 16,012 | 0 | 0.113 |
| 3 | 64,031 | 16,008 | 0 | 0.113 |
| 4 | 64,033 | 16,006 | 0 | 0.113 |
| 5 | 64,032 | 16,007 | 0 | 0.113 |

Overall training set readmission rate: 0.113 (identical to every fold's validation
rate).

**Conclusion:** both guarantees hold perfectly across all 5 folds — zero patient
leakage in any fold, and essentially no drift in class balance (every fold's
validation readmission rate matches the overall rate to 3 decimal places). This
confirms `get_cv_folds()` is safe to use for the planned 3-way model comparison
(Logistic Regression, Random Forest, XGBoost), and that any performance differences
observed between models won't be an artifact of uneven data splits.

---

## 8. Model feature exclusion check (`get_model_features()`)

**Why:** confirm `get_model_features()` drops exactly the intended set of columns —
`NON_FEATURE_COLS` (patient_nbr, race, gender — retained elsewhere for grouping/
fairness audit) plus `LOW_SIGNAL_COLS` (15 low-variance medication columns + 3 raw
diagnosis codes, superseded by `add_hrrp_features()`'s engineered flags) — and
nothing else, before this becomes the actual input to model training.

**Method:** `check_model_features()` in `src/verify_pipeline.py` — compares the
column set removed by `get_model_features(featured_df)` against the expected
union of `NON_FEATURE_COLS` and `LOW_SIGNAL_COLS`.

**Result:**

- Original columns: 52
- Model-ready columns: 31
- Expected to drop: 21
- Actually dropped: 21
- Match: True

**Conclusion:** `get_model_features()` is working exactly as designed — no
unexpected columns dropped, none missed. The 31 remaining columns (demographics
minus race/gender, admin codes, visit-intensity counts, the 4 HRRP comorbidity
flags, the age-60+ flag, the 8 retained medication columns, `change`/`diabetesMed`,
and `number_diagnoses`) are what `train.py` will actually use as model inputs.

**Clarifying note (added after building `train.py`):** this check ran
`get_model_features()` on `featured_df` directly — before it's split into `X`
(features) and `y` (target) — so the "31 model-ready columns" figure above
still includes the target column, `readmitted_30d`, since `get_model_features()`
was never designed to remove it (that's `train_test_split_df()`'s job, via
`X = df.drop(columns=[target])`). In `train.py`, the target is already removed
*before* `get_model_features()` runs, so the real, final predictor count used
for modeling is **30**, not 31. Both numbers are correct — they're just
answering slightly different questions (with vs. without the target column
still present).

---

## Template for future entries

```markdown
## N. [Check name]

**Why:** [what question this check answers, and why it matters]

**Method:** [script/function used, what logic was applied]

**Result:** [actual numbers/output]

**Conclusion:** [what this means for the project, any decisions made as a result]
```
