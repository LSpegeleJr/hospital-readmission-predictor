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

## Template for future entries

```markdown
## N. [Check name]

**Why:** [what question this check answers, and why it matters]

**Method:** [script/function used, what logic was applied]

**Result:** [actual numbers/output]

**Conclusion:** [what this means for the project, any decisions made as a result]
```
