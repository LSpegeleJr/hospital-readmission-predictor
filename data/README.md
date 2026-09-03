# Data

Not checked into git (too large / licensing). Download manually:

1. Go to https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
2. Download the zip, extract `diabetic_data.csv` and `IDs_mapping.csv` into this folder.

Expected files:
- `diabetic_data.csv` — main dataset (~101,766 rows x 50 columns)
- `IDs_mapping.csv` — decodes categorical IDs (admission_type_id, discharge_disposition_id, etc.)

Target column: `readmitted` with values `<30`, `>30`, `NO`.
For this project we binarize: `1` if `<30`, else `0`.

## CMS HRRP condition coverage

CMS's Hospital Readmissions Reduction Program (HRRP) officially tracks 6 conditions:
AMI (heart attack), Heart Failure, Pneumonia, COPD, CABG (bypass surgery), and
Hip/Knee Replacement (THA/TKA).

This dataset's base population is diabetic encounters, not HRRP-condition encounters —
diabetes was required as at least one diagnosis for a row to be included at all. However,
patients can carry additional diagnoses in `diag_1`, `diag_2`, and `diag_3`, which allows
us to identify comorbid HRRP conditions where present.

**Verified coverage (checked against actual ICD-9 prefixes in this file):**

| Condition | Identifiable? | ICD-9 prefix(es) used | Encounters found | % of dataset |
|---|---|---|---|---|
| AMI | Yes (diagnosis) | 410 | 4,356 | 4.3% |
| Heart Failure | Yes (diagnosis) | 428 | 17,464 | 17.2% |
| Pneumonia | Yes (diagnosis) | 480–486 | 6,147 | 6.0% |
| COPD | Yes (diagnosis) | 490–492, 496 | 10,984 | 10.8% |
| CABG | **No** — procedure, not diagnosis | n/a | n/a | n/a |
| Hip/Knee Replacement | **No** — procedure, not diagnosis | n/a | n/a | n/a |

## Diagnosis codes

**Code source:** ICD-9-CM prefixes verified against icd9data.com and standard clinical
comorbidity groupings (e.g., Quan et al. Elixhauser/Charlson comorbidity ICD-9 code
lists). Note COPD grouping (490-492, 496) follows a common comorbidity-index convention;
slightly different code sets appear across studies since 493 (asthma) and edge cases
around 490 are sometimes handled differently.

## Medicare-age (60+) subgroup

CMS's HRRP applies specifically to Medicare fee-for-service beneficiaries, who are
generally eligible starting at age 65. This dataset only records age in 10-year
brackets (`[60-70)`, `[70-80)`, etc.), so an exact age-65 cutoff isn't possible.

**Decision:** we treat `[60-70)` and above as a "Medicare-age proxy" population,
acknowledging this knowingly includes some patients aged 60-64 who would not yet
be Medicare-eligible. This is a deliberate simplification made necessary by the
data's bracketed age format, not an attempt at precision we don't have.

**HRRP condition counts, Medicare-age proxy (60+) vs. all ages:**

| Condition | All ages | Medicare-age proxy (60+) |
|---|---|---|
| AMI | 4,356 | 3,261 |
| Heart Failure | 17,464 | 14,476 |
| Pneumonia | 6,147 | 4,854 |
| COPD | 10,984 | 8,893 |

**Why CABG and Hip/Knee Replacement can't be checked:** these are surgical procedures,
not diagnoses, and are coded in a separate system (procedure codes) from diagnosis codes
(ICD-9-CM). This dataset only contains a procedure *count* (`num_procedures`), not
procedure *codes* — so there is no field that could identify which specific procedures,
if any, a patient underwent. Confirmed directly against this dataset's actual column list
(no `proc_1`, `cpt_code`, or similar field exists).

**Implication for modeling:** this project treats diabetes as the base population and
uses the 4 identifiable HRRP conditions as comorbidity flags/features (e.g.
`has_heart_failure`), rather than claiming to model the 6 HRRP conditions directly.
