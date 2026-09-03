# Hospital Readmission Risk Predictor

Predicts 30-day hospital readmission risk from patient discharge records, with an
explainability layer so a clinician can see *why* a patient is flagged high-risk.

## Why this matters
- CMS penalizes hospitals financially for excess 30-day readmissions (Hospital
  Readmissions Reduction Program).
- Average readmission costs ~$15K; early intervention (follow-up calls, med
  reconciliation, transport to appointments) is far cheaper.
- A model that just says "high risk" isn't useful to a clinician — it needs to say
  *why*, so this project treats explainability (SHAP) as a first-class deliverable,
  not an afterthought.

## Project structure
```
readmission-risk-predictor/
├── data/           # raw + processed data (gitignored - see data/README.md)
├── notebooks/       # exploratory analysis, model dev
├── src/             # reusable pipeline code (data prep, training, inference)
├── app/              # Streamlit clinician-facing app
└── reports/         # model card, evaluation report
```

## Why UCI diabetes data instead of MIMIC-IV or HCUP NRD

Three data sources were seriously considered for this project:

- **MIMIC-IV** (PhysioNet) — richer clinical data, ICU-level detail, and free-text
  discharge notes, but requires credentialed access (identity verification, a CITI
  human-subjects training course, and a signed Data Use Agreement), and even after
  approval, the DUA prohibits redistributing the data or publishing individual
  records/notes. A live public demo or open GitHub repo using real MIMIC-IV data would
  not be permitted.
- **HCUP Nationwide Readmissions Database (NRD)** — all-payer claims data that could
  isolate all 6 HRRP conditions directly, but requires purchasing the data, completing
  a Data Use Agreement Training Course, and carries the same redistribution
  restrictions as MIMIC (no public posting of raw data, no publicly shareable demo
  using real records).
- **UCI Diabetes 130-US Hospitals dataset** — public domain, no credentialing, no cost,
  no redistribution restrictions.

Since a core goal of this project is to be **fully shareable with prospective
employers** — a public repo, a live demo, and real data an interviewer could
independently inspect — the UCI dataset was chosen specifically for its
unrestricted shareability, even though it required verifying(and accepting) a
narrower scope: only 4 of the 6 CMS HRRP conditions can be identified in this data
as diabetes comorbidities (see `data/README.md` for the full breakdown), and the
dataset's base population is diabetic encounters rather than a general hospital
population.

This tradeoff — narrower condition coverage in exchange for zero legal or
distribution friction — was a deliberate design decision, not a default.

## Data sources
1. **UCI Diabetes 130-US Hospitals (1999-2008)** — primary dataset, ~100K encounters,
   50 features, labeled readmission outcome. Start here.
   https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

     > **Scope note:** this dataset's base population is diabetic encounters. Of the 6
   > CMS HRRP-tracked conditions, 4 diagnosis-based conditions (AMI, Heart Failure,
   > Pneumonia, COPD) are identifiable as comorbidities in this data; CABG and Hip/Knee
   > Replacement are procedures and cannot be identified here. See `data/README.md` for
   > full verification details.
   
2. **MIMIC-IV** (PhysioNet, free credentialed access) — richer ICU data + free-text
   clinical notes, good for the NLP/LLM stretch goal.
   https://physionet.org/content/mimiciv/
3. **CMS Hospital Readmissions Reduction Program data** — hospital-level readmission
   rates for benchmarking your model's predictions against real-world penalty data.
   https://data.cms.gov/provider-data/topics/hospitals

## Roadmap
- [ ] Phase 1: Data ingestion + cleaning (`src/data_prep.py`)
- [ ] Phase 2: Baseline model — logistic regression (`notebooks/01_baseline.ipynb`)
- [ ] Phase 3: Gradient boosting model (XGBoost) + comparison
- [ ] Phase 4: SHAP explainability layer
- [ ] Phase 5: Streamlit app (`app/app.py`)
- [ ] Phase 6 (stretch): LLM layer that reads discharge notes (MIMIC-IV) and generates
      plain-language risk explanations / care plans

## Setup
```bash
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Model evaluation approach
Because readmission is a minority class (~11% for <30 days), accuracy alone is
misleading. Track: ROC-AUC, PR-AUC, recall at a fixed precision threshold clinicians
can act on, and calibration (a "70% risk" prediction should mean ~70% actually
readmit).

## Deploying the live demo (Streamlit Community Cloud)
The app (`app/app.py`) loads the trained model from `reports/final_model.joblib`
using a path relative to the repo root, so it will run as-is once the repo is on
GitHub — no code changes needed.

1. **Push this repo to GitHub** (public, so Streamlit Community Cloud and
   interviewers can both reach it):
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   ```
   Create a new empty repo at https://github.com/new (don't initialize it with a
   README), then:
   ```bash
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git branch -M main
   git push -u origin main
   ```
2. **Deploy on Streamlit Community Cloud**:
   - Go to https://share.streamlit.io and sign in with GitHub.
   - Click "New app" → select this repo and the `main` branch.
   - Set **Main file path** to `app/app.py`.
   - Click **Deploy**. The first build takes a few minutes (installing
     `requirements.txt`).
3. **If the deploy fails on the model file**: the app was trained locally with
   whatever `scikit-learn`/`xgboost` versions were installed in `venv/` at the
   time. `requirements.txt` doesn't pin versions, so Streamlit Cloud installs the
   latest ones, which can occasionally be incompatible with a `joblib`-pickled
   model. If that happens, run this locally to see what trained the model:
   ```bash
   venv\Scripts\activate
   pip freeze | findstr "scikit-learn xgboost shap joblib numpy pandas"
   ```
   and pin those exact versions (`package==x.y.z`) in `requirements.txt`, then
   push again.
4. Before the first push, skim what `git add .` picked up (the `.gitignore` excludes
   `venv/`, `data/*.csv`, `data/*.zip`, and the large
   `reports/train_test_split_visualization.xlsx`) — decide whether the working
   Office docs (`Basis Document.docx`, `Results.docx`, the `.pptx` files) belong in
   a public repo or should stay local.
