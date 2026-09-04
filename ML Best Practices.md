# ML Project Template — Phases, Philosophy, and Reusable Patterns

This document generalizes the methodology built in the Hospital Readmission Risk
Predictor project into a reusable template for future tabular ML problems. Each
phase covers: **Philosophy** (why this phase matters), **Generalized code skeleton**
(adapt directly), **Key Python/ML concepts** (the transferable skills), and
**Follow-up questions** (what to check before moving on).

---

## Phase 1: Dataset Selection & Scoping

**Philosophy:**
Before writing any code, deliberately choose your data source with real tradeoffs in
mind — not just "whatever's easiest to download." Document *why* you chose it,
including what you gave up by not choosing alternatives (legal/privacy constraints,
shareability, richness of features, cost). This becomes your project's origin story
and protects you from being caught flat-footed when someone asks "why this dataset?"

**No code skeleton for this phase** — it's a research and documentation exercise.
Write your reasoning directly into your project's README.

**Key concepts:**
- Data use agreements / licensing — public domain vs. credentialed access vs. paid
- Shareability as a first-class project requirement, not an afterthought, if the
  project is meant to be portfolio/public-facing
- De-identification vs. re-identification risk (why "no names in the data" isn't
  the same as "safe to publish")

**Follow-up questions:**
- Can I legally and practically share this data/demo with the audience I'm building
  this for?
- What did I give up by choosing this dataset over the alternatives (real-world
  applicability, richness, condition coverage)?
- Have I written this reasoning down somewhere permanent, not just decided it in my
  head?

---

## Phase 2: Check Conditions — Establish Data Coverage Before Committing

**Philosophy:**
Verify *empirically* how well your dataset actually covers the population/conditions
you care about, before building features around them. Never assume coverage from a
dataset's general description alone.

**Generalized code skeleton:**

```python
import pandas as pd

CONDITIONS_OF_INTEREST = {
    "Category A": ["prefix1", "prefix2"],
    "Category B": ["prefix3"],
}

def matches_any(value, criteria):
    if pd.isna(value):
        return False
    value = str(value)
    return any(value.startswith(c) for c in criteria)

def row_matches(row, criteria, columns_to_check):
    return any(matches_any(row[col], criteria) for col in columns_to_check)

if __name__ == "__main__":
    df = pd.read_csv("your_data.csv")
    print(f"Total rows: {len(df)}")
    for category_name, criteria in CONDITIONS_OF_INTEREST.items():
        matches = df.apply(lambda row: row_matches(row, criteria, ["col1", "col2", "col3"]), axis=1)
        print(f"{category_name}: {matches.sum()} rows ({matches.mean()*100:.1f}%)")
```

**Key concepts:**
- Dictionaries as lookup tables for category → criteria mappings
- Defensive functions (`pd.isna` checks) for real-world messy data
- String methods (`.startswith()`) for prefix/code matching
- `any()` + generator expressions for compact multi-criteria checks
- `df.apply(..., axis=1)` + `lambda` for row-wise, multi-column logic
- Boolean Series + `.sum()`/`.mean()` for counting/percentage tricks

**Follow-up questions:**
- Does the dataset's structure let me check every category I care about, or are some
  simply absent (e.g., a code system the dataset doesn't capture at all)?
- Should this be combined with a demographic/subgroup filter?
- Are the coverage numbers themselves worth documenting permanently (a verification
  log), so future-you or a reviewer can see the real numbers, not just a claim?

---

## Phase 3: Data Cleaning

**Philosophy:**
Separate *cleaning* (fixing/removing untrustworthy data) from every later step.
Cleaning should be the one place in your pipeline responsible for: fixing a
dataset's specific missing-value conventions, removing columns that are pure
identifiers or too corrupted to trust, and removing rows that don't belong in your
target population at all. Keep this function narrowly scoped — it should not decide
which *features* are useful, only which data is *trustworthy*.

**Generalized code skeleton:**

```python
import pandas as pd
import numpy as np

DROP_COLS = ["id_column", "mostly_missing_column"]

def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Fix dataset-specific missing-value markers
    df = df.replace("?", np.nan)  # adjust to match your dataset's convention

    # Defensive column drop — never crashes on a missing/typo'd name
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # Binarize / clean the target variable
    df["target_binary"] = (df["raw_target"] == "positive_value").astype(int)
    df = df.drop(columns=["raw_target"])

    # Remove rows that don't belong in your population of interest
    if "some_exclusion_flag" in df.columns:
        df = df[~df["some_exclusion_flag"].isin([/* excluded codes */])]

    # Fill remaining missing categoricals with an explicit placeholder
    cat_cols = df.select_dtypes(include=["object", "str"]).columns
    for c in cat_cols:
        df[c] = df[c].fillna("Unknown")

    return df.reset_index(drop=True)
```

**Key concepts:**
- List comprehensions for safe, defensive column operations
- `.replace()` for normalizing dataset-specific missing-value conventions to `NaN`
- Boolean masking + `~` (negation) for row filtering
- `select_dtypes()` for type-based column selection (and the `"object"` vs. `"str"`
  forward-compatibility issue in newer pandas versions)
- `.reset_index(drop=True)` for cleaning up row numbering after filtering
- Type hints (`-> pd.DataFrame`) and docstrings for documenting function contracts

**Follow-up questions:**
- Have I verified my missingness/drop-reason claims against the real data, rather
  than assuming?
- Am I dropping anything I'll actually need later (e.g., an ID column needed for
  grouping during train/test splitting)? If so, defer dropping it — see Phase 6.
- Does my target binarization correctly handle every possible raw value, including
  edge cases?

---

## Phase 4: Feature Engineering

**Philosophy:**
Keep feature engineering as its own step, separate from cleaning — cleaning is
subtractive/corrective, feature engineering is additive. This function should never
remove existing rows/columns, only add new, derived signal. Keeping this boundary
clean means you can add, remove, or modify engineered features without ever risking
the integrity of your cleaning logic.

**Generalized code skeleton:**

```python
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()  # never mutate the caller's DataFrame

    for category_name, criteria in CONDITIONS_OF_INTEREST.items():
        column_name = CATEGORY_COLUMN_NAMES[category_name]  # explicit mapping, not string-parsing
        df[column_name] = df.apply(lambda row: row_matches(row, criteria, [...]), axis=1)

    df["some_derived_flag"] = df["some_column"].isin([...])

    return df
```

**Key concepts:**
- `.copy()` to avoid unintended side effects on the caller's data
- Explicit name-mapping dictionaries (not auto-generated strings) for reliability —
  a lesson learned the hard way when auto-generated column names silently truncated
- `.isin()` for membership-based binary flags
- The difference between `axis=1` (row-wise, needed when a function must see
  multiple columns at once) and operations that work on a single already-isolated
  column (no `axis` needed)

**Follow-up questions:**
- Does every new feature have a clear, defensible real-world rationale, not just
  "because I could compute it"?
- Have I verified the new columns' counts against independently-known numbers
  (e.g., recomputing a coverage check from Phase 2 using the new columns)?

---

## Phase 5: Full Column Review — Keep, Drop, or Transform

**Philosophy:**
Go through *every single column* deliberately — group by theme (demographics,
administrative, clinical, etc.), and make an explicit, documented call for each one.
This is where ethical/fairness considerations, low-signal features, and
high-cardinality/redundant features all get identified — before any modeling code
is written, not as an afterthought.

**No single code skeleton** — this phase is a structured review process. Useful
sub-checks:

```python
# Missingness check (informs "mostly missing" drop decisions)
missing_pct = (df[col] == "?").mean() * 100

# Variance check (informs low-signal feature drop decisions)
value_counts = df[col].value_counts()
dominant_value_pct = value_counts.iloc[0] / len(df) * 100
# if dominant_value_pct > ~95-99%, the column carries very little signal

# Cardinality check (informs "too sparse to one-hot encode directly" decisions)
n_unique = df[col].nunique()
```

**Key concepts:**
- `.value_counts()` for understanding a categorical column's real distribution
- Recognizing when a numeric-looking dtype is secretly categorical (coded IDs)
- Separating "this column is bad data" (Phase 3, `DROP_COLS`) from "this column is
  fine but not useful for modeling" (a separate, later exclusion list — Phase 7)
- Sensitive-attribute handling: exclude from modeling, but consider retaining for a
  post-hoc fairness audit rather than deleting entirely (see Phase 11)

**Follow-up questions:**
- For every column I'm dropping, do I have a *verified* reason (a real percentage,
  a real variance check), not just an assumption?
- Have I checked whether any "quantity-looking" column is actually a coded category
  in disguise?
- Are there sensitive attributes (protected characteristics) that need special
  handling — excluded from the model, but retained for later fairness evaluation?

---

## Phase 6: Train/Test Split with Entity-Level Grouping

**Philosophy:**
If your data can contain multiple rows per real-world entity (patient, customer,
user), a naive random split risks leaking that entity's information across both
train and test — inflating your apparent performance. Always check for this risk
empirically, and use a group-aware splitter if it's real.

**Generalized code skeleton:**

```python
def check_entity_leakage_risk(df, entity_col):
    total_rows = len(df)
    unique_entities = df[entity_col].nunique()
    entity_counts = df[entity_col].value_counts()
    repeat_entities = (entity_counts > 1).sum()
    print(f"Total rows: {total_rows}, unique entities: {unique_entities}")
    print(f"Entities with >1 row: {repeat_entities} ({repeat_entities/unique_entities*100:.1f}%)")

def train_test_split_df(df, target, entity_col, test_size=0.2, seed=42):
    from sklearn.model_selection import GroupShuffleSplit
    X = df.drop(columns=[target])
    y = df[target]
    groups = df[entity_col]
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
```

**Key concepts:**
- `GroupShuffleSplit` — guarantees all of one entity's rows land in the same split
- Generators (`.split()` returns one) — must be consumed fresh, can't be reused
- `.iloc[]` for position-based row selection after getting index arrays back
- Verifying the fix worked: `set()` intersection between train/test entity IDs,
  confirming zero overlap

**Follow-up questions:**
- Have I actually checked (not assumed) whether entities repeat in this dataset?
- Have I verified, with real numbers, that the split produces zero entity overlap?
- Do my row counts and entity counts reconcile before vs. after any cleaning step
  that removes rows (a cleaning step can silently also remove entities entirely)?

---

## Phase 7: Cross-Validation with Grouping + Stratification

**Philosophy:**
A single train/test split risks judging model performance off one lucky/unlucky
split. Cross-validation addresses this — but if your data has both entity-grouping
risk (Phase 6) AND class imbalance, use a splitter that respects both
simultaneously.

**Generalized code skeleton:**

```python
def get_cv_folds(X, y, entity_col, n_splits=5, seed=42):
    from sklearn.model_selection import StratifiedGroupKFold
    groups = X[entity_col]
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return splitter.split(X, y, groups=groups)
```

**Key concepts:**
- `StratifiedGroupKFold` — combines group-safety (Phase 6's concern) with stable
  class balance across folds
- Why this must be called fresh, inside any loop that uses it more than once
  (generator exhaustion)
- The two-stage split structure: a final untouched holdout test set (Phase 6) is
  carved off FIRST; cross-validation (this phase) only ever touches the remaining
  development data — never mix these roles

**Follow-up questions:**
- Have I verified zero entity overlap AND stable class balance across every fold,
  with real printed numbers?
- Is my holdout test set genuinely untouched by every step in this phase?

---

## Phase 8: Feature Selection for Modeling

**Philosophy:**
Keep a clear separation between "what stays in my working DataFrame" (needed for
grouping, later audits, etc.) and "what the model actually sees as input." A single
function should be the one source of truth for this final filtering step, applied
at the last possible moment before training.

**Generalized code skeleton:**

```python
# Retained in the DataFrame for grouping/auditing, excluded from modeling
NON_FEATURE_COLS = ["entity_id", "sensitive_attribute_1", "sensitive_attribute_2"]

# Excluded from modeling due to low signal / high sparsity / redundancy with
# engineered features — but still valid, still needed elsewhere in the pipeline
LOW_SIGNAL_COLS = ["raw_high_cardinality_col_1", "near_constant_col_1", ...]

def get_model_features(X: pd.DataFrame) -> pd.DataFrame:
    exclude = NON_FEATURE_COLS + LOW_SIGNAL_COLS
    return X.drop(columns=[c for c in exclude if c in X.columns])
```

**Key concepts:**
- List concatenation (`+`) for combining exclusion reasons while keeping them
  separately named/documented
- Why this differs from `DROP_COLS` (Phase 3) — bad data vs. merely
  unhelpful-as-a-feature data that other steps still depend on

**Follow-up questions:**
- Have I verified (with a real before/after column count check) that this function
  drops exactly what I expect, nothing more or less?
- Am I applying this at the right moment — late enough that grouping/splitting
  logic still has access to what it needs?

---

## Phase 9: Preprocessing Pipeline

**Philosophy:**
Models need fully numeric input. Build one reusable transformer that correctly
routes each column to the right treatment — but don't trust a column's stored
dtype blindly; some numeric-looking columns are secretly categorical codes.

**Generalized code skeleton:**

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

FORCE_CATEGORICAL_COLS = ["coded_id_column_1", "coded_id_column_2"]

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "str", "bool"]).columns.tolist()

    for col in FORCE_CATEGORICAL_COLS:
        if col in numeric_cols:
            numeric_cols.remove(col)
            categorical_cols.append(col)

    return ColumnTransformer([
        ("numeric", StandardScaler(), numeric_cols),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ])
```

**Key concepts:**
- `ColumnTransformer` for applying different transforms to different column subsets
- `StandardScaler` — what "mean 0, standard deviation 1" actually means, and why it
  only affects the model's internal math, never your reported/displayed values
- `OneHotEncoder(handle_unknown="ignore")` for robustness against unseen categories
- Recognizing coded categorical columns stored as integers, and forcing correct
  treatment rather than trusting automatic dtype detection

**Follow-up questions:**
- Have I checked every numeric-dtype column for whether it's secretly a coded
  category, rather than assuming dtype tells the whole story?
- Am I building this preprocessor fresh per fold/training run (never letting it see
  validation/test data during fitting)?

---

## Phase 10: Model Comparison

**Philosophy:**
Compare multiple model families deliberately (e.g., a linear model, a bagging
ensemble, a boosting ensemble) rather than picking one out of habit. Use identical
cross-validation folds across all models for a fair comparison, and change only one
variable at a time when tuning — this is what turns tuning into genuine, defensible
experiments rather than guesswork.

**Generalized code skeleton:**

```python
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score

MODELS = {
    "Model A": SomeLinearModel(...),
    "Model B": SomeBaggingModel(...),
    "Model C": SomeBoostingModel(...),
}

def run_cross_validation(X_train, y_train):
    results = []
    oof_predictions = {name: pd.Series(index=X_train.index, dtype=float) for name in MODELS}
    model_features = get_model_features(X_train)

    for model_name, model in MODELS.items():
        folds = get_cv_folds(X_train, y_train, entity_col="entity_id")  # fresh generator per model
        for fold_num, (fold_train_idx, fold_val_idx) in enumerate(folds, start=1):
            X_fold_train, X_fold_val = model_features.iloc[fold_train_idx], model_features.iloc[fold_val_idx]
            y_fold_train, y_fold_val = y_train.iloc[fold_train_idx], y_train.iloc[fold_val_idx]

            pipeline = Pipeline([
                ("preprocessor", build_preprocessor(X_fold_train)),
                ("classifier", model),
            ])
            pipeline.fit(X_fold_train, y_fold_train)

            proba = pipeline.predict_proba(X_fold_val)[:, 1]
            results.append({
                "model": model_name, "fold": fold_num,
                "roc_auc": roc_auc_score(y_fold_val, proba),
                "pr_auc": average_precision_score(y_fold_val, proba),
            })
            oof_predictions[model_name].iloc[fold_val_idx] = proba

    return results, oof_predictions
```

**Key concepts:**
- Dictionaries mapping display names to live, configured model objects
- `Pipeline` for bundling preprocessing + model into one fit/predict unit
- `predict_proba(...)[:, 1]` — 2D array indexing, row selector vs. column selector
- Out-of-fold predictions — pooling every fold's validation predictions into one
  full, honest prediction set covering the entire training data
- ROC-AUC vs. PR-AUC: ROC-AUC is a ranking metric (probability a random positive
  outranks a random negative — not an accuracy measure); PR-AUC is more sensitive to
  minority-class performance, with a random-guess baseline equal to the class
  prevalence, not a fixed 0.5
- One-variable-at-a-time tuning discipline, so each experiment's result is
  interpretable

**Follow-up questions:**
- Am I giving every model the exact same folds, so the comparison is fair?
- Have I checked whether any numeric-looking hyperparameter defaults (e.g.,
  unconstrained tree depth) are silently working against a specific model?
- Is each tuning change documented as its own experiment, with a clear before/after
  comparison?

---

## Phase 11: Fairness Audit (Post-Training)

**Philosophy:**
If sensitive attributes were deliberately excluded from modeling (Phase 5/8), the
job isn't done — check whether the model's *errors* are evenly distributed across
those subgroups anyway, since bias can enter through correlated proxy features even
when a sensitive attribute is never a direct input.

**Generalized code skeleton (sketch — to be built once a final model is selected):**

```python
def fairness_audit(y_true, y_pred_proba, sensitive_attr_series, threshold=0.5):
    import pandas as pd
    df = pd.DataFrame({
        "y_true": y_true, "proba": y_pred_proba, "group": sensitive_attr_series
    })
    df["y_pred"] = (df["proba"] >= threshold).astype(int)

    for group_value in df["group"].unique():
        subset = df[df["group"] == group_value]
        false_negative_rate = ((subset["y_true"] == 1) & (subset["y_pred"] == 0)).sum() / (subset["y_true"] == 1).sum()
        print(f"{group_value}: FNR = {false_negative_rate:.3f}, n = {len(subset)}")
```

**Key concepts:**
- False negative rate as a particularly important metric in clinical contexts
  (missing a genuinely high-risk case is often more costly than a false alarm)
- Subgroup analysis without using the sensitive attribute as a model input

**Follow-up questions:**
- Are error rates meaningfully different across subgroups, and if so, is that
  driven by a correlated proxy feature I can identify?
- Is subgroup sample size large enough to trust the comparison (a tiny subgroup can
  show noisy, unreliable metrics)?

---

## Phase 12: Verification & Experiment Logging Habits

**Philosophy:**
Two permanent, living documents pay for themselves many times over: one tracking
every empirical *data* verification check (with real numbers, not assumptions), and
one tracking every *model experiment* (settings changed, results, interpretation).
Together they turn "trust me, I checked" into a citable, reviewable record — and
they're genuinely persuasive portfolio artifacts.

**Generalized code skeleton — the consolidated verification script pattern:**

The actual project built this as `verify_pipeline.py`: one script, growing over
time, holding one small function per guarantee you want to prove — imported from
your other modules, never duplicating their logic. Each function follows the same
shape: print a header, compute something real, print the result.

```python
import pandas as pd
from data_prep import load_and_clean, add_engineered_features, train_test_split_df, get_cv_folds

def check_entity_leakage_risk(df, entity_col):
    """Why this exists: quantifies how often an entity repeats, to justify
    whether group-based splitting is actually necessary."""
    print("=== Entity leakage risk ===")
    total_rows = len(df)
    unique_entities = df[entity_col].nunique()
    repeat_entities = (df[entity_col].value_counts() > 1).sum()
    print(f"Total rows: {total_rows}, unique entities: {unique_entities}")
    print(f"Entities with >1 row: {repeat_entities}")
    print()

def check_split_integrity(df, entity_col, target, entity_col_name):
    """Why this exists: proves the group-based split actually eliminated
    entity overlap, rather than trusting the splitter's theory."""
    print("=== Train/test split integrity ===")
    X_train, X_test, y_train, y_test = train_test_split_df(df, target=target)
    overlap = set(X_train[entity_col]) & set(X_test[entity_col])
    print(f"Train entities: {X_train[entity_col].nunique()}")
    print(f"Test entities: {X_test[entity_col].nunique()}")
    print(f"Overlap (should be 0): {len(overlap)}")
    print()

def check_cv_fold_integrity(X_train, y_train, entity_col):
    """Why this exists: proves StratifiedGroupKFold's two guarantees
    (no leakage, stable class balance) hold in practice, per fold."""
    print("=== Cross-validation fold integrity ===")
    overall_rate = y_train.mean()
    for fold_num, (train_idx, val_idx) in enumerate(get_cv_folds(X_train, y_train, entity_col), start=1):
        train_entities = set(X_train.iloc[train_idx][entity_col])
        val_entities = set(X_train.iloc[val_idx][entity_col])
        overlap = train_entities & val_entities
        val_rate = y_train.iloc[val_idx].mean()
        print(f"Fold {fold_num}: overlap={len(overlap)}, val rate={val_rate:.3f} (overall={overall_rate:.3f})")
    print()

# Add one new check function like this every time a new pipeline guarantee
# needs proving — never delete old ones, since they document the project's history.

if __name__ == "__main__":
    raw_df = pd.read_csv("your_data.csv")
    cleaned_df = load_and_clean("your_data.csv")
    featured_df = add_engineered_features(cleaned_df)
    X_train, X_test, y_train, y_test = train_test_split_df(featured_df, target="target_binary")

    check_entity_leakage_risk(raw_df, entity_col="entity_id")
    check_split_integrity(featured_df, entity_col="entity_id", target="target_binary", entity_col_name="entity_id")
    check_cv_fold_integrity(X_train, y_train, entity_col="entity_id")
```

**Why this script pattern matters, beyond just "run some checks":**
- **Reproducibility** — anyone (including future you) can rerun the entire
  verification suite with one command, any time the data or pipeline changes,
  rather than manually re-deriving numbers
- **It imports, never duplicates** — every check calls the *real* pipeline functions
  (`load_and_clean`, `get_cv_folds`, etc.), so a check can never silently drift out
  of sync with the actual pipeline logic it's supposed to be validating
- **It grows incrementally** — each check function was added at the moment a new
  design decision needed proving (e.g., adding `check_cv_fold_integrity` right when
  `StratifiedGroupKFold` was introduced), building a natural, chronological record
  of the project's methodology
- **Numbered checks map directly to log entries** — each `check_*()` function's
  printed output becomes one dated entry in the markdown verification log (see
  format below), so the code and the documentation stay tightly linked

**Markdown log structure that works well for both data checks and model experiments**
(see this project's `verification_log.md` and `model_experiment_log.md` for full
examples):

```markdown
## N. [Check or experiment name]

**Why:** [what question this answers]

**Method:** [what code/function produced this]

**Result:** [actual numbers/output]

**Conclusion:** [what this means, what decision it informed]
```

**Follow-up questions:**
- Would a stranger reading only this log understand what was checked and why,
  without needing to read the underlying code first?
- Does every claim in my README/report trace back to an entry in one of these logs?
- Does my verification script import from my real pipeline modules, or does it
  quietly re-implement logic that could drift out of sync over time?

---

## Phase 13: Model Finalization & Deployment

**Philosophy:**
Once model comparison and tuning are done, "final" happens in two separate,
deliberate steps: first, freeze the winning configuration and touch the holdout
test set exactly once to get an honest final number (train on the full dev set,
evaluate on holdout, never re-run this to fish for a better number); second,
persist the fitted pipeline as a portable artifact so downstream consumers (an
app, an API, a colleague) can load it without ever needing to retrain, or even
have access to the training code, data, or environment.

**Generalized code skeleton:**

```python
# finalize_model.py -- run this exactly once, after all tuning in train.py is done.
# Importing from train.py re-runs only its fast data loading, never its
# expensive cross-validation comparison (that logic lives inside train.py's
# own `if __name__ == "__main__":` block, which only runs when train.py is
# executed directly, never when it's imported).

import joblib
from train import (
    X_train, X_test, y_train, y_test,
    train_final_model_and_evaluate, plot_roc_pr_curves,
)

def train_final_model_and_evaluate(X_train, X_test, y_train, y_test):
    X_train_features = get_model_features(X_train)
    X_test_features = get_model_features(X_test)

    pipeline = Pipeline([
        ("preprocessor", build_preprocessor(X_train_features)),
        ("classifier", FINAL_MODEL),
    ])
    pipeline.fit(X_train_features, y_train)

    proba = pipeline.predict_proba(X_test_features)[:, 1]
    print(f"ROC-AUC: {roc_auc_score(y_test, proba):.3f}")
    print(f"PR-AUC: {average_precision_score(y_test, proba):.3f}")

    # Persist the fully fitted pipeline (preprocessor + model together) as one
    # portable binary artifact -- reloadable anywhere without retraining.
    joblib.dump(pipeline, "reports/final_model.joblib")
    return pipeline, proba

if __name__ == "__main__":
    final_pipeline, test_proba = train_final_model_and_evaluate(X_train, X_test, y_train, y_test)
    plot_roc_pr_curves(y_test, {"Final Model": test_proba},
                        title_suffix=" — Final Holdout Test",
                        save_path="reports/final_holdout_curves.png")
```

**Key concepts:**
- The holdout test set is spent, not reusable — evaluating on it is a one-way
  action; running the finalize script more than once to "see if a tweak helps"
  quietly turns your honest final number into another tuning experiment
- `joblib.dump()` / `joblib.load()` for serializing a full scikit-learn `Pipeline`
  (preprocessing + model together) to a single binary file — the consuming code
  never needs the original training data, feature-engineering code, or
  environment, only the artifact and matching library versions
- Keeping the "train the final model" script separate from the "compare
  candidate models" script, importing shared setup rather than duplicating it,
  so there's exactly one place that defines what "final" means

**Follow-up questions:**
- Have I actually finished tuning before running this — am I treating the
  holdout result as genuinely final, not as feedback for another round of
  changes?
- Does the saved artifact include everything downstream code needs (the fitted
  preprocessor, not just the classifier)?
- Is the artifact's path/filename something the deployment code can find
  deterministically (e.g., a fixed `reports/` path checked into a known
  location)?

---

## Phase 14: Deployment & Interactive UI Design

**Philosophy:**
A trained model creates value only once someone who isn't you can use it.
Turning a pipeline into an interactive tool surfaces a different category of
problem than modeling did — state management, input validation, and a deploy
pipeline — and it rewards the same "verify empirically" instinct as the rest of
the project: test UI behavior programmatically before trusting a manual
click-through, and understand exactly what triggers a live update before
assuming one happened.

**Generalized code skeleton:**

```python
import streamlit as st
import joblib

MODEL_PATH = "reports/final_model.joblib"

# @st.cache_resource: run this function once per app process, then hand back
# the same object on every rerun -- without it, every widget interaction
# would reload and re-deserialize the model from disk.
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    pipeline = load_model()
except FileNotFoundError:
    st.warning(f"No trained model found at `{MODEL_PATH}`. Run the finalize script first.")
    st.stop()

# --- Blank-by-default inputs ---
# index=None + a placeholder starts a selectbox genuinely empty, instead of
# silently defaulting to option 0. `key=` binds the widget to
# st.session_state so its value survives reruns -- but once
# st.session_state[key] exists, Streamlit uses THAT over any index=/value=
# you pass on future reruns. This is why a reset button can't just re-render
# the widget with a different default -- it has to clear the session_state
# entry instead.
FORM_KEYS = ["field_a", "field_b", "field_c"]

choice = st.selectbox("Field A", ["Option 1", "Option 2"], index=None,
                       placeholder="Make a selection", key="field_a")

def reset_form():
    """on_click callbacks run BEFORE the script reruns, so popping session_state
    here is safe -- the widgets below then redraw with their original defaults."""
    for k in FORM_KEYS:
        st.session_state.pop(k, None)

st.button("Reset form", on_click=reset_form)

# --- Validate before predicting ---
if st.button("Predict", type="primary"):
    missing = [name for name, val in {"Field A": choice}.items() if val is None]
    if missing:
        st.warning("Please make a selection for: " + ", ".join(missing))
        st.stop()  # halts the script here without needing to reindent everything below into an else-branch
    # ... build input row, pipeline.predict_proba(), display result ...
```

```python
# Testing UI logic without a browser, using Streamlit's own test framework
from streamlit.testing.v1 import AppTest

at = AppTest.from_file("app.py", default_timeout=60).run()
at.selectbox(key="field_a").set_value("Option 1").run()
at.button(key="Reset form")[0].click().run()
assert at.session_state.get("field_a") is None  # confirms reset actually cleared it
```

**Key concepts:**
- `st.cache_resource` — caches an object (like a loaded model) across reruns
  rather than recomputing it on every widget interaction
- `st.session_state` + `key=` — the mechanism that gives each widget persistent
  state across reruns; critically, a keyed widget's stored session_state value
  overrides any `index=`/`value=` default once it exists, which is why resetting
  requires clearing the state, not re-specifying a default
- `on_click` callbacks — run before the script reruns, making them the right
  place for state mutations like a reset, rather than trying to branch on a
  button's return value
- `st.stop()` as a lightweight early-exit for validation failures, avoiding a
  large reindent of existing code into an `else:` block
- `streamlit.testing.v1.AppTest` — runs the app headlessly and lets you set
  widget values, click buttons, and assert on session_state/warnings/outputs, so
  a UI change can be verified the same way a data pipeline change is verified
  (Phase 12's instinct, applied to UI)
- Deployment as a side effect of version control: a platform like Streamlit
  Community Cloud watches a GitHub branch and redeploys automatically on every
  push — meaning "did my change ship" reduces to "is it committed AND pushed,"
  not just saved locally

**Follow-up questions:**
- Have I tested the new UI behavior with something like `AppTest`, rather than
  only clicking through it once by hand?
- If a change isn't showing up after deploying, have I confirmed it's actually
  committed (not just staged) and pushed (not just committed) to the branch the
  deployment platform watches?
- Does the file path the deployment platform is configured to run match the
  file I'm actually editing (a stray duplicate file or wrong path is a common,
  confusing cause of "my push didn't work")?
- Does the app fail gracefully (a clear warning, not a raw traceback) if its
  model artifact is missing or hasn't been generated yet?

---

## Meta-lesson: the recurring judgment calls across every phase

A few decision patterns showed up repeatedly across this entire project, worth
internalizing as general instincts for any future tabular ML work:

1. **Verify empirically, every time** — never accept "this column is mostly
   missing" or "this dataset covers X" without actually running the check against
   real data.
2. **Separate concerns into narrowly-scoped functions** — cleaning, feature
   engineering, splitting, and feature selection are different jobs; keeping them
   in different functions makes each one easier to reason about and modify.
3. **Change one thing at a time** — whether tuning a model or fixing a bug,
   isolating a single variable is what makes a result interpretable.
4. **A column's stored dtype is a hint, not a guarantee** — always sanity-check
   whether "numeric" really means quantity, or secretly means category.
5. **Defensive coding by default** — safe list comprehensions, `pd.isna()` checks,
   `handle_unknown="ignore"` — small habits that prevent an entire pipeline from
   crashing on a single unexpected value.
6. **Document the "why," not just the "what"** — a verification log or experiment
   log is only valuable if it captures reasoning, not just numbers.
7. **Ship it like you built it** — test interactive/UI changes programmatically
   (Phase 14) before trusting a manual click-through, and confirm a deploy is
   actually live by checking it left your machine (committed AND pushed), not
   just that a file was saved.
