# GETTING STARTED
    # Correct Interpreter 
        # CTRL+SHIFT+P -> "Python: Select Interpreter" -> .\venv\Scripts\python.exe
    # Navigate to project root
        # cd "C:\Larry\Education\Self Projects\Hospital Readmission Risk Predictor"
    # Activate correct environment
        # venv\Scripts\activate
    # Click Play or run from terminal "python src\train.py"
        # Play works here if interpretor is correct.  
# This will take several minutes -- trains 15 model/fold combinations (3 models x 5 folds)

#################################### 1. LOADING AND PREPARING DATA ####################################

"""
Trains and compares Logistic Regression, Random Forest, and XGBoost for
30-day readmission risk, using 5-fold cross-validation on the 80% development
set, then evaluates the winning model once on the untouched 20% holdout.

Usage:
    python src/train.py
"""
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from data_prep import (
    load_and_clean, add_hrrp_features, train_test_split_df,
    get_cv_folds, get_model_features
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

from sklearn.metrics import roc_curve, precision_recall_curve
import matplotlib.pyplot as plt

DATA_PATH = "data/diabetic_data.csv"

# These columns are numeric but the numbers  are codes that represent specific categories.  
# Scaling these numbers would make them useless with the codes no longer meaning what they were meant to.  
# FORCE_CATEGORICAL_COLS is to be used to prevent these from being scaled.  
FORCE_CATEGORICAL_COLS = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]

df = load_and_clean(DATA_PATH)
df = add_hrrp_features(df) # Creates binary columns for categorical features to support Logistic Regress & XGBoost
X_train, X_test, y_train, y_test = train_test_split_df(df)

#################################### 2. PREPROCESSING PIPELINE ####################################

# Builds reusable transformation of numerical & text columns to allow every alorithm to be used.  
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Builds a preprocessing step that scales numeric columns and one-hot
    encodes categorical columns, so any model downstream receives clean,
    fully-numeric input regardless of the original column types.  
    Logistic Regression and XGBoost models need everything in numbers.  
    Some columns are stored as integers but are
    genuinely categorical (coded IDs, not true quantities) — these are forced
    into the categorical bucket regardless of their stored dtype.
    """
    
    numeric_cols = X.select_dtypes(include="number").columns.tolist() # List of all column names for numeric columns
    categorical_cols = X.select_dtypes(include=["object", "str", "bool"]).columns.tolist() # Same but for text columns

    # Removes any columns from FORCE_CATEGORICAL_COLS that are found in numeric_cols
    # then adds them to categorical_cols.  
    for col in FORCE_CATEGORICAL_COLS:
        if col in numeric_cols:
            numeric_cols.remove(col)
            categorical_cols.append(col)

    # ColumnTransformer() peforms transformations to different columns simultaneously, then combines results
    # into one unified numeric table.  Orchestrates following:
        # Numeric columns get scaled
        # Categorical columns go through one-hot encoding.  
    return ColumnTransformer([
        # StandardScaler() rescales numeric columns so they are on comparables scales (mean 0. std dev 1).  
        # New values are = (orignal value - column mean) / column std dev
            # Every original vlaue in column has this equation applied to it
            # Makes new mean 0 and new std dev 1
            # Spread of values typically become -2 and 2
            # Original spread of values were 
                # time_in_hospital 1-14
                # num_lab_procedures 0-130
            # New values used only in this portion of logic for model training.  
        # Matters most for Logistic Regression, sensitive to Features on wildly different scales.  
        ("numeric", StandardScaler(), numeric_cols),
        # OneHotEncoder() converts each category into own binary 0/1 column.  
            # age becomes 10 separate columns, one per bracket, and each 0 or 1.  
        # handle_inknown="ignore" is defensive strategy.  
            # If a category shows up in test, but not in training, then encodes all rows to 0 for that column
            # Prevents an error from occuring.  
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ])

#################################### 3. 5-FOLD CROSS VALIDATIONS ####################################

# MODELS {} is a dictionary.
    # Keys are the 3 strings, representing each model.  
    # Values are the 3 complex objects themselves.  The configured instances of the models w/ specific settings.  
MODELS = {

    # max_iter settings sets maximum iterations before it gives up.  Default in scikit-learn is 100
        # Since we have many one-hot encoded categorical columns we need more, 1,000 selected.  
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),

    # n_estimators setting sets # of trees.  More trees means more stable. less noisy predicitons, at cost of compute time.  
        # 300 trees is common setting.  
        # 300 tree built in parallel
    # Sample strategy, default taken.  bootstrap=True.
        # Each tree trains on random sample of rows, drawn w/ replacement
        # max_features="sqrt" by default
        # At each split point in a tree, only a random subset of features are considered (not all of them)
        # subset is specfically sqrt of total feature count.  
    # Depth
        # No depth limit is set this round, meaning trees keep splitting until every leaf is "pure" (contains 1 class)
        # or has too few samples left split further, default minimum is 2
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=10, class_weight="balanced", random_state=42),

    # n_estimator=300
        # For XGBoost this means 300 trees built in sequentially
        # Each new tree trained to correct errors of previous trees
    # max_depth=4
        # Depth set to 4 trees, shallower than Random Forest's unbounded depth.  
        # For XGBoost, shallow depths work with algorithm philosophy, each tree makes small incremental correction.  
        # Works best with many shallow trees. 
    # learning_rate=0.05
        # Controls how much each new tree's correction gets applied to overall prediction.  
        # Small learning rates mean each individual tree's contribution is intentionally shrunk
        # Model learns cautiously and gradually across 300 trees, instead of 1 tree overcorrecting
        # reduces overfitting Graident Boosting.  
    # subsample=0.8
        # For each individual tree XGBoost builds, only a fresh random 80% of training rows will be used
        # No tree sees full dataset (default is all threes see all data)
    # colsample_bytree=0.8
        # Similar to subsample each tree see a random 80% of feature columns
    # scal_pos_weight
        # XGBoost's version of class_weight="balanced"
        # Computers negative (no readmission) to positive (readmission) examples in training data
        # Since ~11% patients readmitted, ratio is ~ 8 (89/11 ~ 8)
        # Tells XGBoost to treat each positive example as worth 8x as much when calculating errors during training
        # Corrects class imbalance so model doesn't just learn to always predict "no readmission"
        # (y_train == 0).sum() counts # of patients not readmitted
        # (y_train == 1).sum() counts # of patients readmitted
        # Since readmission rate = 11.3% we are expecting the 0.887 / 0.113 when dividing the 2
        # ~ 8
    # eval_metric
        # Specifices mathematical loss funtction XGBoost tracks & tries to minimize during training.  
        # Log loss (logarithmic loss) is a standard choice for binacry classificaiton
        # Penalizes confident wrong prediction much more heavily than uncertain wrong predictions
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        eval_metric="logloss", random_state=42,
    ),
}


def run_cross_validation(X_train, y_train):
    """
    Runs 5-fold StratifiedGroupKFold cross-validation for each model in MODELS,
    returning a list of per-fold results (model name, fold number, ROC-AUC, PR-AUC).
    """
    results = []

    # oof_predicitons is a dictionary that loops through MODELS directly (no .items())
    # For each model name in MODELS, create brand-new, empty Series - one slot for every row in X_train
    # and store it in a dictionary, keyed by that model's name
        # Provides keys only
        # name takes on each models name
    # name: pd.Series(index=X_train.index, dtype=float) creates key: value pair
        # key = model name
        # value is brand new empty pd.Series
        # Empty series has same row labels as X_train via index=X_train
    oof_predictions = {name: pd.Series(index=X_train.index, dtype=float) for name in MODELS}

    # model_features called outside of loops since it doesn't change per-model or per-fold.  
    model_features = get_model_features(X_train)

    for model_name, model in MODELS.items():

        # get_cv_folds() called fresh, inside model loop.  
        # ensures each model gets its own set of 5 folds
        # Additonally, since the .split() creates a generator (scikit-learn rules), you need to place
        # folds inside the loop to ensure it runs for each loop iterations.  
        folds = get_cv_folds(X_train, y_train)
        for fold_num, (fold_train_idx, fold_val_idx) in enumerate(folds, start=1):
            X_fold_train = model_features.iloc[fold_train_idx] # set to indexed training data within model_features
            X_fold_val = model_features.iloc[fold_val_idx] # set to indexed validation data within model_features
            y_fold_train = y_train.iloc[fold_train_idx] # set to indexed training data
            y_fold_val = y_train.iloc[fold_val_idx] # set to indexed validation data

            # preprocessor set to build_preprocessor(x_fold_train), training data only
            preprocessor = build_preprocessor(X_fold_train) 

            # pipeline bundles preporcessor and model together.  
                # Allows .fit() and .predict_proba() to run data through both steps
            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", model),
            ])

            # pipeline.fit() triggers .fit() on both the prepocessor & the classifier in sequence
                # 1) Preprocessor
                    # "learns" training data.  
                    # StandardScaler calcualtes actual mean & std dev of each numeric column (for this fold's training row)
                    # OneHotEncoder discovers which categories exist in each categorical column.  
                    # Then transforms x_fold_train with what it learns.  
                # 2) Classifier
                    # Trains on transformed data for whichever model this iteration is on
                    # Receives fully numeric trnsformed data (x_fold_train and y_fold_train)
                    # adjusting its internal parameters to learn patterns that predict readmission (ML portion)
            pipeline.fit(X_fold_train, y_fold_train)

            # predict_proba returns a 2D array [rows,columns]
            # Returns 2 columns, probability of class 0 and class 1.  
            # [:, 1] sets all rows "unbounded" and 2nd column, probability of readmission
                # ROC-AUC & PR-AUC need readmission
            proba = pipeline.predict_proba(X_fold_val)[:, 1]

            # Single score: how well model ranks risk, per fold
            roc_auc = roc_auc_score(y_fold_val, proba)
            # Single score: precision/recall tradeoff, this fold (more senstive to 11% minority class)
            pr_auc = average_precision_score(y_fold_val, proba) 

            # Builds a dictionary to record specific model, specifc fold, and specific values for roc_auc and pr_auc
            results.append({
                "model": model_name, "fold": fold_num,
                "roc_auc": roc_auc, "pr_auc": pr_auc,
            })
            print(f"{model_name} — Fold {fold_num}: ROC-AUC={roc_auc:.3f}, PR-AUC={pr_auc:.3f}")

            # Fills in the placeholders in previously created empty series
                # Targets exact rows that were thi's folds validation set.  
                # Replaces empty placeholders with real predictions
            oof_predictions[model_name].iloc[fold_val_idx] = proba

    return results, oof_predictions

#################################### 4. PLOTTING CURVES ####################################

def plot_roc_pr_curves(y_train, oof_predictions, title_suffix="", save_path=None):
    """
    Plots real ROC and PR curves for all 3 models, overlaid, using pooled
    out-of-fold predictions. Save this figure alongside each experiment's
    entry in reports/model_experiment_log.md.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    colors = {"Logistic Regression": "#2980B9", "Random Forest": "#95A5A6", "XGBoost": "#27AE60"}

    ax = axes[0]
    for model_name, proba in oof_predictions.items():
        fpr, tpr, _ = roc_curve(y_train, proba)
        auc = roc_auc_score(y_train, proba)
        ax.plot(fpr, tpr, color=colors.get(model_name, "#333333"), linewidth=2, label=f"{model_name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#999999", linewidth=1.3, label="Random guess (AUC=0.5)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve{title_suffix}", fontweight="bold")
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    baseline = y_train.mean()
    for model_name, proba in oof_predictions.items():
        precision, recall, _ = precision_recall_curve(y_train, proba)
        ap = average_precision_score(y_train, proba)
        ax.plot(recall, precision, color=colors.get(model_name, "#333333"), linewidth=2, label=f"{model_name} (AP={ap:.3f})")
    ax.axhline(baseline, linestyle="--", color="#999999", linewidth=1.3, label=f"Random guess (baseline={baseline:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve{title_suffix}", fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor="white")
        print(f"Chart saved to: {save_path}")

#################################### 5. FINAL MODEL - TEST ####################################

import joblib

FINAL_MODEL = XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    eval_metric="logloss", random_state=42,
)

def train_final_model_and_evaluate(X_train, X_test, y_train, y_test):
    """
    Trains the winning model (XGBoost, Experiment 3 settings) on the FULL 80%
    development set, then evaluates it ONCE on the untouched 20% holdout test
    set. This is the final, honest performance number -- no further tuning
    decisions should be made based on this result.
    """
    X_train_features = get_model_features(X_train)
    X_test_features = get_model_features(X_test)

    # Build the pipeline for this specific model, don't rely on previous iterations
    preprocessor = build_preprocessor(X_train_features)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", FINAL_MODEL),
    ])
    pipeline.fit(X_train_features, y_train)

    proba = pipeline.predict_proba(X_test_features)[:, 1]
    roc_auc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)

    print("=== FINAL HOLDOUT TEST RESULTS ===")
    print(f"ROC-AUC: {roc_auc:.3f}")
    print(f"PR-AUC: {pr_auc:.3f}")

    # Takes fully trained pipeline object (preprocessor & classifier already fit)
        # Permanently saves it to disk at location
        # Allows it to be reloaded without retraining
        # The .joblib file is in a binary serialization formant, not human-readable
    joblib.dump(pipeline, "reports/final_model.joblib") 
    print("Model saved to reports/final_model.joblib")

    return pipeline, proba

EXPERIMENT_NAME = "experiment3_xgb_subsampling" # Change this before each new run

if __name__ == "__main__":
    results, oof_predictions = run_cross_validation(X_train, y_train)
    plot_roc_pr_curves(y_train, oof_predictions, title_suffix=f" — {EXPERIMENT_NAME}",
                        save_path=f"reports/{EXPERIMENT_NAME}_curves.png")