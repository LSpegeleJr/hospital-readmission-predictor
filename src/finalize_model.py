# GETTING STARTED
    # Correct Interpreter 
        # CTRL+SHIFT+P -> "Python: Select Interpreter" -> .\venv\Scripts\python.exe
    # Navigate to project root
        # cd into the project root (wherever you cloned this repo)
    # Activate correct environment
        # venv\Scripts\activate
    # Click Play or run from terminal "python src\finalize_model.py"
        # Play works here if interpretor is correct.  
# Run this ONLY after you've finished tuning in train.py -- this trains the
# FINAL model on the full 80% dev set and touches the holdout test set ONCE.
# Importing from train.py here only re-runs its fast data loading, not the
# expensive 15-fold comparison (that logic is inside train.py's own __main__)

"""
Trains the final, selected model (XGBoost, Experiment 3 settings) on the full
80% development set, and evaluates it once on the untouched 20% holdout test
set.

Importing from train.py only re-runs its data loading/cleaning (fast) -- it
does NOT re-run the 3-model cross-validation comparison, since that lives
inside train.py's own `if __name__ == "__main__":` block, which only
triggers when train.py itself is run directly, never when it's imported.

Usage:
    python src/finalize_model.py
"""
from train import (
    X_train, X_test, y_train, y_test,
    train_final_model_and_evaluate, plot_roc_pr_curves,
)

if __name__ == "__main__":
    final_pipeline, test_proba = train_final_model_and_evaluate(X_train, X_test, y_train, y_test)
    plot_roc_pr_curves(y_test, {"XGBoost (Final Model)": test_proba},
                        title_suffix=" — Final Holdout Test",
                        save_path="reports/final_holdout_curves.png")