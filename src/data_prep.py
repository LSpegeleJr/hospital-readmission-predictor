# GETTING STARTED
    # Correct Interpreter 
        # CTRL+SHIFT+P -> "Python: Select Interpreter" -> .\venv\Scripts\python.exe
    # Navigate to project root
        # cd into the project root (wherever you cloned this repo)
    # Activate correct environment
        # venv\Scripts\activate
    # Click Play or run from terminal "python src\data_prep.py"
        # Play works here if interpretor is correct. 

"""
Data loading and cleaning for the Diabetes 130-US Hospitals dataset.

Usage:
    from src.data_prep import load_and_clean
    df = load_and_clean("data/diabetic_data.csv")
"""
# Numpy for numerical operations efficiency, pandas built on top of it.  
# Primarily using for np.nan, numpy's standard for representing "missing values" in a numeric manner.  
import pandas as pd
import numpy as np
from check_hrrp_conditions import HCP, has_condition

HRRP_COLUMN_NAMES = {
    "AMI (heart attack)": "has_ami",
    "Heart Failure": "has_heart_failure",
    "Pneumonia": "has_pneumonia",
    "COPD": "has_copd",
}

###################  PREPARING DATA ###################

# Columns that are IDs / near-duplicates / leak the target, drop up front
# encounter_id - a unique ID for each hospital visit, no predictive meaning
# weight — mostly missing
# payer_code — insurance billing code, not needed for readmission risk.  Lots missing values
DROP_COLS = ["encounter_id", "weight", "payer_code"]

# Columns that must stay in the DataFrame for grouping (patient_nbr) or a later
# fairness audit (race, gender), but should NOT be used as direct model inputs.
NON_FEATURE_COLS = ["patient_nbr", "race", "gender"]

# Low-variance medication columns (99%+ single value — see verification discussion)
# and raw diagnosis codes (too sparse/high-cardinality; add_hrrp_features() already
# extracts the useful signal from these). Excluded from modeling, not from the
# DataFrame itself, since there's no leakage/grouping reason to keep them around
# elsewhere the way there is for NON_FEATURE_COLS.
LOW_SIGNAL_COLS = [
    "nateglinide", "chlorpropamide", "acetohexamide", "tolbutamide", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide", "citoglipton",
    "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
    "diag_1", "diag_2", "diag_3",
]

###################  REMOVES BAD DATA & CLEANS DATA ###################

# load_and_clean function defined.  
# CLEANS DATA: 
#   Replaces "?" with pandas friendly NaN.  
#   Replaces empty cells with "Unknown", for text fields only.  
# REMOVES IRRELEVANT DATA:
    # Columns not relevant to our goal, DROP_COLS
    # Death/hospice patients (can't be readmitted)
# CREATES NECESSARY COLUMNS
    # "readmitted_30d"  Binary check if patient readmitted in 30 days.  
# "->" represents a note informing us the laod_and_clean() funtion results in a pandas DataFrame.  
# "->" can only be used after a functions parameter list and before the ":"
def load_and_clean(path: str) -> pd.DataFrame:
    # Reads a CSV file from the chosen path and sets the data frame.  
    df = pd.read_csv(path)

    # The dataset uses "?" for missing values instead of NaN, this replaced "?" with NaN.  
    df = df.replace("?", np.nan)

    # Drop columns that are IDs, mostly-missing, or not useful as-is
    # c for c in DROP_COLS if c in df.columns is looping through DROP_COLS one item at a time
    # calling each one c.  For each c, we check if it exists in df.columns.  If it does, c gets
    # included in the new list being built (otherwise skipped) for that iteration.  
    # The df.drop(columns) then removes the columns found in the list that was built by looping 
    # through DROP_COLS, keeping only the names that actually exist in df.columns.  
    # Safety feature to prevent system crashing if DROP_COLS has a typo
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # Binarize target: 1 = readmitted within 30 days, 0 = otherwise
    # Creating a column here called "readmitted_30d"
    # Since this column is binary it will hold a 1 if original value under readmitted was "0" or "< 30"
    # A zero for anything "">30" or "NO"
    # After creating the new column the old "readmitted" column is removed (dropped)
    df["readmitted_30d"] = (df["readmitted"] == "<30").astype(int)
    df = df.drop(columns=["readmitted"])

    # Drop encounters where the patient died or went to hospice (not real readmission risk)
    if "discharge_disposition_id" in df.columns:
        # 11 = expired, 19/20/21 = expired/hospice variants in IDs_mapping.csv
        # if discharge_disposition_id is 11, 19, 20 or 21 then produces a True, False otherwise
        # The negation operator ~ flips True to False & vice versa, so all 4 codes become False
        # pandas filters the True/Falso column in [], where it keeps only the Trues
        # Trues being rows that don't have the 4 IDs mentioned.  
        df = df[~df["discharge_disposition_id"].isin([11, 19, 20, 21])]

    # Simple missing-value handling for remaining categoricals
    # df.select_dtypes(include="object") selects columns by their data type, not by content
    # object is pandas catch-all label for columns with strings.  
    # So we are looking through all columns for text
    # cat_cols holds the list of column names that are text-typed
    # loop through each column name one at a time and .fillna("Unknown") replaces every missing TEXT
    # value with "Unknown".  
    cat_cols = df.select_dtypes(include=["object", "str"]).columns
    for c in cat_cols:
        df[c] = df[c].fillna("Unknown")

    # With several rows of data removed due to deaths and hospice, the DataFrame index is out of whack
    # this resets the indexing for the rows to a clean index count.  drop=True means don't create a column 
    # for the old gappy index as a new column.  
    # Return ends function and sends cleaned up df to be used.  
    return df.reset_index(drop=True)

###################  ADDING FEATURES TO IDENTIFY CONDITIONS & AGE ###################

def add_hrrp_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds one binary comorbidity flag column per identifiable HRRP condition
    (has_ami, has_heart_failure, has_pneumonia, has_copd), plus an age-based
    proxy flag (is_60_and_over), as new engineered features.

    Does not remove or modify any existing rows/columns — purely additive.
    """
    # Make copy of a DataFrame since modifying it can affect the original.  
    # copy() creates fully independent duplicate, changes stay local to this function
    df = df.copy()

    for condition_name, prefixes in HCP.items():
        # column_name gets set to whatever condition_name represents in a given iteration
        # The [] tells Python for the HRRP_COLUMN_NAMES treat the input as the key NOT value.  
        # When condition_name = "Heart Failure" -> column_name ="has_heart_failure"
        column_name = HRRP_COLUMN_NAMES[condition_name]

        # df[column_name] = ... creates a NEW column (named whatever column_name currently
        # holds, e.g. "has_copd") and assigns it the result of df.apply(...) — which goes
        # through every row in df, running has_condition for each one...
        # has_condition checks all 3 diagnosis columns against the current condition's 
        # prefixes, which are iterated through from HCP
        # lambda defines a function with less code and without naming the function
        # In this case the function for every row returns has_condition(row,prefixes)
        # Since axis = 1 it runs once for every row, if axis=0 then once for every column
        # So df[column_name] ends up being a full column of True/False values, one per row
        # On the COPD iteration it determines if patients diag_1/2/3 match any COPD prefix 
        # for all 101,766 rows.
        df[column_name] = df.apply(lambda row: has_condition(row, prefixes), axis=1)

    # Creates a new column "is_60_and_over" and indciates True if age falls in our interested ranges.  
    df["is_60_and_over"] = df["age"].isin(["[60-70)", "[70-80)", "[80-90)", "[90-100)"])

    return df

###################  SPLITS DATA 80% TRAIN & 20% TEST ###################

def train_test_split_df(df: pd.DataFrame, target="readmitted_30d", test_size=0.2, seed=42):
    from sklearn.model_selection import GroupShuffleSplit

    X = df.drop(columns=[target]) # X are Features, so remove output column
    y = df[target] # Y is the output, readmitted_30d

    groups = df["patient_nbr"] # tells splitter which patient each row belongs to, to keep things together

    # Creates splitter object n_splits=1, gives just one train/test split, will do multiple splits later
    # for cross validation.  
    # Splitter created with scikit-learn function GroupShuffleSplit(), which ignores X and Y but focuses on groups
    # The emphasis on groups here allows all visits by one patient to be grouped together and not split up
    # across different training and test sets. 
    # random_state=seed makes "random" split reproducible, results in identical splits for each run
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)

    # splitter.split(X, y, groups=groups) doesn't create split, creates a generator to produce
    # train/test index positions, which rows are train or test.  
    # next() pulls the one (and only) split from generator since n_splits=1
    # train_idx and test_idx tells us which rows designated as train and test
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    #X.iloc[train_idx] is pandas method for selecting rows by index position
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    return X_train, X_test, y_train, y_test

################### SETTING K-FOLDS ###################

def get_cv_folds(X: pd.DataFrame, y: pd.Series, n_splits=5, seed=42):
    from sklearn.model_selection import StratifiedGroupKFold
    groups = X["patient_nbr"]

    # StrateifiedGroupKFold creates groups where the class imbalance of 11% readmission rate is near constant
    # for each fold.  Shuffle=True randomizes the rows that are in each fold.  
    # The splitter here returns the splitters.split() result directly, instead of looping through to build
    # train/validation DataFrames (how train_test_split_df()) was done.  We allow the function to split the data.  
    # .split() returns a generator due to specific rules in scikit-learn library.  
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return splitter.split(X, y, groups=groups)

                                #### CONSOLIDATES FEATURES ####

# Removes columns that won't be Features.  
def get_model_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Removes columns that must be retained for grouping/splitting or a later
    fairness audit (see NON_FEATURE_COLS), and columns with too little signal
    or too much sparsity to be useful model inputs (see LOW_SIGNAL_COLS).
    """
    exclude = NON_FEATURE_COLS + LOW_SIGNAL_COLS
    return X.drop(columns=[c for c in exclude if c in X.columns])

# Opening this file directly will have the __name__ set to "__main__"
# If opened via another file then this files __name__ variable will be set to the name
# of this module "data_prep".  
if __name__ == "__main__":
    df = load_and_clean("data/diabetic_data.csv")
    print(f"Rows after cleaning: {len(df)}")
    print(f"Readmission rate (<30d): {df['readmitted_30d'].mean():.3f}")
