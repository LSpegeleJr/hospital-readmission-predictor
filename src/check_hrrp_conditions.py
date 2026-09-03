import pandas as pd

# ICD-9 prefixes for each HRRP diagnosis-based condition.
# Note: CABG and hip/knee replacement are procedure codes, not diagnosis codes,
# so they can't be checked this way — this dataset has no procedure code field.
# HCP stands fpr HRRP Condition Prefixes
# HCP is a dictionary that maps the key value pairs for conditions to ICD-9 code.  
# HCP is in all caps to let Python know it is a constanct and won't change.  
HCP = {
    "AMI (heart attack)": ["410"],
    "Heart Failure": ["428"],
    "Pneumonia": ["480", "481", "482", "483", "484", "485", "486"],
    "COPD": ["490", "491", "492", "496"],
}

# Creating a function that uses variables named code and prefixes
# if pd.isna(code): return False
# Checks to see if code is missing (NaN),  If it is missing, we can't check it
# against any prefix, so we immediately exit the function and hand back False
# ("not a match") to whoever called this function.  This does not loop to the next row
# That looping happens elsewhere (in df.apply).  This function only ever answers 
# True/False for the one code it was given
def matches_any_prefix(code, prefixes):
    """Check if a single ICD-9 code (as string) starts with any given prefix."""
    if pd.isna(code):
        return False
    code = str(code)
    return any(code.startswith(p) for p in prefixes)

def has_condition(row, prefixes):
    """Check diag_1, diag_2, and diag_3 — return True if ANY of them match."""
    return (
        matches_any_prefix(row["diag_1"], prefixes)
        or matches_any_prefix(row["diag_2"], prefixes)
        or matches_any_prefix(row["diag_3"], prefixes)
    )

# --- Age bracket logic ---
# Medicare eligibility generally starts at 65, but this dataset only records
# age in 10-year brackets. We treat [60-70) and above as our "Medicare-age"
# proxy population, acknowledging this includes some patients aged 60-64
# who would not yet be Medicare-eligible.
MEDICARE_AGE_BRACKETS = ["[60-70)", "[70-80)", "[80-90)", "[90-100)"]

# Opening this file directly will have the __name__ set to "__main__"
# If opened via another file then this files __name__ variable will be set to the name
# of this module "check_hrrp_conditions".  
if __name__ == "__main__":
    df = pd.read_csv("data/diabetic_data.csv")

    # Sets is_medicare_age varible to a column of True/False on where age column falls
    # within age backets in MEDICARE_AGE_BRACKETS.  
    # isin() checks every single value in tht list.  
    is_medicare_age = df["age"].isin(MEDICARE_AGE_BRACKETS)

    print(f"Total encounters: {len(df)}")
    print(f"Medicare-age proxy (60+): {is_medicare_age.sum()}\n")

    # Assigns variables "condition_name" and "prefixes" as the names for the key value
    # pair in dictionary HCP{}.  This is done once per iteration, cycling through
    # all four key value pairs in HCP.  
    # has_cond variable goes through every row in df and through the defined function 
    # has_condition checks all 3 diagnosis columns against the current condition's 
    # prefixes, which are iterated through from HCP
    # lambda defines a function with less code and without naming the function
    # In this case the function for every row returns has_condition(row,prefixes)
    # Since axis = 1 it runs once for every row, if axis=0 then once for every column
    # So has_cond function ends up being a full column of True/False values, one per row
    #  On the COPD iteration it determinse if patients diag_1/2/3 match any COPD prefix 
    # for all 101,766 rows.  
    
    # count = the total number of patients who BOTH have the current HRRP condition
    # (from has_cond) AND fall in the Medicare-age proxy bracket (from is_medicare_age).
    # The & combines the two True/False columns row-by-row (True only where both are
    # True), and .sum() adds up all the True values (True=1, False=0) into one number.
    for condition_name, prefixes in HCP.items():
        has_cond = df.apply(lambda row: has_condition(row, prefixes), axis=1)
        count = (has_cond & is_medicare_age).sum()
        print(f"{condition_name}: {count} (Medicare-age proxy)")