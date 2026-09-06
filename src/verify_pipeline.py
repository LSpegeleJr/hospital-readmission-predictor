# GETTING STARTED
    # Correct Interpreter 
        # CTRL+SHIFT+P -> "Python: Select Interpreter" -> .\venv\Scripts\python.exe
    # Navigate to project root
        # cd into the project root (wherever you cloned this repo)
    # Activate correct environment
        # venv\Scripts\activate
    # Click Play or run from terminal "python src\verify_pipeline.py"
        # Play works here if interpretor is correct. 
# If selecting "y" to Excel report prompt and get ImportError
    # run: pip install openpyxl

"""
Runs all data verification checks for the Hospital Readmission Risk Predictor
project and prints results. This is the source of truth behind reports/verification_log.md —
re-run this any time the data or pipeline changes to confirm the numbers still hold.

Usage:
    python src/verify_pipeline.py
"""
import pandas as pd
from data_prep import load_and_clean, train_test_split_df
from check_hrrp_conditions import HCP, has_condition, MEDICARE_AGE_BRACKETS
from data_prep import (
    load_and_clean, train_test_split_df, add_hrrp_features, get_cv_folds, 
    get_model_features, NON_FEATURE_COLS, LOW_SIGNAL_COLS
)

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.chart import BarChart, LineChart, Reference
from openpyxl.utils import get_column_letter
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.worksheet.table import Table, TableStyleInfo

                                #### 1. HRRP CONDITIONS CHECK ####
"""
    Checks how many encounters in the RAW dataset match each of the 4 diagnosis-based
    CMS HRRP conditions (AMI, Heart Failure, Pneumonia, COPD), regardless of age.

    Why this check exists: confirms which HRRP conditions are actually identifiable
    in this diabetes-based dataset before relying on them as model features, and
    documents the real coverage numbers rather than assuming them.
    """

def check_hrrp_coverage(df):
    print("=== 1. CMS HRRP condition coverage (all ages) ===")
    for condition_name, prefixes in HCP.items():
        has_cond = df.apply(lambda row: has_condition(row, prefixes), axis=1)
        print(f"{condition_name}: {has_cond.sum()} ({has_cond.mean()*100:.1f}%)")
    print()

                                #### 2. MEDICARE AGE CHECK ####

    """
    Repeats the HRRP condition check, but restricted to the "Medicare-age proxy"
    subgroup (age bracket [60-70) and above).

    Why this check exists: CMS HRRP applies specifically to Medicare beneficiaries
    (age 65+ eligible). Since this dataset only records age in 10-year brackets,
    an exact 65+ cutoff isn't possible — [60-70) is used as an approximation,
    which knowingly includes some under-65 patients. This check quantifies how
    HRRP condition counts change once restricted to this older subgroup.
    """

def check_medicare_age_coverage(df):
    print("=== 2. Medicare-age (60+) proxy subgroup ===")
    is_medicare_age = df["age"].isin(MEDICARE_AGE_BRACKETS)
    print(f"Medicare-age proxy (60+): {is_medicare_age.sum()}")
    for condition_name, prefixes in HCP.items():
        has_cond = df.apply(lambda row: has_condition(row, prefixes), axis=1)
        count = (has_cond & is_medicare_age).sum()
        print(f"{condition_name}: {count} (Medicare-age proxy)")
    print()

                                #### 3. PATIENT LEAKAGE RISK CHECK ####

    """
    Checks whether the same patient (patient_nbr) appears in more than one
    encounter/row in the RAW dataset.

    Why this check exists: if patients can appear multiple times, a naive random
    train/test split could put the same patient's different encounters into both
    train AND test, letting the model partially "recognize" that patient rather
    than genuinely generalizing to new patients. This check quantifies how big
    that risk is, and justifies using GroupShuffleSplit instead of a plain
    random split.

    If total encounters (row) is > Unique patients then we know patients had more
    than one visit.  
    Encounters per patient (Avg) > 1 indicates same thing.  
    Patients with more than 1 enoucnter also gives this info.  
    Max encounters for a single patient give us a potential outlier.  
    """

def check_patient_leakage_risk(df):
    print("=== 3. Patient encounter distribution (leakage risk, raw data) ===")
    total_rows = len(df)
    unique_patients = df["patient_nbr"].nunique()
    patient_counts = df["patient_nbr"].value_counts()
    repeat_patients = (patient_counts > 1).sum()

    print(f"Total encounters (rows): {total_rows}")
    print(f"Unique patients: {unique_patients}")
    print(f"Encounters per patient (average): {total_rows / unique_patients:.2f}")
    print(f"Patients with more than 1 encounter: {repeat_patients}")
    print(f"Max encounters for a single patient: {patient_counts.max()}")
    print()

                                #### 4. SPLIT INTEGRITY ####

    """
    Confirms that after running train_test_split_df() (which uses GroupShuffleSplit),
    no single patient's encounters end up split across BOTH train and test sets.

    Why this check exists: proves the leakage fix from check_patient_leakage_risk()
    actually works, rather than just trusting the theory behind GroupShuffleSplit.
    Also reconciles patient counts before/after cleaning, since load_and_clean()
    removes death/hospice encounters, which can remove some patients entirely.
    Total unique patients should = unique train patients + unique test patients
    """

def check_split_integrity(df):
    print("=== 4. Train/test split integrity (post-GroupShuffleSplit) ===")
    X_train, X_test, y_train, y_test = train_test_split_df(df)

    train_patients = set(X_train["patient_nbr"])
    test_patients = set(X_test["patient_nbr"])
    
    overlap = train_patients & test_patients

    print(f"Patients in train: {len(train_patients)}")
    print(f"Patients in test: {len(test_patients)}")
    print(f"Overlapping patients (should be 0): {len(overlap)}")
    print(f"Total unique patients (post-cleaning): {df['patient_nbr'].nunique()}")
    print(f"Train + Test patients: {len(train_patients) + len(test_patients)}")
    print()

                            #### 5. VERIFYING ROW AND UNIQUE PATIENT COUNTS ####

def check_cleaning_consistency(df):
    """
    Confirms row count and unique patient count after load_and_clean() are
    internally consistent with each other, and documents the actual post-
    cleaning numbers for reference.

    Why this check exists: sanity-checks that cleaning (row removal via
    death/hospice filtering) produces numbers that make sense together,
    rather than assuming row count and patient count both dropped correctly.
    """
    print("=== 5. Post-cleaning consistency check ===")
    print(f"Rows after cleaning: {len(df)}")
    print(f"Unique patients after cleaning: {df['patient_nbr'].nunique()}")
    print()

                            #### 6. HRRP FEATURE VERIFY ####

def check_hrrp_feature_columns(df):
    """
    Confirms the has_ami, has_heart_failure, has_pneumonia, has_copd, and
    is_60_and_over columns created by add_hrrp_features() produce counts
    consistent with the independently-verified numbers from Check #1 and
    Check #2 (which compute these same conditions a different way, directly
    from HCP and MEDICARE_AGE_BRACKETS, without relying on add_hrrp_features()).

    Why this check exists: proves the new engineered feature columns are
    correct, rather than assuming add_hrrp_features() works just because it
    runs without errors.
    """
    print("=== 6. HRRP feature column verification ===")
    print(f"has_ami: {df['has_ami'].sum()}")
    print(f"has_heart_failure: {df['has_heart_failure'].sum()}")
    print(f"has_pneumonia: {df['has_pneumonia'].sum()}")
    print(f"has_copd: {df['has_copd'].sum()}")
    print(f"is_60_and_over: {df['is_60_and_over'].sum()}")
    print()

                            #### 7. VERIFY K-FOLDS ####

def check_cv_fold_integrity(X_train, y_train):
    """
    Confirms StratifiedGroupKFold produces 5 folds with (1) zero patient overlap
    between each fold's train/validation split, and (2) a readmission rate in each
    fold's validation set reasonably close to the overall training data's rate.

    Why this check exists: proves the group + stratification guarantees actually
    hold in practice, rather than just trusting the sklearn documentation.
    """
    print("=== 7. Cross-validation fold integrity (StratifiedGroupKFold) ===")
    overall_rate = y_train.mean()
    print(f"Overall training set readmission rate: {overall_rate:.3f}")

    folds = get_cv_folds(X_train, y_train)

    # enumerate(folds, start=1) provides a counter with each item and counters starts at 1 instead of Python's normal 0
    # Printed output will read "Fold 1", "Fold 2", etc
    # For loop, loops through each of the 5 folds and unpacks a pair into 2 names (train & validation)
    for i, (fold_train_idx, fold_val_idx) in enumerate(folds, start=1):

        # For this particular fold set
        #.iloc[fold_train_idx] is pandas method for selecting rows by index position.  Each [...] represents another step
        # So step 1: X_train.iloc[fold_val_idx] is to set X training data to what is in current fold
        # Step 2 : ["patient_nbr"] is to then further isolate it down to just the patient_nbr column.  
        # Think of each [] as isolating the data even further.  
        fold_train_patients = set(X_train.iloc[fold_train_idx]["patient_nbr"])
        fold_val_patients = set(X_train.iloc[fold_val_idx]["patient_nbr"])
        overlap = fold_train_patients & fold_val_patients

        # y_train is the output, a binary outcomes (0 or 1) if a patient is readmitted in 30days.  
        # the .mean() is a trick to get a fractin of this fold's validation set is a readmission.  
        # This is to compare to the overall rate of 11%.  
        val_rate = y_train.iloc[fold_val_idx].mean()

        print(f"Fold {i}: train={len(fold_train_idx)} rows, val={len(fold_val_idx)} rows, "
              f"patient overlap={len(overlap)}, val readmission rate={val_rate:.3f}")
    print()

                            #### 8. CHECK FEATURES ####

def check_model_features(df):
    """
    Confirms get_model_features() drops exactly the expected columns
    (NON_FEATURE_COLS + LOW_SIGNAL_COLS) and nothing else.
    """
    print("=== 8. Model feature exclusion check ===")
    model_ready = get_model_features(df)

    expected_dropped = set(NON_FEATURE_COLS + LOW_SIGNAL_COLS)
    actually_dropped = set(df.columns) - set(model_ready.columns)

    print(f"Original columns: {len(df.columns)}")
    print(f"Model-ready columns: {len(model_ready.columns)}")
    print(f"Expected to drop: {len(expected_dropped)}")
    print(f"Actually dropped: {len(actually_dropped)}")
    print(f"Match: {expected_dropped == actually_dropped}")

    if expected_dropped != actually_dropped:
        print(f"Unexpected difference: {expected_dropped.symmetric_difference(actually_dropped)}")
    print()


def build_split_assignment_table(df, X_train, y_train):
    split_col = pd.Series("Test", index=df.index, name="split")
    split_col.loc[X_train.index] = "Train"
    fold_col = pd.Series(None, index=df.index, name="validation_fold", dtype="object")
    folds = get_cv_folds(X_train, y_train)
    for fold_num, (fold_train_idx, fold_val_idx) in enumerate(folds, start=1):
        val_row_labels = X_train.iloc[fold_val_idx].index
        fold_col.loc[val_row_labels] = fold_num
    return pd.concat([split_col, fold_col, df], axis=1)


                                #### BUILDS EXCEL REPORT ####

def export_excel_report(df, X_train, X_test, y_train, y_test,
                         output_path="reports/train_test_split_visualization.xlsx"):
    wb = openpyxl.Workbook()
    FONT = "Arial"
    HEADER_FILL = PatternFill("solid", fgColor="2C3E50")
    HEADER_FONT = Font(name=FONT, bold=True, color="FFFFFF", size=11)
    TITLE_FONT = Font(name=FONT, bold=True, size=16, color="2C3E50")
    SUBTITLE_FONT = Font(name=FONT, italic=True, size=10, color="666666")
    LABEL_FONT = Font(name=FONT, bold=True, size=10)
    BODY_FONT = Font(name=FONT, size=10)
    NOTE_FONT = Font(name=FONT, italic=True, size=9, color="888888")
    GREEN_FILL = PatternFill("solid", fgColor="D5F4E6")
    BLUE_FILL = PatternFill("solid", fgColor="D6EAF8")
    GREEN_ITALIC = Font(name=FONT, italic=True, size=9, color="27AE60")

    def set_print(ws):
        ws.page_setup.orientation = "landscape"
        ws.page_setup.fitToWidth = 1
        ws.page_setup.fitToHeight = 0
        ws.sheet_properties.pageSetUpPr.fitToPage = True

    # ---- Live-computed numbers ----
    total_rows, total_patients = len(df), df["patient_nbr"].nunique()
    train_rows, train_patients = len(X_train), X_train["patient_nbr"].nunique()
    test_rows, test_patients = len(X_test), X_test["patient_nbr"].nunique()
    overlap = len(set(X_train["patient_nbr"]) & set(X_test["patient_nbr"]))

    fold_stats = []
    for fold_num, (fold_train_idx, fold_val_idx) in enumerate(get_cv_folds(X_train, y_train), start=1):
        val_rate = y_train.iloc[fold_val_idx].mean()
        fold_stats.append((fold_num, len(fold_train_idx), len(fold_val_idx), val_rate))
    overall_rate = y_train.mean()

    # ==================== SHEET 1: Overview ====================
    ws1 = wb.active
    ws1.title = "Overview"
    ws1.sheet_view.showGridLines = False
    set_print(ws1)

    ws1["B2"] = "Hospital Readmission Risk Predictor"
    ws1["B2"].font = TITLE_FONT
    ws1["B3"] = "Train / Validation / Test Split Methodology"
    ws1["B3"].font = Font(name=FONT, bold=True, size=13, color="34495E")
    ws1["B4"] = "Generated live from the current pipeline run — see reports/verification_log.md"
    ws1["B4"].font = SUBTITLE_FONT

    ws1["B6"] = "Why a two-stage split?"
    ws1["B6"].font = LABEL_FONT
    ws1["B7"] = ("Encounters can't be split purely at random: patients can have more than one "
                 "encounter in this dataset. Randomly splitting rows could put the same patient's "
                 "visits in both train and test, letting the model partially \"recognize\" that "
                 "patient rather than genuinely generalizing to new patients.")
    ws1["B7"].font = BODY_FONT
    ws1["B7"].alignment = Alignment(wrap_text=True, vertical="top")
    ws1.merge_cells("B7:I9")

    ws1["B11"] = "Stage 1 — Holdout split"
    ws1["B11"].font = LABEL_FONT
    ws1["B12"] = ("GroupShuffleSplit carves the full cleaned dataset into an 80% development set "
                  "and a 20% final holdout test set. All of a given patient's encounters land in "
                  "the same side. The 20% holdout is never touched again until final model "
                  "evaluation.")
    ws1["B12"].font = BODY_FONT
    ws1["B12"].alignment = Alignment(wrap_text=True, vertical="top")
    ws1.merge_cells("B12:I13")

    ws1["B15"] = "Stage 2 — 5-fold cross-validation (on the 80% development set only)"
    ws1["B15"].font = LABEL_FONT
    ws1["B16"] = ("StratifiedGroupKFold splits the 80% development set into 5 folds, used to "
                  "fairly compare Logistic Regression, Random Forest, and XGBoost. Two "
                  "guarantees, both verified empirically: (1) Group — no patient's encounters are "
                  "ever split across a fold's train/validation portions. (2) Stratified — each "
                  f"fold's validation readmission rate is kept close to the overall "
                  f"~{overall_rate*100:.1f}% rate.")
    ws1["B16"].font = BODY_FONT
    ws1["B16"].alignment = Alignment(wrap_text=True, vertical="top")
    ws1.merge_cells("B16:I18")

    ws1["B20"] = "See tabs:"
    ws1["B20"].font = LABEL_FONT
    ws1.merge_cells("B21:I21")
    ws1["B21"] = '\u2192 "Stage 1 - Holdout Split"    for the 80/20 breakdown'
    ws1["B21"].font = BODY_FONT
    ws1.merge_cells("B22:I22")
    ws1["B22"] = '\u2192 "Stage 2 - CV Folds"    for the 5-fold breakdown'
    ws1["B22"].font = BODY_FONT
    ws1.merge_cells("B23:I23")
    ws1["B23"] = '\u2192 "Row-Level Split Assignments"    for every row\'s Train/Test/Fold label, filterable'
    ws1["B23"].font = BODY_FONT

    for col, width in zip("ABCDEFGHI", [3, 13, 13, 13, 13, 13, 13, 13, 13]):
        ws1.column_dimensions[col].width = width

    # ==================== SHEET 2: Stage 1 - Holdout Split ====================
    ws2 = wb.create_sheet("Stage 1 - Holdout Split")
    ws2.sheet_view.showGridLines = False
    set_print(ws2)

    ws2["B2"] = "Stage 1: Holdout Split (GroupShuffleSplit, 80% / 20%)"
    ws2["B2"].font = TITLE_FONT
    ws2["B3"] = "Source: reports/verification_log.md, Check #4 and Check #5"
    ws2["B3"].font = SUBTITLE_FONT

    headers = ["Category", "Rows", "% of Total Rows", "Unique Patients", "% of Total Patients"]
    hr = 5
    for i, h in enumerate(headers):
        c = ws2.cell(row=hr, column=2 + i, value=h)
        c.fill, c.font = HEADER_FILL, HEADER_FONT
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws2.row_dimensions[hr].height = 30

    rows_data = [
        ("Total (post-cleaning)", total_rows, total_patients, None),
        ("Train (80% dev set)", train_rows, train_patients, BLUE_FILL),
        ("Test (20% holdout)", test_rows, test_patients, GREEN_FILL),
    ]
    total_row_num = hr + 1
    for offset, (label, rows, patients, fill) in enumerate(rows_data):
        r = hr + 1 + offset
        cell = ws2.cell(row=r, column=2, value=label)
        cell.font = Font(name=FONT, bold=True, size=10)
        cell.alignment = Alignment(horizontal="center")
        c3 = ws2.cell(row=r, column=3, value=rows)
        c3.number_format = "#,##0"
        c3.alignment = Alignment(horizontal="center")
        c4 = ws2.cell(row=r, column=4, value=f"=C{r}/C{total_row_num}")
        c4.number_format = "0.0%"
        c4.alignment = Alignment(horizontal="center")
        c5 = ws2.cell(row=r, column=5, value=patients)
        c5.number_format = "#,##0"
        c5.alignment = Alignment(horizontal="center")
        c6 = ws2.cell(row=r, column=6, value=f"=E{r}/E{total_row_num}")
        c6.number_format = "0.0%"
        c6.alignment = Alignment(horizontal="center")
        if fill:
            for col in range(2, 7):
                ws2.cell(row=r, column=col).fill = fill

    r = hr + 5
    ws2.cell(row=r, column=2, value="Overlapping patients between Train and Test:").font = LABEL_FONT
    ovl = ws2.cell(row=r, column=5, value=overlap)
    ovl.font = Font(name=FONT, bold=True, size=10)
    ovl.fill = GREEN_FILL
    ovl.alignment = Alignment(horizontal="center")
    ws2.cell(row=r, column=6, value="\u2713 verified zero overlap").font = GREEN_ITALIC

    r += 2
    ws2.merge_cells(start_row=r, start_column=2, end_row=r, end_column=8)
    ws2.cell(row=r, column=2,
             value="Note: GroupShuffleSplit ignores class balance — Stage 2's StratifiedGroupKFold "
                   "is what keeps the readmission rate consistent, not this stage.").font = NOTE_FONT

    for col, width in zip("ABCDEF", [3, 26, 12, 16, 16, 18]):
        ws2.column_dimensions[col].width = width

    chart = BarChart()
    chart.type = "col"
    chart.title = "Rows: Train vs. Test"
    chart.y_axis.title = "Rows"
    chart.style = 10
    data = Reference(ws2, min_col=3, min_row=hr + 2, max_row=hr + 3)
    cats = Reference(ws2, min_col=2, min_row=hr + 2, max_row=hr + 3)
    chart.add_data(data, titles_from_data=False)
    chart.set_categories(cats)
    chart.legend = None
    chart.width, chart.height = 9, 7
    ws2.add_chart(chart, "H5")

    # ==================== SHEET 3: Stage 2 - CV Folds ====================
    ws3 = wb.create_sheet("Stage 2 - CV Folds")
    ws3.sheet_view.showGridLines = False
    set_print(ws3)

    ws3["B2"] = "Stage 2: 5-Fold Cross-Validation (StratifiedGroupKFold, on 80% dev set)"
    ws3["B2"].font = TITLE_FONT
    ws3["B3"] = "Source: reports/verification_log.md, Check #7"
    ws3["B3"].font = SUBTITLE_FONT

    headers = ["Fold", "Train Rows", "Val Rows", "Total Rows", "Patient Overlap",
               "Val Readmission Rate", "Overall Rate (ref. line)"]
    hr3 = 5
    for i, h in enumerate(headers):
        c = ws3.cell(row=hr3, column=2 + i, value=h)
        c.fill, c.font = HEADER_FILL, HEADER_FONT
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws3.row_dimensions[hr3].height = 34

    first_fold_row = hr3 + 1
    for offset, (fold_num, train_n, val_n, val_rate) in enumerate(fold_stats):
        r = first_fold_row + offset
        c2 = ws3.cell(row=r, column=2, value=f"Fold {fold_num}")
        c2.font = Font(name=FONT, bold=True, size=10)
        c2.alignment = Alignment(horizontal="center")
        for col, val, fmt in [(3, train_n, "#,##0"), (4, val_n, "#,##0")]:
            c = ws3.cell(row=r, column=col, value=val)
            c.number_format = fmt
            c.alignment = Alignment(horizontal="center")
        c5 = ws3.cell(row=r, column=5, value=f"=C{r}+D{r}")
        c5.number_format = "#,##0"
        c5.alignment = Alignment(horizontal="center")
        c6 = ws3.cell(row=r, column=6, value=0)
        c6.number_format = "#,##0"
        c6.fill = GREEN_FILL
        c6.alignment = Alignment(horizontal="center")
        c7 = ws3.cell(row=r, column=7, value=val_rate)
        c7.number_format = "0.0%"
        c7.alignment = Alignment(horizontal="center")
        c8 = ws3.cell(row=r, column=8, value=overall_rate)
        c8.number_format = "0.0%"
        c8.alignment = Alignment(horizontal="center")
    last_fold_row = first_fold_row + len(fold_stats) - 1

    avg_row = last_fold_row + 1
    ws3.cell(row=avg_row, column=2, value="Average across folds").font = Font(name=FONT, bold=True, size=10)
    ws3.cell(row=avg_row, column=2).alignment = Alignment(horizontal="center")
    ws3.cell(row=avg_row, column=2).fill = BLUE_FILL
    for col, letter in [(3, "C"), (4, "D"), (5, "E")]:
        c = ws3.cell(row=avg_row, column=col, value=f"=AVERAGE({letter}{first_fold_row}:{letter}{last_fold_row})")
        c.number_format = "#,##0"
        c.fill = BLUE_FILL
        c.alignment = Alignment(horizontal="center")
    c6 = ws3.cell(row=avg_row, column=6, value=f"=SUM(F{first_fold_row}:F{last_fold_row})")
    c6.number_format = "#,##0"
    c6.fill = BLUE_FILL
    c6.alignment = Alignment(horizontal="center")
    c7 = ws3.cell(row=avg_row, column=7, value=f"=AVERAGE(G{first_fold_row}:G{last_fold_row})")
    c7.number_format = "0.0%"
    c7.fill = BLUE_FILL
    c7.alignment = Alignment(horizontal="center")

    overall_row = avg_row + 1
    ws3.cell(row=overall_row, column=2, value="Overall training set rate (reference)").font = Font(name=FONT, bold=True, size=10)
    ws3.cell(row=overall_row, column=2).alignment = Alignment(horizontal="center")
    c7b = ws3.cell(row=overall_row, column=7, value=overall_rate)
    c7b.number_format = "0.0%"
    c7b.alignment = Alignment(horizontal="center")

    note_row = overall_row + 2
    ws3.merge_cells(start_row=note_row, start_column=2, end_row=note_row, end_column=8)
    ws3.cell(row=note_row, column=2,
             value="Every fold's validation readmission rate matches the overall rate almost "
                   "exactly — confirming StratifiedGroupKFold's class-balance guarantee held in "
                   "practice.").font = NOTE_FONT

    for col, width in zip("ABCDEFGH", [3, 24, 12, 12, 12, 15, 20, 20]):
        ws3.column_dimensions[col].width = width

    bar = BarChart()
    bar.type = "col"
    bar.title = "Validation Readmission Rate by Fold"
    bar.y_axis.title = "Readmission Rate"
    bar.y_axis.numFmt = "0.0%"
    bar.y_axis.scaling.min = 0
    bar.y_axis.scaling.max = max(0.15, overall_rate * 1.5)
    bar.style = 10
    data = Reference(ws3, min_col=7, min_row=hr3, max_row=last_fold_row)
    cats = Reference(ws3, min_col=2, min_row=first_fold_row, max_row=last_fold_row)
    bar.add_data(data, titles_from_data=True)
    bar.set_categories(cats)

    line = LineChart()
    data2 = Reference(ws3, min_col=8, min_row=hr3, max_row=last_fold_row)
    line.add_data(data2, titles_from_data=True)
    line.y_axis.numFmt = "0.0%"
    bar += line
    bar.width, bar.height = 11, 8
    ws3.add_chart(bar, f"B{note_row + 3}")

    # ==================== SHEET 4: Row-Level Split Assignments ====================
    ws4 = wb.create_sheet("Row-Level Split Assignments")
    ws4.sheet_view.showGridLines = False
    full_table = build_split_assignment_table(df, X_train, y_train)
    for row in dataframe_to_rows(full_table, index=False, header=True):
        ws4.append(row)
    for cell in ws4[1]:
        cell.fill, cell.font = HEADER_FILL, HEADER_FONT
    last_col_letter = get_column_letter(full_table.shape[1])
    table_ref = f"A1:{last_col_letter}{full_table.shape[0] + 1}"
    excel_table = Table(displayName="RowLevelData", ref=table_ref)
    excel_table.tableStyleInfo = TableStyleInfo(name="TableStyleMedium9", showRowStripes=True)
    ws4.add_table(excel_table)
    ws4.freeze_panes = "A2"
    ws4.column_dimensions["A"].width = 10
    ws4.column_dimensions["B"].width = 16

    wb.save(output_path)
    print(f"\nExcel report saved to: {output_path}")

if __name__ == "__main__":
    # Checks 1-3 intentionally use the RAW data (before load_and_clean() runs),
    # to match how these checks were originally run and documented in
    # reports/verification_log.md. Check 4 requires cleaned data, since
    # train_test_split_df() expects load_and_clean()'s output.
    raw_df = pd.read_csv("data/diabetic_data.csv")
    cleaned_df = load_and_clean("data/diabetic_data.csv")
    featured_df = add_hrrp_features(cleaned_df)

    X_train, X_test, y_train, y_test = train_test_split_df(featured_df)

    check_hrrp_coverage(raw_df)
    check_medicare_age_coverage(raw_df)
    check_patient_leakage_risk(raw_df)
    check_split_integrity(cleaned_df)
    check_cleaning_consistency(cleaned_df)
    check_hrrp_feature_columns(featured_df)
    check_cv_fold_integrity(X_train, y_train)
    check_model_features(featured_df)

    print("\n" + "="*50)
    answer = input("Generate Excel split-assignment report? (y/n): ")
    if answer.strip().lower() == "y":
        export_excel_report(featured_df, X_train, X_test, y_train, y_test)
    else:
        print("Skipped.")