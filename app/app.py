# Getting started
    # Correct Interpretor
    # CTRL + SHIFT + P
    # Select Python: Select Interpreer
    # .\venv\Scripts\python.exe
# Active virtual enviornment
    # If venv does not show in project folder
    # type in terminal "venv\Scripts\activate"
    # Streamlit sits in venv so it must be active.  
# app.py must be launched with streamlit
    # Do not click play in VS Code - streamlit apps require own launcher
    # Type in terminal "streamlit run app/app.py"

"""
Streamlit app: clinician enters a patient's full discharge record (all 30
features the model was trained on), gets a 30-day readmission risk score
plus a SHAP breakdown of the top factors driving that specific prediction.
 
Run with:
    streamlit run app/app.py
"""
import sys, os

# os.path.dirname(__file__)
    # __file__ is like __name__ except represents file path (location)
    # os.path.dirname(...) strips off file name, leaving its parent folder, "app/" in this case.  
# os.path.join(os.path.dirname(..., "..", "src")
    # os.path.join()  safely combines pieces (handling \ vs / slash differences between windows & other os)
    # ".." means go up a folder, so goes from "app/" to project folder
    # "src" is joined to create new folder location src/
# sys.path.append(...)
    # sys.path is list of folders Python earches through whenever you write import something
    # Automatically and dynamically fills it in
# This process let's us connect to any file under src/, essentially future proofing for future projects.  
    # How we connect to data_prep.py without calling for it.  
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
 
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Streamlit funcitons, not general Python
    # Every st.somethin() draws something on a webpage.  
    # set_page_config controls page-level settings
    # st.title draws a large heading
    # st.caption draws small muted subyext below it
st.set_page_config(page_title="Readmission Risk Predictor", layout="wide")
st.title("30-Day Readmission Risk Predictor")
st.caption("Prototype — trained on the UCI Diabetes 130-US Hospitals dataset. Not for clinical use.")
 
MODEL_PATH = "reports/final_model.joblib"
 
 # @st.cache_resource is a decorator
    # Sits directly above a function definition (no blank line between)
    # modifies how function behaves
    # cache_resource is a tool that lives in Streamlit
        # tells Streamlit to run this function once, & remember what is returned
        # on every future call just hand back what was returned.  
@st.cache_resource
def load_model():

    # .load(MODEL_PATH)
        # Reads from MODEL_PATH, which is reports/final_model.joblib
        # Comes back w/ same fitted Pipeline object (preprocessor + classified, already trained)
    return joblib.load(MODEL_PATH)
 
# If finalize_model.py  got moved or deleted, then except would run
    # st.warning Provide warning
    # st.stop stops rest of script from running
try:
    pipeline = load_model()
except FileNotFoundError:
    st.warning(f"No trained model found at `{MODEL_PATH}`. Run `python src/finalize_model.py` first.")
    st.stop()
 
# ---------------------------------------------------------------------------
# Human-readable labels for the 3 coded ID columns (from IDs_mapping.csv).
# Dropdowns display the description; the underlying numeric code is what
# actually gets sent to the model.
# ---------------------------------------------------------------------------
ADMISSION_TYPE_OPTIONS = {
    1: "Emergency", 2: "Urgent", 3: "Elective", 4: "Newborn",
    5: "Not Available", 6: "Unknown/NaN", 7: "Trauma Center", 8: "Not Mapped",
}
DISCHARGE_DISPOSITION_OPTIONS = {
    1: "Discharged to home", 2: "Transferred to another short term hospital",
    3: "Transferred to SNF", 4: "Transferred to ICF",
    6: "Transferred to home with home health service", 7: "Left AMA",
    8: "Transferred to home under IV provider care",
    12: "Still patient / expected to return for outpatient services",
    15: "Transferred within institution to Medicare swing bed",
    16: "Transferred/referred to another institution for outpatient services",
    17: "Transferred/referred to this institution for outpatient services",
    22: "Transferred to another rehab facility", 23: "Transferred to long term care hospital",
    24: "Transferred to nursing facility (Medicaid only)",
    27: "Transferred to a federal health care facility",
    28: "Transferred to a psychiatric hospital", 29: "Transferred to a Critical Access Hospital",
}
ADMISSION_SOURCE_OPTIONS = {
    1: "Physician Referral", 2: "Clinic Referral", 3: "HMO Referral",
    4: "Transfer from a hospital", 5: "Transfer from a Skilled Nursing Facility",
    6: "Transfer from another health care facility", 7: "Emergency Room",
    8: "Court/Law Enforcement", 9: "Not Available",
    10: "Transfer from critical access hospital", 11: "Normal Delivery",
    12: "Premature Delivery", 13: "Sick Baby", 14: "Extramural Birth",
    18: "Transfer From Another Home Health Agency",
    19: "Readmission to Same Home Health Agency", 22: "Transfer from hospital inpatient",
    23: "Born inside this hospital", 24: "Born outside this hospital",
    25: "Transfer from Ambulatory Surgery Center", 26: "Transfer from Hospice",
}
 
AGE_BRACKETS = ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
                "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"]
 
MEDICAL_SPECIALTIES = [
    "AllergyandImmunology", "Anesthesiology", "Anesthesiology-Pediatric", "Cardiology",
    "Cardiology-Pediatric", "DCPTEAM", "Dentistry", "Dermatology", "Emergency/Trauma",
    "Endocrinology", "Endocrinology-Metabolism", "Family/GeneralPractice", "Gastroenterology",
    "Gynecology", "Hematology", "Hematology/Oncology", "Hospitalist", "InfectiousDiseases",
    "InternalMedicine", "Nephrology", "Neurology", "Neurophysiology",
    "Obsterics&Gynecology-GynecologicOnco", "Obstetrics", "ObstetricsandGynecology", "Oncology",
    "Ophthalmology", "Orthopedics", "Orthopedics-Reconstructive", "Osteopath", "Otolaryngology",
    "OutreachServices", "Pathology", "Pediatrics", "Pediatrics-AllergyandImmunology",
    "Pediatrics-CriticalCare", "Pediatrics-EmergencyMedicine", "Pediatrics-Endocrinology",
    "Pediatrics-Hematology-Oncology", "Pediatrics-InfectiousDiseases", "Pediatrics-Neurology",
    "Pediatrics-Pulmonology", "Perinatology", "PhysicalMedicineandRehabilitation",
    "PhysicianNotFound", "Podiatry", "Proctology", "Psychiatry", "Psychiatry-Addictive",
    "Psychiatry-Child/Adolescent", "Psychology", "Pulmonology", "Radiologist", "Radiology",
    "Resident", "Rheumatology", "Surgeon", "Surgery-Cardiovascular",
    "Surgery-Cardiovascular/Thoracic", "Surgery-Colon&Rectal", "Surgery-General",
    "Surgery-Maxillofacial", "Surgery-Neuro", "Surgery-Pediatric", "Surgery-Plastic",
    "Surgery-PlasticwithinHeadandNeck", "Surgery-Thoracic", "Surgery-Vascular",
    "SurgicalSpecialty", "Unknown", "Urology",
]
 
MED_STATUS_OPTIONS = ["No", "Steady", "Up", "Down"]
 
# ---------------------------------------------------------------------------
# Input form -- organized into sections for readability, all 30 model
# features represented. Numeric widgets use real min/median/max from the
# training data as sensible bounds/defaults.
# ---------------------------------------------------------------------------
st.subheader("Patient encounter details")

# Splits screen into 3 columns with the 3 specific objects called out
    # 1 object per column
col1, col2, col3 = st.columns(3)

# Everything indented under col1 gets placed inside, same for other 2 objects.   
with col1:
    st.markdown("**Demographics & Admission**")
    age = st.selectbox("Age bracket", AGE_BRACKETS, index=6)
    admission_type_label = st.selectbox("Admission type", list(ADMISSION_TYPE_OPTIONS.values()), index=0)
    admission_source_label = st.selectbox("Admission source", list(ADMISSION_SOURCE_OPTIONS.values()), index=6)
    discharge_disposition_label = st.selectbox("Discharge disposition", list(DISCHARGE_DISPOSITION_OPTIONS.values()), index=0)
    medical_specialty = st.selectbox("Admitting specialty", MEDICAL_SPECIALTIES, index=18)
    is_60_and_over = st.checkbox("Age 60 and over (Medicare-age proxy)", value=False)
 
with col2:
    st.markdown("**Visit Intensity**")
    time_in_hospital = st.slider("Days in hospital", 1, 14, 4)
    num_lab_procedures = st.slider("Number of lab procedures", 1, 132, 44)
    num_procedures = st.slider("Number of procedures", 0, 6, 1)
    num_medications = st.slider("Number of medications", 1, 81, 15)
    number_diagnoses = st.slider("Number of diagnoses", 1, 16, 8)
    number_outpatient = st.number_input("Prior outpatient visits (past yr)", 0, 42, 0)
    number_emergency = st.number_input("Prior ER visits (past yr)", 0, 76, 0)
    number_inpatient = st.number_input("Prior inpatient visits (past yr)", 0, 21, 0)
 
with col3:
    st.markdown("**Labs & Diabetes Management**")
    max_glu_serum = st.selectbox("Max glucose serum test", ["Unknown", "Norm", ">200", ">300"], index=0)
    a1c_result = st.selectbox("A1C test result", ["Unknown", "Norm", ">7", ">8"], index=0)
    change = st.selectbox("Diabetes medication changed?", ["No", "Ch"], index=0)
    diabetes_med = st.selectbox("On any diabetes medication?", ["No", "Yes"], index=1)
    st.markdown("**HRRP Comorbidity Flags**")
    has_ami = st.checkbox("History of AMI (heart attack)", value=False)
    has_heart_failure = st.checkbox("History of heart failure", value=False)
    has_pneumonia = st.checkbox("History of pneumonia", value=False)
    has_copd = st.checkbox("History of COPD", value=False)
 
st.markdown("**Individual Medications** (No / Steady / Up / Down)")
med_col1, med_col2, med_col3, med_col4 = st.columns(4)
with med_col1:
    metformin = st.selectbox("Metformin", MED_STATUS_OPTIONS, index=0)
    repaglinide = st.selectbox("Repaglinide", MED_STATUS_OPTIONS, index=0)
with med_col2:
    glimepiride = st.selectbox("Glimepiride", MED_STATUS_OPTIONS, index=0)
    glipizide = st.selectbox("Glipizide", MED_STATUS_OPTIONS, index=0)
with med_col3:
    glyburide = st.selectbox("Glyburide", MED_STATUS_OPTIONS, index=0)
    pioglitazone = st.selectbox("Pioglitazone", MED_STATUS_OPTIONS, index=0)
with med_col4:
    rosiglitazone = st.selectbox("Rosiglitazone", MED_STATUS_OPTIONS, index=0)
    insulin = st.selectbox("Insulin", MED_STATUS_OPTIONS, index=0)
 
# ---------------------------------------------------------------------------
# Build the single-row input DataFrame. Column names and order don't
# technically need to match the training data's order (ColumnTransformer
# matches by NAME, not position) -- but every column name must match exactly.
# ---------------------------------------------------------------------------
def build_input_row():

    # for k, v in ADMISSION_TYPE_OPTIONS.items()
        # .items() loops through every key-value pair
        # k = code (like 1)
        # v = label (like Emergency)
    # if v == admission_type_label
        # Actual reverse-lookup, keeps only pair who value matches user selection
    # [k for ... if ...]
        # builds a list of keys (k) where condition is true
        # [0] since dictionary values are unique (no 2 codes have same label) list will only have 1 match
            # grabs that single result
    admission_type_id = [k for k, v in ADMISSION_TYPE_OPTIONS.items() if v == admission_type_label][0]
    admission_source_id = [k for k, v in ADMISSION_SOURCE_OPTIONS.items() if v == admission_source_label][0]
    discharge_disposition_id = [k for k, v in DISCHARGE_DISPOSITION_OPTIONS.items() if v == discharge_disposition_label][0]

    # {...} dictionary, one key-value pair per model feature
        # key is exact column name the model expects
        # Must match training, recall ColumnTransformer
        # ColunTransformer was fit on named columns, needs to receive data in that format
        # Value is whatever user selected/entered
    # [{...}] wraps single dictionary in a list
        # pd.DataFrame() expects a list of row-dictionaries (even if just one row)
        # A list containing one dictionary produces a DataFrame with exactly one row
    # pd.DataFrame([...]) constructs actual single-row table
        # column names match dictionary keys
        # ready to be handed to pipeline.predict_proba(...)
    return pd.DataFrame([{
        "age": age,
        "admission_type_id": admission_type_id,
        "discharge_disposition_id": discharge_disposition_id,
        "admission_source_id": admission_source_id,
        "time_in_hospital": time_in_hospital,
        "medical_specialty": medical_specialty,
        "num_lab_procedures": num_lab_procedures,
        "num_procedures": num_procedures,
        "num_medications": num_medications,
        "number_outpatient": number_outpatient,
        "number_emergency": number_emergency,
        "number_inpatient": number_inpatient,
        "number_diagnoses": number_diagnoses,
        "max_glu_serum": max_glu_serum,
        "A1Cresult": a1c_result,
        "metformin": metformin,
        "repaglinide": repaglinide,
        "glimepiride": glimepiride,
        "glipizide": glipizide,
        "glyburide": glyburide,
        "pioglitazone": pioglitazone,
        "rosiglitazone": rosiglitazone,
        "insulin": insulin,
        "change": change,
        "diabetesMed": diabetes_med,
        "has_ami": has_ami,
        "has_heart_failure": has_heart_failure,
        "has_pneumonia": has_pneumonia,
        "has_copd": has_copd,
        "is_60_and_over": is_60_and_over,
    }])
 
# draws horizontal line across page
st.divider()

# st.button(...) draws clickable button
    # Returns True only on single script-rerun after someone clicks it
    # Returns False upon initial page load or whenever a rerun is triggered.  
    # Everything under if only executes if button clicked.  
    # type="primary" makes button standout, typically colored.  
if st.button("Predict readmission risk", type="primary"):
    # input_row has only one ow (hypothetical patient)
    input_row = build_input_row()

        # pipeline.predict_proba(input_row)[:, 1] returns array w/ exactly one # in it
        # pipeline ensure parameters flow through trained model
        # [:,1] represents all rows with ":" unbounded, and column 1 (probability of readmission)
        # [0] selects that # and pulls out raw # itself
        # proba becomes a plain float.  
    proba = pipeline.predict_proba(input_row)[:, 1][0]

    # st.columns([1, 2]) uses a list this time, indicates the width of the columns
        # 2nd column, chart_col will be twice as wide as risk_col
    risk_col, chart_col = st.columns([1, 2])
    with risk_col:

        # Streamlit widget to display one prominent number w/ a label styled like dashboard KPI card
        # f"{proba:.1%}"
            # f string that joins probability value in proba
            # .1% inside f-string formats  as percentage w/ 1 decimal place
            # automatically multiplies by 100 and adds a %
        st.metric("30-day readmission risk", f"{proba:.1%}")
        if proba >= 0.5:
            st.error("High risk")
        elif proba >= 0.2:
            st.warning("Moderate risk")
        else:
            st.success("Lower risk")
 
    # -----------------------------------------------------------------
    # SHAP = SHapley Additive exPlanations
        # Takes baseline (expected model output over background dataset)
        # Adds summation of SHAP values for each feature
        # Demonstrates how much each feature pushed prediction up or down relative to baseline
    # SHAP explanation for this specific prediction. The pipeline's
    # preprocessor transforms the raw input into the encoded feature
    # space the classifier actually trained on; TreeExplainer works on
    # that transformed representation, since it needs the real XGBoost
    # booster and matching numeric input.
    # -----------------------------------------------------------------
    
    # .named_steps
        # From scikit-learn and is an attribute of a Pipeline
        # Gives dictionary mapping each step's name to actual estimator/transformer object
        # for our pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", model)])
        # pipeline.named_steps results in
            # {
            # "preprocessor": preprocessor(...)
            # "classifier": model(...)
            # }
        # Allows for inspecting internal components
        # pipeline.named_steps["preprocessor"] pulls out already-fitted ColumnTransformer
        # Separate from classifier
        # SHAP needs to work with each piece individually, not as combined whole
    fitted_preprocessor = pipeline.named_steps["preprocessor"]
    fitted_classifier = pipeline.named_steps["classifier"]

    # fitted_preprocessor.transform(input_row)
        # Runs single patient through exact same scailing/encoding rules preprocessor learned in training
        # Produces fully numeric array the classifier operates on internally (2D array, 1 row w/ many columns)
        # Necessary since TreeExplainer needs real input representation, not human-readable version
    transformed_input = fitted_preprocessor.transform(input_row)

    # .get_feature_names_out()
        # From scikit-learn and returns names of output features produced by transformer
        # Standard way for getting which columns come out of transformer
        # Includes names of all columns created with one-hot encoding
    feature_names = fitted_preprocessor.get_feature_names_out()

    # Strip the "numeric__"/"categorical__" prefix ColumnTransformer adds,
    # for a cleaner display.
    # .split("__",1)[1]
        # splits on first double underscore
        # 1 limits it to first split only
        # protects columns that have own underscores further in
        # [1] grabs everything after first split
        # Essentially, stips off "numeric__" or "categorical__" prefixes ColumnTransformer automatically adds
    # if "__" in n else n
        # Guards against any name that doesn't have prefix pattern
        # Leaves untouched instead of crashing
    clean_names = [n.split("__", 1)[1] if "__" in n else n for n in feature_names]

    # shap.TreeExplainer(fitted_classifier)
        # Creates a SHAP explainer specifically built for tree-based models
        # explainer becomes an object, instance of TreeExplainer class
    explainer = shap.TreeExplainer(fitted_classifier)

    #### NOTE: SHAP values are real no-zero impact score for each feature, even ones that have a 0 for this patient ####

    # explainer(transformed_input)
        # Runs actual explanation
        # Calculates how much each of the 100+ encoded features pushed prediction up or down from baseline/average output
        # shap_values becomes a SHAP Explanation object
    shap_values = explainer(transformed_input)

    # shap_values.values[0]
        # Becomes a 2D array, 1 row * however many encoded columns you have
        # [0] pulls out row 0 of array, only row (one patient)
        # Results in flat 1D array: 1 SHAP value per encoded feature
        # No longer wrapped in outer "rows" layer (no column names)
    # pd.Series() with index=clean_names
        # Takes 1D array and wraps into pd.Series
        # Pairs each number with corresponding feature name as a label
        # Still a 1D array
        # clean_names has the "numeric__" and "categorical__" prefixes removed.  
        # w/o label pairing you would have no idea how to assign values to features
    contributions = pd.Series(shap_values.values[0], index=clean_names)
 
    # Aggregate one-hot encoded columns back to their original source feature
    # (e.g. "is_60_and_over_True"/"is_60_and_over_False" -> one combined
    # "is_60_and_over" total), so the display shows each real feature once.
    
    # fitted_preprocessor.named_transformers_
        # Similar to .named_steps["preprocessor"] let you reach into pipeline and grab a step.  
        # Let's you reach into ColumnTransformer.  
        # Pulls out OneHotEncoder object "categorical"
    cat_transformer = fitted_preprocessor.named_transformers_["categorical"]

    # fitted_preprocessor.named_transformers_[1][2]
        # Reaches into same ColumnTransformer but with positions instead of name.  
        # .transformers_ is a list of three-item groups
        # (name, transformer_object, column_list) - one group per transformer you originally defined
        # [1] grabs second group - "categorical", since"numeric" defined first.  
        # [2] grabs third item within that group
            # orgiinal column names that transformer was actually given
            # categorical_cols after FORCE_CATEGORICAL_COLS adjustment
    categorical_cols_fitted = fitted_preprocessor.transformers_[1][2]
    source_feature_map = {}

    # zip()
        # Takes two lists and pairs them up, position by position
        # 1st item of categorical_cols_fitted paired w/ 1st item in cat_transformer.categories_
        # 2nd w/ 2nd and so forth
    # col
        # zip() pairs position by position, on each pass through loop
        # col is a single plain string - just one column name at each iteration
        # becomes original column name (like "is_60_and_over")     
    # cats 
        # becomes specific column's array of discovered categories (like [False, True])
        # OneHotEncoder already determined True/False values, cats is just reporting values
    # inner loop f"{col}_{cat}"
        # Goes through each individual category value and builds dictionary entry
        # The encoded column name it would have become
        # "is_60_and_over_True" mapped back to original source column "is_60_and_over"
    # source_feature_map 
        # becomes a lookup table like this
        # Problem is OneHotEncoding split features like "is_60_and_over" into "is_60_and_over_True" and "is_60_and_over_False"
        # Conceptually we are only interested in how much did this patiet's 60 status affect risk?  
        # That one answer is split between two rows, therefore need to know following
        # 1) which encoded columns belong together and 2) add their values back up
        # The lookup table (source_feature_map) does step 1 & produces following dictionary key : value pair
        # {"is_60_and_over_True": "is_60_and_over", "is_60_and_over_False": "is_60_and_over", "age_[60-70)": "age", ...}
    for col, cats in zip(categorical_cols_fitted, cat_transformer.categories_):
        for cat in cats:
            source_feature_map[f"{col}_{cat}"] = col

    # Recall contributions is a pd.Series that is a 1D array of values that contain labels of column names
    # groupby()
        # New pandas tool - takes Series (or DataFrame) and bundle rows together into groups
        # based on rules you give it, allows you to do things like sum, average, count by group instead of calling every row
    # lambda name: source_feature_map.get(name, name)
        # This is rule that decides which group each entry belongs to
        # Looks up name in source_feature_map
        # if found, returns mapped source column
        # if not found, falls back to returning name itself, unchanged.  
        # Fall back matters because source_feature_map only contains entries for categorical columns, has nothing on numeric columns
        # numeric columns were never one-hot encoded it first place and don't need grouping
        # For numeric columns .get(name, name) just returns their name, each forms own group of 1
    # .get(name, name)
        # is a dictionary.get(key_to_look_up, value_to_use_if_not_found)
    # .sum()
        # For every group identifed by .groupby(), the SHAP values belonging to that groups get added into a single number
        # At this moment "is_60_and_over_True" and "is_60_and_over_False" individual SHAP values get combined 
        # into one "is_60_and_over" total
        # # Not the 0 or 1, the SHAP value is impact score for every feature.  
        # Answers how much does the fact that this particular category is active-or-not, for this patient, change 
        # the prediction compared to average.
    aggregated = contributions.groupby(lambda name: source_feature_map.get(name, name)).sum()

    #### Sorting & taking top 10 ####
    # aggregated.abs()
        # Takes absolute value of every #, allows us to rank magnitude of impact
    # .sort_values(ascending=False)
        # Sorts values largest to smallest
    # .index
        # Pulls out just feature names, now ordered by importance
        # Discards actual sorted numbers (don't need them yet)
    # aggregated.reindex(...)
        # Reorders aggregated (real, signed values, not absolute ones) to follow same name-ordering
        # Remember aggregated.abs() didn't change aggregated, created brand new temporary series of absolute values
        # Biggest impact features come first while keeping their postive/negative sign intact
    # .head(10)
        # Just keeps the top 10
    top_contributions = aggregated.reindex(aggregated.abs().sort_values(ascending=False).index).head(10)
 
                            #### DISPLAYING THE CHART ####

    with chart_col:
        st.markdown("**Top factors driving this prediction**")
        fig, ax = plt.subplots(figsize=(7, 4.5))

        # for every value in top_contributions, choose red ("#E74C3C") if positive, blue ("#2980B9") if not. 
        # Produces a list of color codes, one per bar, matching the "red = increases risk, blue = decreases risk" 
        # convention mentioned in the axis label.
        colors = ["#E74C3C" if v > 0 else "#2980B9" for v in top_contributions.values]

        # ax.barh()
            # Draws horiz bar chart, barh = "bar, horizontal", ax.bar() would be vertical bars
        # [::-1]
            # Python slicing syntax that reverses a sequence
            # sequence[start:stop:step]
            # start & stop are blank, so entire sequence covered
            # step = -1 means go backwards
            # We are reverse because in matplotlib. horiz bars draw from bottom of chart up
            # Without a reverse #1 would be plotted at bottom.  
        ax.barh(top_contributions.index[::-1], top_contributions.values[::-1], color=colors[::-1])

        # Labels x-axis
        ax.set_xlabel("SHAP value (red = increases risk, blue = decreases risk)")

        # Draws thin vertical reference black line at x=0, helps visualize bars as negative or positive
        ax.axvline(0, color="black", linewidth=0.8)

        # Automatically adjusts spacing so labels/titles don't get cut off or overlap
        plt.tight_layout()

        # Streamlit call that takes fully-built matplotlib figure and renders it on web page.  
        # st.pyplot(fig) 
            # Different Streamlit function than st.bar_chart or other Streamlit-native chart tools
            # Exists to let you draw with matplotlib's full toolkit
            # Allowed top 10, colored, horizontal, signal aware chart
        st.pyplot(fig)
 
    st.caption(
        "SHAP values show how much each factor pushed this specific patient's "
        "predicted risk up or down, relative to the model's average prediction. "
        "This explains THIS prediction, not general feature importance."
    )
 