# Credit Card Approval Prediction

A machine learning web application that predicts whether a credit card application would be **approved or rejected**, based on a full applicant profile — demographics, household, housing, employment, and income. Built as part of the SmartBridge / SkillWallet program.

---

## 📌 Project Overview

Credit card issuers need a fast, consistent way to gauge whether an applicant is likely to be approved before manual underwriting. This project builds that estimator end-to-end:

- Collects and explores real-world applicant data
- Cleans, engineers, and encodes the data for modeling
- Trains and compares multiple classification models
- Serves the best-performing model through a Flask web app with a live, styled UI

> **Note on labels:** The source `application_record.csv` dataset does not contain a genuine "approved/rejected" column, and the accompanying `creditcard.csv` (an anonymized fraud-transaction dataset) does not share an ID space with it. This project builds a clearly-documented **synthetic approval label** so the models have a real two-class target to learn from. `creditcard.csv` is used only for general exploratory analysis. Swap in a real approval column if/when one is available.

---

## 🧱 Project Plan (Epics)

| Epic | Description |
|------|-------------|
| **Epic 1** | Data Collection |
| **Epic 2** | Visualizing and Analysing the Data |
| **Epic 3** | Data Pre-processing |
| **Epic 4** | Model Building |
| **Epic 5** | Application Building |

---

## 📂 Repository Structure

```
Credit-card-approval-SmartBridge/
├── 1. Brainstorming & Ideation/
│   └── Define Problem Statements.docx
├── 2. Requirement Analysis/
│   ├── Customer Journey Map.docx
│   ├── Data Flow Diagram.docx
│   ├── Solution Requirements.docx
│   └── Technology Stack.docx
├── 3. Project Design Phase/
│   ├── Problem-Solution Fit.docx
│   ├── Proposed Solution.docx
│   └── Solution Architecture.docx
├── 4. Project Planning Phase/
│   └── Project Planning.docx
├── 5. Project Development Phase/
│   ├── Code-Layout, Readability and Reusability.docx
│   ├── Coding & Solution.docx
│   └── No. of Functional Features Included in the Solution.docx
├── 6. Project Testing/
│   └── Performance Testing.docx
├── 7. Project Documentation/           # The application itself lives here, alongside its writeup
│   ├── templates/                       # Flask HTML templates
│   │   ├── index.html                    # Full application form + live card preview + custom validation
│   │   └── result.html                   # Prediction result page with a realistic card face
│   ├── .gitignore
│   ├── app.py                           # Flask application (inference + routing)
│   ├── application_record.csv           # Applicant demographic/financial dataset
│   ├── creditcard.csv                   # Anonymized fraud-transaction dataset (supplementary EDA only)
│   ├── ER_Diagram.png                   # Entity-relationship diagram for the data
│   ├── locust_file.py                   # Load testing script (Locust)
│   ├── main.ipynb                       # EDA, preprocessing, model training notebook
│   ├── model.pkl                        # Bundled trained models + encoders + feature order
│   ├── Project Executable Files.docx
│   └── Project_Documentation.docx
├── 8. Project Demonstration/
│   ├── Communication.docx
│   ├── Demo_Video_link.txt
│   ├── Demonstration of Proposed Features.docx
│   ├── Project Demo Planning.docx
│   ├── Scalability & Future Plan.docx
│   └── Team Involvement in Demonstration.docx
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## 📊 Dataset

- **`application_record.csv`** — 438,557 rows × 18 columns. Applicant demographic and financial fields including `CODE_GENDER`, `FLAG_OWN_CAR`, `FLAG_OWN_REALTY`, `CNT_CHILDREN`, `AMT_INCOME_TOTAL`, `NAME_INCOME_TYPE`, `NAME_EDUCATION_TYPE`, `NAME_FAMILY_STATUS`, `NAME_HOUSING_TYPE`, `OCCUPATION_TYPE`, `CNT_FAM_MEMBERS`, `DAYS_BIRTH`, `DAYS_EMPLOYED`, and more.
- **`creditcard.csv`** — 284,807 rows × 31 columns. An unrelated, anonymized fraud-detection dataset (`Time`, `V1`...`V28`, `Amount`, `Class`), used only for supplementary EDA.

---

## ⚙️ Tech Stack

- **Language:** Python 3.13
- **Data & ML:** pandas, numpy, scikit-learn (`LabelEncoder`, `train_test_split`, `LogisticRegression`, `RandomForestClassifier`, `DecisionTreeClassifier`)
- **Visualization:** matplotlib, seaborn
- **Web Framework:** Flask
- **Load Testing:** Locust
- **Frontend:** HTML5, CSS (Google Fonts — Fraunces, Inter, IBM Plex Mono), vanilla JavaScript

---

## 📥 Epic 1 — Data Collection

- **Download the Dataset** — Sourced `application_record.csv` (applicant demographic/financial records) and `creditcard.csv` (anonymized fraud-transaction data) for exploratory use.

## 🔍 Epic 2 — Visualizing and Analysing the Data

- **Importing the Libraries** — pandas, numpy, matplotlib, seaborn, scikit-learn modules.
- **Read the Dataset** — Loaded both CSVs into pandas DataFrames and inspected shapes/structure (`application_record.csv`: 438,557 × 18; `creditcard.csv`: 284,807 × 31).
- **Univariate Analysis** — Examined individual feature distributions (e.g. income, education type, family status, housing type) in isolation.
- **Multivariate Analysis** — Explored relationships and correlations between multiple features (e.g. income type vs. income amount, education vs. approval-related patterns).
- **Descriptive Analysis** — Computed summary statistics (mean, median, spread, category frequencies) to understand the overall dataset profile.

## 🧹 Epic 3 — Data Pre-processing

- **Drop Duplicate Features** — Removed redundant/duplicate columns and rows not needed for modeling.
- **Handling Missing Values** — Addressed nulls (e.g. missing `OCCUPATION_TYPE` entries, filled as `"Unknown"`) via imputation or exclusion as appropriate.
- **Data Cleaning and Merging** — Cleaned inconsistent entries and consolidated the relevant applicant fields into a single working DataFrame.
- **Feature Engineering** —
  - Derived `AGE_YEARS` from `DAYS_BIRTH`.
  - Derived `YEARS_EMPLOYED` from `DAYS_EMPLOYED`, correcting the dataset's documented sentinel value (`DAYS_EMPLOYED > 0`, used for pensioners / not-currently-employed applicants) to `0` years rather than the ~1,000-year artifact it would otherwise produce.
  - Engineered the synthetic `APPROVED` target column (since no real approval label exists in the source data) from a weighted score across income, income type, education, car/property ownership, age, years employed, marital stability, housing stability, and number of children, then flipped ~12% of labels at random so the pattern isn't a perfectly clean lookup table.
  - Selected the final model features (13 total): `GENDER_ENC`, `OWN_CAR_ENC`, `OWN_REALTY_ENC`, `CNT_CHILDREN`, `INCOME_TYPE_ENC`, `EDUCATION_TYPE_ENC`, `FAMILY_STATUS_ENC`, `HOUSING_TYPE_ENC`, `OCCUPATION_TYPE_ENC`, `CNT_FAM_MEMBERS`, `AGE_YEARS`, `YEARS_EMPLOYED`, `AMT_INCOME_TOTAL`.
- **Handling Categorical Values** — Label-encoded every categorical feature, one `LabelEncoder` per field:
  - **Gender** (2 classes): `F`, `M`
  - **Own car** (2 classes): `N`, `Y`
  - **Own realty** (2 classes): `N`, `Y`
  - **Income type** (5 classes): `Commercial associate`, `Pensioner`, `State servant`, `Student`, `Working`
  - **Education type** (5 classes): `Academic degree`, `Higher education`, `Incomplete higher`, `Lower secondary`, `Secondary / secondary special`
  - **Family status** (5 classes): `Civil marriage`, `Married`, `Separated`, `Single / not married`, `Widow`
  - **Housing type** (6 classes): `Co-op apartment`, `House / apartment`, `Municipal apartment`, `Office apartment`, `Rented apartment`, `With parents`
  - **Occupation type** (19 classes): the dataset's 18 occupation categories plus an `Unknown` fallback for missing entries

Data was then split 80/20, stratified on the target, for training and evaluation.

## 🤖 Epic 4 — Model Building

Three classifiers were trained and evaluated on the same train/test split, now over the full 13-feature set:

- **Logistic Regression Model**
- **Random Forest Model** (100 estimators)
- **Decision Tree Model**

> Accuracy figures depend on the trained `model.pkl`. The original 3-feature version of this project (income type, education, income only) scored Logistic Regression 77.8%, Random Forest 88.2% (best), Decision Tree 88.1%. Re-run `main.ipynb` top to bottom to regenerate `model.pkl` on the expanded 13-feature set and get current numbers for all three models.

All three models, one `LabelEncoder` per categorical feature, and the exact feature order used at training time are bundled together into `model.pkl` so the Flask app can load everything in one step at inference time.

## 🌐 Epic 5 — Application Building

- **Building HTML Pages**
  - `index.html` — Full application form, organized into three sections:
    - *Personal details* — full name, gender, age, marital status, family members, number of children
    - *Housing & assets* — car ownership, property ownership, housing type
    - *Employment & finances* — income type, occupation, education, years employed, annual income
    A live, styled credit card preview and a summary panel update as the user types. Native browser validation is disabled (`novalidate`); on submit, JavaScript checks every required field and, if any are empty, blocks submission, outlines the incomplete fields in red, and shows a banner listing exactly which fields still need to be filled in.
  - `result.html` — Displays the prediction outcome (Approved/Declined) with an animated stamp, a receipt-style summary of the submitted details, and a realistic card face: chip, contactless icon, embossed card number, a "valid thru" expiry derived from the card number, and a Visa- or Mastercard-style network mark chosen from the card number's leading digit. Declined cards render slightly desaturated.
- **Build the Python Script**
  - `app.py` loads the `model.pkl` bundle (models + all encoders + feature order) and serves `/` (home form) and `/predict` (POST) routes.
  - Every submitted categorical field is encoded via its saved `LabelEncoder`, falling back gracefully to the most common training category if an unseen value is submitted; numeric fields fall back to sane defaults if missing.
  - The feature vector is built generically from `feature_order`, so the app always matches whatever feature set the notebook was last trained on.
  - The active model is configurable via the `MODEL_NAME` environment variable (`logistic_regression`, `random_forest`, or `decision_tree`); it defaults to `random_forest`, the best-performing model.
  - Approved applicants are issued a deterministic, per-applicant masked card number that passes a genuine Luhn checksum (`generate_card_number`), giving the demo a realistic feel without generating a real card number.
- **Run the Application** — Launched locally via `python app.py`, served by Flask's development server.

---

## 🧠 How a Prediction Is Made

1. The form collects a full applicant profile (13 fields).
2. `app.py` encodes every categorical field (income type, education, gender, car/property ownership, marital status, housing type, occupation) into the numeric codes the model was trained on, using the saved `LabelEncoder`s.
3. All fields are assembled into one feature vector, in the exact column order (`feature_order`) stored in `model.pkl`.
4. The selected classifier predicts `1` (approved) or `0` (rejected) for that vector.
5. Since the source dataset has no real approval outcome, the model was trained on a **synthetic, rule-based label** (see Epic 3) rather than actual bank decisions — so results are illustrative of a genuine ML pipeline, not a real underwriting engine.

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.10+
```

### Installation

```bash
git clone https://github.com/Divisukumar3010/Credit-card-approval-SmartBridge.git
cd Credit-card-approval-SmartBridge
pip install -r requirements.txt
cd "7. Project Documentation"
```

### Train the Model (required after pulling the expanded feature set)

From inside `7. Project Documentation/`, run `main.ipynb` from top to bottom. This regenerates `model.pkl` as a bundle containing:
- `models` — dict of the three trained classifiers
- `gender_encoder`, `own_car_encoder`, `own_realty_encoder`, `income_type_encoder`, `education_type_encoder`, `family_status_encoder`, `housing_type_encoder`, `occupation_type_encoder` — fitted `LabelEncoder`s, one per categorical field
- `feature_order` — the exact 13-column order used at training time

If `model.pkl` is still in the older 3-feature format, `app.py` will raise a clear `RuntimeError` on startup telling you to re-run the notebook.

### Run the App

From inside `7. Project Documentation/`:

```bash
python app.py
```

By default this starts the Flask development server at `http://127.0.0.1:5000/`.

To choose a specific model at inference time:

```bash
# macOS / Linux
MODEL_NAME=logistic_regression python app.py

# Windows (PowerShell)
$env:MODEL_NAME="decision_tree"; python app.py
```

### Load Testing

From inside `7. Project Documentation/`:

```bash
locust -f locust_file.py
```

Then open the Locust web UI (default `http://localhost:8089`) to configure and run a load test against the running Flask app.

---

## 🖥️ Usage

1. Open the app in your browser.
2. Fill in your personal details, housing & assets, and employment & finances — the card preview updates live.
3. Click **Check approval**. If any field is missing, the form highlights it and lists what's incomplete instead of submitting.
4. View the result: an approved application shows a generated masked card number with a realistic card face (network mark, expiry, chip); a declined application shows the reason and submitted details.

---

## ⚠️ Limitations & Disclaimers

- This is a **demo/educational model** — while it now uses a much fuller applicant profile than a bare-bones version would, it is still **not a real credit decision engine**.
- The approval label used for training is **synthetic**, since the source dataset has no real approval outcome.
- Generated card numbers, expiry dates, and network marks are cosmetic (Luhn-valid for realism) and are **not real, usable card numbers**.

---

## 👤 Authors

**Divi Sukumar & Suroju Teja**

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.