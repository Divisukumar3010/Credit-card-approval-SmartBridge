# Credit Card Approval Prediction

A machine learning web application that predicts whether a credit card application would be **approved or rejected**, based on applicant details such as income type, education, and annual income. Built as part of the SmartBridge / SkillWallet program.

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
├── Project Documentation/     # Reports, proposal docs, sprint/demo sheets
├── templates/                 # Flask HTML templates
│   ├── index.html              # Application form + live card preview
│   └── result.html             # Prediction result page
├── app.py                     # Flask application (inference + routing)
├── main.ipynb                 # EDA, preprocessing, model training notebook
├── model.pkl                  # Bundled trained models + encoders + feature order
├── locust_file.py             # Load testing script (Locust)
├── ER_Diagram.png             # Entity-relationship diagram for the data
└── .gitignore
```

---

## 📊 Dataset

- **`application_record.csv`** — 438,557 rows × 18 columns. Applicant demographic and financial fields including `CODE_GENDER`, `FLAG_OWN_CAR`, `FLAG_OWN_REALTY`, `CNT_CHILDREN`, `AMT_INCOME_TOTAL`, `NAME_INCOME_TYPE`, `NAME_EDUCATION_TYPE`, `NAME_FAMILY_STATUS`, `NAME_HOUSING_TYPE`, `DAYS_BIRTH`, `DAYS_EMPLOYED`, and more.
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
- **Handling Missing Values** — Addressed nulls (e.g. missing `OCCUPATION_TYPE` entries) via imputation or exclusion as appropriate.
- **Data Cleaning and Merging** — Cleaned inconsistent entries and consolidated the relevant applicant fields into a single working DataFrame.
- **Feature Engineering** — Engineered the synthetic `APPROVED` target column (since no real approval label exists in the source data) and selected the final model features: `INCOME_TYPE_ENC`, `EDUCATION_TYPE_ENC`, `AMT_INCOME_TOTAL`.
- **Handling Categorical Values** — Label-encoded categorical features:
  - **Income type** (5 classes): `Commercial associate`, `Pensioner`, `State servant`, `Student`, `Working`
  - **Education type** (5 classes): `Academic degree`, `Higher education`, `Incomplete higher`, `Lower secondary`, `Secondary / secondary special`

Data was then split: 71,999 training rows / 18,000 test rows (80/20 stratified split).

## 🤖 Epic 4 — Model Building

Three classifiers were trained and evaluated on the same train/test split:

- **Logistic Regression Model**
- **Random Forest Model** (100 estimators)
- **Decision Tree Model**

| Model | Accuracy |
|-------|----------|
| Logistic Regression | **77.8%** |
| Random Forest | **88.2%** ✅ Best |
| Decision Tree | **88.1%** |

All three models, both label encoders, and the exact feature order used at training time are bundled together into `model.pkl` so the Flask app can load everything in one step at inference time.

## 🌐 Epic 5 — Application Building

- **Building HTML Pages**
  - `index.html` — Application form (name, income type, education, annual income) with a live, styled credit card preview that updates as the user types.
  - `result.html` — Displays the prediction outcome (Approved/Declined) with an animated stamp, the generated card preview, and a receipt-style summary of the submitted details.
- **Build the Python Script**
  - `app.py` loads the `model.pkl` bundle and serves `/` (home form) and `/predict` (POST) routes.
  - Submitted categorical fields are encoded via the saved encoders, falling back gracefully to the most common training category if an unseen value is submitted.
  - The feature vector is built in the exact order used at training time and passed to the selected model.
  - The active model is configurable via the `MODEL_NAME` environment variable (`logistic_regression`, `random_forest`, or `decision_tree`); it defaults to `random_forest`, the best-performing model.
  - Approved applicants are issued a deterministic, per-applicant masked card number that passes a genuine Luhn checksum (`generate_card_number`), giving the demo a realistic feel without generating a real card number.
- **Run the Application** — Launched locally via `python app.py`, served by Flask's development server.

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
pip install flask pandas numpy scikit-learn matplotlib seaborn locust
```

### Train the Model (optional — `model.pkl` is already included)

Run `main.ipynb` from top to bottom. This regenerates `model.pkl` as a bundle containing:
- `models` — dict of the three trained classifiers
- `income_type_encoder`, `education_type_encoder` — fitted `LabelEncoder`s
- `feature_order` — the exact column order used at training time

### Run the App

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

```bash
locust -f locust_file.py
```

Then open the Locust web UI (default `http://localhost:8089`) to configure and run a load test against the running Flask app.

---

## 🖥️ Usage

1. Open the app in your browser.
2. Fill in your name, income type, education, and annual income — the card preview updates live.
3. Click **Check approval**.
4. View the result: an approved application shows a generated masked card number; a declined application shows the reason and submitted details.

---

## ⚠️ Limitations & Disclaimers

- This is a **demo/educational model** trained on a small feature set (income type, education, annual income only) — it is **not a real credit decision engine**.
- The approval label used for training is **synthetic**, since the source dataset has no real approval outcome.
- Generated card numbers are cosmetic (Luhn-valid for realism) and are **not real, usable card numbers**.

---

## 👤 Authors

**Divi Sukumar & Suroju Teja**

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.