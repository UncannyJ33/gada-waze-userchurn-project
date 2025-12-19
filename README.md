# Waze User Churn Analysis & Prediction

This project analyzes user behavior in a synthetic Waze dataset and builds predictive models to estimate monthly user churn. The work progresses from initial data inspection through exploratory data analysis (EDA), statistical testing, regression modeling, and tree‑based machine learning (Random Forest, XGBoost).

The notebooks and results are intended as a portfolio project for the Google Advanced Data Analytics Professional Certificate.

---

## Project structure

.
├── data/
│ └── waze_dataset.csv # Waze user‑level dataset (synthetic)
├── notebooks/
│ ├── 01_data_overview.ipynb # Data types, missingness, first churn insights
│ ├── 02_eda.ipynb # Univariate/bivariate EDA and behavioral patterns
│ ├── 03_stats_hypothesis.ipynb
│ │ # Two‑sample hypothesis test (iPhone vs Android rides)
│ ├── 04_regression_modeling.ipynb
│ │ # Logistic regression churn model and diagnostics
│ └── 05_ml_tree_models.ipynb # Random Forest and XGBoost churn models
├── README.md
└── requirements.txt # Project dependencies

## Data and problem statement

The dataset contains anonymized, user‑level records with monthly app usage and driving behavior, including:[web:9][web:63]

- App activity: sessions, total_sessions, activity_days  
- Driving behavior: drives, driven_km_drives, driving_days, duration_minutes_drives  
- Profile/tenure: n_days_after_onboarding, device  
- Target: `label` indicating whether the user was **retained** or **churned**

**Goal:**  
Predict whether a user will churn within the month and identify behavioral drivers of churn to inform retention strategies (for example, heavy long‑distance drivers vs frequent short‑trip users).[web:1][web:85]

---

## Notebooks overview

### 01 – Data overview (`01_data_overview.ipynb`)

Focus: verify schema, inspect missing values, and compute initial churn and behavior summaries.

- Confirms 14,999 users with 700 missing churn labels, all in `label`; missingness appears roughly random.[web:9]  
- Finds a moderately imbalanced target: roughly 17–18% churn vs 82–83% retention.  
- Compares medians for churned vs retained users (drives, distance, driving days) and constructs early hypotheses about “super‑drivers” and churn risk.

### 02 – Exploratory data analysis (`02_eda.ipynb`)

Focus: visual EDA and behavioral segmentation.

- Explores distributions for sessions, drives, distance, duration, activity_days, driving_days, and device.  
- Identifies a high‑intensity segment: users with very long monthly distances and hours driven (likely professional/long‑haul drivers).[web:9]  
- Shows churn is highest among users with **zero** driving days and lowest among those who drive almost every day.  
- Finds that churn probability rises with higher `km_per_driving_day`, while higher driving **frequency** (more driving days) is associated with retention.  
- Examines recent activity via `percent_sessions_in_last_month`, noting many long‑tenure users suddenly show high recent engagement.

### 03 – Statistical analysis (`03_stats_hypothesis.ipynb`)

Focus: two‑sample hypothesis test on ride counts by device.

- Research question: Is there a statistically significant difference in mean drives between iPhone and Android users?  
- Uses descriptive statistics and a Welch two‑sample t‑test to compare mean rides.  
- Finds a small observed difference in average drives but **fails to reject** the null hypothesis at 5% significance—no strong evidence that device type drives ride volume.[web:1]

### 04 – Logistic regression modeling (`04_regression_modeling.ipynb`)

Focus: interpretable baseline churn model.

- Engineers features such as `km_per_driving_day`, `km_per_drive`, `professional_driver`, and device encoding.  
- Handles missing labels and caps extreme outliers for key usage variables.  
- Fits a binomial logistic regression model and inspects coefficients and log‑odds linearity.  
- Finds:
  - **activity_days** is the strongest retention predictor (more active days → lower churn odds).  
  - Very high driving intensity features do not remain as strong once other variables are included.  
  - Recall on churners is limited, making this a useful explanatory baseline but not a high‑recall production model.

### 05 – Tree‑based machine learning (`05_ml_tree_models.ipynb`)

Focus: higher‑capacity models and feature importance.

- Reuses engineered features and encodes `device` and `label` as numeric.  
- Splits data into 60/20/20 train/validation/test sets with stratification.  
- Models:
  - **Random Forest** with GridSearchCV over a compact hyperparameter grid.  
  - **XGBoost** with a broader grid for depth, learning rate, and child weight.  
- Uses **recall** as the primary selection metric due to the cost of missing churners.  
- Results:
  - XGBoost achieves higher recall than both logistic regression and Random Forest, with comparable accuracy and precision.[web:85]  
  - Confusion matrix shows many churners are still missed at the default threshold, so the model is better suited as decision support than as a fully automated system.  
  - Feature importance emphasizes engineered variables (e.g., intensity and recency metrics) alongside core usage signals, reinforcing the value of thoughtful feature engineering.
