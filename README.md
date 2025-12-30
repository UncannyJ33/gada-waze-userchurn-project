[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/UncannyJ33/gada-waze-userchurn-project/main)


# Waze User Churn Analysis & Prediction

This project analyzes user behavior in a synthetic Waze dataset and builds predictive models to estimate monthly user churn. The work progresses from initial data inspection through exploratory data analysis (EDA), statistical testing, regression modeling, and tree‑based machine learning (Random Forest, XGBoost).

The notebooks and results are intended as a portfolio project for the Google Advanced Data Analytics Professional Certificate.

---

## Project structure
```
.
├── data/
│ └── waze_dataset.csv # Waze user‑level dataset (synthetic)
│
├── notebooks/
│ ├── 1_Data_Overview.ipynb # Data types, missingness, first churn insights
│ ├── 2_Exploratory_Data_Analysis.ipynb # Univariate/bivariate EDA and behavioral patterns
│ ├── 3_Statistical_Analysis.ipynb # Two‑sample hypothesis test (iPhone vs Android rides)
│ ├── 4_Logistic_Regression_Modeling.ipynb # Logistic regression churn model and diagnostics
│ └── 5_Tree-based_Machine_Learning.ipynb # Random Forest and XGBoost churn models
│
├── README.md
└── requirements.txt # Project dependencies
```


---

## Data and problem statement

The dataset contains anonymized, user‑level records with monthly app usage and driving behavior, including:

- App activity: `sessions`, `total_sessions`, `activity_days`  
- Driving behavior: `drives`, `driven_km_drives`, `driving_days`, `duration_minutes_drives`  
- Profile/tenure: `n_days_after_onboarding`, `device`  
- Target: `label` indicating whether the user was **retained** or **churned**

**Goal:**  
Predict whether a user will churn within the month and identify behavioral drivers of churn to inform retention strategies (for example, heavy long‑distance drivers vs frequent short‑trip users).

---

## Notebooks overview

### 1 – Data overview ([1_Data_Overview.ipynb](notebooks/1_Data_Overview.ipynb))

**Focus:** verify schema, inspect missing values, and compute initial churn and behavior summaries.

- Confirms 14,999 users with 700 missing churn labels (~5%), all in `label`; missingness appears roughly random  
- Finds a moderately imbalanced target: ~17–18% churn vs 82–83% retention  
- Compares medians for churned vs retained users (`drives`, distance, driving days) and constructs early hypotheses about "super‑drivers" and churn risk

### 2 – Exploratory data analysis ([2_Exploratory_Data_Analysis.ipynb](notebooks/2_Exploratory_Data_Analysis.ipynb))

**Focus:** visual EDA and behavioral segmentation.

- Explores distributions for `sessions`, `drives`, distance, duration, `activity_days`, `driving_days`, and `device`  
- Identifies a high‑intensity segment: users with very long monthly distances/hours (likely professional/long‑haul drivers)  
- Shows churn highest among users with **zero** driving days, lowest among daily drivers  
- Finds churn probability rises with higher `km_per_driving_day`, but higher driving **frequency** correlates with retention  
- Examines recent activity via `percent_sessions_in_last_month`, noting many long‑tenure users suddenly show high recent engagement

### 3 – Statistical analysis ([3_Statistical_Analysis.ipynb](notebooks/3_Statistical_Analysis.ipynb))

**Focus:** two‑sample hypothesis test on ride counts by device.

- Research question: Is there a statistically significant difference in mean drives between iPhone and Android users?  
- Uses descriptive statistics and Welch two‑sample t‑test  
- Finds small observed difference but **fails to reject** null hypothesis at 5% significance—no strong evidence device type drives ride volume

### 4 – Logistic regression modeling ([4_Logistic_Regression_Modeling.ipynb](notebooks/4_Logistic_Regression_Modeling.ipynb))

**Focus:** interpretable baseline churn model.

- Engineers features: `km_per_driving_day`, `km_per_drive`, `professional_driver`, device encoding  
- Handles missing labels, winsorizes extreme outliers at 95th percentile  
- Fits binomial logistic regression; **activity_days** strongest retention predictor  
- Recall limited (~9%), useful explanatory baseline but not high‑recall production model

### 5 – Tree‑based machine learning ([5_Tree-based_Machine_Learning.ipynb](notebooks/5_Tree-based_Machine_Learning.ipynb))

**Focus:** higher‑capacity models and feature importance.

- Reuses engineered features; 60/20/20 stratified train/validation/test split  
- **Random Forest**: GridSearchCV over compact hyperparameter grid  
- **XGBoost**: Broader grid (depth, learning rate, child weight); **recall** primary metric  
- **Results**: XGBoost > Random Forest > Logistic Regression on recall; engineered features dominate importance  
- Confusion matrix shows many churners missed at default threshold—better as decision support

---

## How to run the project

1. **Clone the repository**
```
git clone https://github.com/UncannyJ33/gada-waze-userchurn-project.git
cd waze-user-churn
```

2. **Create and activate a virtual environment (recommended)**
```
python -m venv .venv
source .venv/bin/activate # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```
pip install -r requirements.txt
```

4. **Open the notebooks**
```
jupyter lab
```

Then run the notebooks in order from `1_Data_Overview` to `5_Tree-based_Machine_Learning`.

---

## Dependencies

Add the following to `requirements.txt` (versions optional, include if you want reproducibility):
```
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
xgboost
jupyterlab
```

---



## Key takeaways

- **Churn drivers**: Frequency (`activity_days`) strongest retention signal; high‑intensity long‑distance drivers more likely to churn
- **Professional drivers** (~7.6% churn vs ~20% overall) identified via domain thresholds
- **Logistic regression**: Interpretable baseline, `activity_days` dominant predictor
- **Tree models**: XGBoost highest recall; engineered features (intensity/recency) crucial
- **Model suitability**: Decision support/experimentation baseline, not production-ready (low recall misses many churners)

This project demonstrates end‑to‑end analytics: EDA → statistical testing → interpretable modeling → advanced ML on realistic churn prediction.

