# Loan Approval — Supervised Classification

A machine learning project that predicts loan approval decisions using supervised 
classification. Built as part of a Data Science Master's program, this project 
demonstrates a full ML pipeline — from exploratory data analysis and feature 
engineering to model evaluation and threshold optimisation.

Created for learning purposes. No claim of production-ready performance is made.

---

## What the Notebook Covers

- **Data Cleaning** — duplicate removal, dtype fixing, binary encoding, 
  one-hot encoding, missing data imputation
- **Exploratory Data Analysis (EDA)** — pairplots, boxplots, linear regression, 
  chi-square tests, t-tests, correlation matrix
- **Feature Engineering** — combined income, loan-to-income ratio
- **Preprocessing** — outlier capping, missing data imputation, standard scaling
- **Modelling** — Logistic Regression and Random Forest with GridSearchCV hyperparameter tuning
- **Evaluation** — F1-score, Recall, Precision, ROC/AUC, generalisation analysis (train vs. test performance), feature importance
- **Model Selection** — justified decision based on generalisation ability and 
  business objective: F1(0) as primary metric, prioritising the identification 
  of bad borrowers to minimise credit default risk

---

## Dataset

- Source: Provided as part of the MasterSchool Data Science curriculum
- ~600 observations, 13 features
- Binary target variable: `loan_status` (1 = approved, 0 = rejected)

---

## Key Results

- **Selected model: Logistic Regression** — despite marginally lower Test F1(0) and 
  Recall than Random Forest, chosen for its stable generalisation (no overfitting)
- **Random Forest: severe overfitting** — Train F1(0) = 100%, Test F1(0) = 59.6%
- **Logistic Regression: stable** — Train F1(0) = 52.8%, Test F1(0) = 59.1%
- **Dominant predictor: `credit_history`** — Odds Ratio = 20.6, Cramér's V = 0.49
- **Optimal classification threshold: 0.55** (marginal improvement over default 0.50)

---

## Project Structure

```
loan-approval-supervised-classification/
├── Data/
│   └── loans_modified.csv
├── loan_approval_ml_pipeline.ipynb
├── LICENSE
└── README.md
```

---

## Known Limitations

**(1) Threshold Optimisation**
- Threshold was optimised and evaluated on the same test set
- A separate validation set would be methodologically cleaner (e.g. 80/10/10 split)
- With ~500 observations, a three-way split is problematic — results should be 
  interpreted with caution

**(2) Dataset Size**
- ~500 observations after cleaning — too small for stable hyperparameter tuning
- GridSearchCV found no improvement over default parameters for either model
- `credit_history` dominates to such an extent that all other features have 
  negligible influence — with a larger, richer dataset, feature engineering 
  would likely yield more meaningful results; more sophisticated strategies 
  such as income-based segmentation with segment-specific models could 
  potentially uncover more nuanced patterns
---

## Requirements

Python 3.11
Install dependencies:
pip install -r requirements.txt

---

## Author

Matthias Muschket
Developed as part of a Data Science Master's program at MasterSchool