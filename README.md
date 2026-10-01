# Loan Approval Prediction

Predicts whether a loan application will be approved or rejected, using a Random Forest classifier served through a Streamlit app.

## Features

- Trained on applicant financial data: income, loan amount/term, CIBIL score, dependents, education, employment status, and asset values
- 11-feature Random Forest pipeline (StandardScaler + RandomForestClassifier, 200 trees)
- Interactive Streamlit UI — enter applicant details and get an instant Approved/Rejected prediction
- Prediction interface validated end-to-end against the trained pipeline's exact expected feature schema and encoding directions

## Tech Stack

![Python](https://img.shields.io/badge/-Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)

## Project Structure

```
loanapprovalproject.ipynb   # EDA, preprocessing, model training
app1.py                     # Streamlit app — loads loan_pipeline.joblib and serves predictions
loan_approval_dataset.csv   # Training dataset
loan_pipeline.joblib        # Trained StandardScaler + RandomForestClassifier pipeline
```

## Getting Started

```bash
pip install -r requirements.txt
streamlit run app1.py
```

## Input Features

`no_of_dependents`, `education`, `self_employed`, `income_annum`, `loan_amount`, `loan_term`, `cibil_score`, `residential_assets_value`, `commercial_assets_value`, `luxury_assets_value`, `bank_asset_value`
