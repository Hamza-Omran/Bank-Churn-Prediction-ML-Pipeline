# Bank Customer Churn Prediction - 2025

## Academic Context
This project was developed as part of the "Data Computation Analysis" course (2025).

**Team Members:**

* Ahmad Khalifa Abdulraouf Mohamad – 22010018
* Moaz Moustafa Abd-Elhamid – 22010263
* Abdurhman Hesham Ragab – 22010136
* Kareem Mohamed Samy Aboshady – 22010378
* Hamza Hussain Omran – 22011501

This project builds a complete machine-learning pipeline for predicting customer churn in a bank using a real dataset.
It includes data understanding, cleaning, exploratory analysis, feature selection, dimensionality reduction, SVM modeling, and a Streamlit web application for prediction.

---

## 1. Dataset Summary

The dataset contains demographic, financial, and behavioral information for bank customers, including:
age, gender, dependents, occupation, city, net-worth category, branch code, balances, credits, debits, last transaction date, and the churn label (0/1).

Libraries used include NumPy, Pandas, Matplotlib, Seaborn, Scikit-Learn, Statsmodels, and Streamlit.

---

## 2. Exploratory Data Analysis (Summary)

Main checks performed:

* Dataset shape, types, missing values, descriptive statistics
* Churn distribution (18.5 percent churn rate)
* Outliers and heavy-tailed financial features
* Correlation heatmap
* Gender and occupation churn patterns
* City-level churn distribution
* Churn trend over time
* Age and transaction behavior visualizations

Key observations:

* Weak linear correlation with churn but strong inter-feature correlation in balances
* High churn concentration in specific cities
* Self-employed customers churn more frequently

---

## 3. Data Cleaning

* Imputed missing values
* Converted types (city, dependents, last_transaction)
* Filtered invalid ages
* Encoded categories
* Scaled numerical features

---

## 4. Feature Selection (Summary)

Multiple techniques were applied:

* Chi-Square: branch_code, city, dependents
* ANOVA: debit and credit features
* Mutual Information: current_balance highly informative
* Pearson Correlation: small but positive correlations for debit/credit features
* Forward and Backward AIC: selected strongest predictors

### Final Selected Features

age, dependents, branch_code, current_balance, previous_month_balance,
average_monthly_balance_prevQ, average_monthly_balance_prevQ2,
current_month_credit, previous_month_credit, current_month_debit,
previous_month_debit, current_month_balance.

---

## 5. Dimensionality Reduction

* PCA applied and components selected to retain 90 percent variance
* LDA reduced data to 1 component for visualization and modeling
* Logistic Regression compared across Original, PCA, and LDA data

Original features performed best.

---

## 6. Modeling (SVM Summary)

Models trained:

* Standard SVM
* Tuned SVM with GridSearchCV
* SVM after PCA
* SVM after LDA
* Soft-margin and hard-margin SVM

### Best Model

RBF Kernel SVM

* C = 10
* gamma = scale
* class_weight = balanced

Delivered highest accuracy and balanced performance.

---

## 7. Deployment

The model and scaler were saved as:

* svm_model.pkl
* scaler.pkl

A Streamlit application was created to allow real-time churn prediction from user-entered feature values.

Run the app:

```bash
streamlit run PythonComputation.py
```

---

## 8. Project Files

* Notebook containing the full analysis
* Python scripts for training and deployment
* svm_model.pkl and scaler.pkl
* churn_prediction.csv dataset
* Streamlit web application

---

## 9. Notes

This project is an academic submission summarizing end-to-end churn prediction, including data preparation, feature engineering, model selection, and deployment.
