# 💳 Bank Fraud Detection using Machine Learning

A complete end-to-end machine learning project that detects fraudulent credit card transactions using XGBoost, with a Streamlit web interface and a FastAPI backend.

---

## 📌 Project Overview

Credit card fraud is a significant problem for financial institutions. This project develops a robust machine learning system that identifies fraudulent transactions with high recall and low false positive rate. The pipeline includes data preprocessing, handling extreme class imbalance via **SMOTE**, hyperparameter tuning, model explainability with **SHAP**, and deployment-ready web applications.

---

## 🧠 Technologies Used

| Category | Tools & Libraries |
|----------|-------------------|
| **Data Handling** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn, XGBoost, Random Forest, Logistic Regression |
| **Imbalanced Data** | imbalanced-learn (SMOTE) |
| **Explainability** | SHAP |
| **Model Serialization** | Joblib |
| **Deployment** | Streamlit, FastAPI, Uvicorn |
| **Environment** | Python 3.8+ |

---

## 📊 Dataset

The dataset used is the [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset from Kaggle.

- **Number of transactions:** ~28,500 (after cleaning)
- **Features:** 30 anonymized numerical features (V1–V28, Time, Amount)
- **Target:** `Class` (0 = Normal, 1 = Fraud)
- **Imbalance:** Fraud transactions account for only **~0.39%** of the data.

---

## 🔍 Machine Learning Workflow

1. **Data Loading & Exploration** – Understand distributions, correlations, and class imbalance.
2. **Data Cleaning** – Remove duplicates and handle missing values.
3. **Train/Validation/Test Split** – 70/10/20 stratified split.
4. **Class Imbalance Handling** – Apply **SMOTE** (Synthetic Minority Over‑sampling) only on training data.
5. **Model Training** – Evaluate three classifiers:
   - Logistic Regression (baseline)
   - Random Forest
   - **XGBoost** (final model)
6. **Hyperparameter Tuning** – RandomizedSearchCV for XGBoost using PR‑AUC as scoring metric.
7. **Threshold Optimization** – Select decision threshold to balance recall and false positive rate.
8. **Model Evaluation** – Metrics: Accuracy, Precision, Recall, F1, ROC‑AUC, PR‑AUC.
9. **Explainability** – SHAP summary plots and feature importance to interpret model predictions.
10. **Model Saving** – Serialize the final pipeline (scaler + SMOTE + XGBoost) with threshold.
11. **Deployment** – Build interactive Streamlit app and REST API using FastAPI.

---

## 📈 Final Model Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 99.86% |
| **Precision** | 78.95% |
| **Recall (Fraud)** | 88.24% |
| **F1 Score** | 83.33% |
| **ROC‑AUC** | 99.87% |
| **PR‑AUC** | 91.13% |
| **False Positive Rate** | 0.09% |
| **Decision Threshold** | 0.65 |

> The model achieves a high fraud recall (>88%) while keeping false positives extremely low, making it suitable for real‑world deployment.

---

## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/bank-fraud-detection.git
cd bank-fraud-detection
