# 💳 Bank Fraud Detection using Machine Learning

## 📌 Project Overview

This project develops an end-to-end machine learning system to identify
fraudulent credit card transactions.

The project compares multiple machine learning algorithms and uses
SMOTE for handling class imbalance. XGBoost is used for fraud
classification, with SHAP explainability to understand model
predictions.

The project also includes:

- Streamlit web interface for interactive fraud prediction
- FastAPI backend for API-based predictions
- Model evaluation and threshold optimization
- SHAP-based model explainability

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- SMOTE
- Logistic Regression
- Random Forest
- XGBoost
- SHAP
- Streamlit
- FastAPI

---

## 🔄 Machine Learning Workflow

1. Data Loading
2. Exploratory Data Analysis
3. Data Preprocessing
4. Train / Validation / Test Split
5. Handling Class Imbalance using SMOTE
6. Logistic Regression
7. Random Forest
8. XGBoost
9. Hyperparameter Tuning
10. Model Evaluation
11. Threshold Optimization
12. SHAP Explainability
13. Model Saving
14. Streamlit Deployment
15. FastAPI Deployment

---

## 📊 Dataset

This project uses the **Credit Card Fraud Detection** dataset.

The target variable is:

- `0` = Normal transaction
- `1` = Fraudulent transaction

### Dataset Source

The dataset was obtained from Kaggle:

**Credit Card Fraud Detection Dataset by iabhishekofficial**

Kaggle Dataset:
`https://www.kaggle.com/datasets/iabhishekofficial/creditcard-fraud-detection`

### Dataset Availability

The original `creditcard.csv` file is approximately **143 MB**, which
exceeds GitHub's 100 MB file size limit.

Therefore, the complete dataset is **not included in this GitHub
repository**.

To run the project locally:

1. Download the dataset from the Kaggle link above.
2. Extract the dataset.
3. Place `creditcard.csv` inside the project folder.
4. Run the notebook or application.

---

## 📈 Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC
- False Positive Rate

Since fraud detection is an imbalanced classification problem, special
attention is given to **Precision, Recall, F1 Score, PR-AUC, and False
Positive Rate**.

---

## 🔍 Explainability

**SHAP (SHapley Additive exPlanations)** is used to understand which
features have the greatest influence on fraud predictions.

This helps make the machine learning model more interpretable.

---

## 🌐 Application

### Streamlit

A Streamlit web interface is provided for interactive fraud prediction.

Users can enter transaction information and receive a prediction from
the trained machine learning model.

### FastAPI

A FastAPI backend is included for API-based fraud prediction.

The API can be used to send transaction data and receive the model's
prediction programmatically.

---

## 📁 Project Structure

```text
bank-fraud-detection-ml/
│
├── api.py
├── app.py
├── fraud_detection_model.pkl
├── Bank_Fraud_Detection_ML (1).ipynb
├── README.md
├── requirements.txt
├── LICENSE
└── creditcard.csv          # Download separately from Kaggle
