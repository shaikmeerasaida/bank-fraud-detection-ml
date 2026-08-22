# bank-fraud-detection-ml
AI-powered fraud detection system for digital banks using XGBoost, SHAP explainability, and real-time inference (&lt;100ms). Handles imbalanced data and includes monitoring/drift detection.
# Bank Fraud Detection using Machine Learning

## 🚀 Overview
This project builds an AI-powered fraud detection engine for a mid-sized digital bank, addressing the critical business challenge of real-time transaction fraud detection. The system handles extreme class imbalance, provides explainability via SHAP, and is designed for low-latency inference (<100ms) – meeting PSD2/PCI DSS compliance requirements.

## 🎯 Business Problem
- **Challenge**: Current rule-based systems generate high false positives and miss sophisticated fraud patterns.
- **Impact**: Rising fraud losses and customer frustration.
- **Goal**: Detect fraud in real-time with recall >80% and false positive rate <5%.

## 📊 Dataset
- [Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) – 284,807 transactions, 0.172% fraud.
- Alternative: Synthetic Financial Dataset (included in repo below).

## 🧠 Approach
### Pipeline
1. **Data Preprocessing** – handle missing values, scale numeric features.
2. **Feature Engineering** – create time-based, velocity, and risk-score features.
3. **Modeling** – compare Logistic Regression, Random Forest, XGBoost, Autoencoders.
4. **Imbalance Handling** – SMOTE, class weights, cost-sensitive learning.
5. **Explainability** – SHAP for global and local interpretability.
6. **Deployment** – FastAPI + Docker, Streamlit dashboard, AWS Lambda (optional).

## 📈 Results
| Model | Recall | Precision | F1-score | FPR |
|-------|--------|-----------|----------|-----|
| XGBoost (cost-sensitive) | 0.85 | 0.22 | 0.35 | 0.04 |
| ... | ... | ... | ... | ... |

(Update with your own metrics after training)

## ⚙️ How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/bank-fraud-detection-ml.git
   cd bank-fraud-detection-ml
