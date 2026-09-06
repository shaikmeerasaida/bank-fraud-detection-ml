import streamlit as st
import pandas as pd
import numpy as np
import joblib


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Bank Fraud Detection",
    page_icon="💳",
    layout="wide"
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

MODEL_FILE = "fraud_detection_model.pkl"
DATA_FILE = "creditcard.csv"

try:
    artifact = joblib.load(MODEL_FILE)

    model = artifact["model"]
    threshold = artifact["threshold"]
    features = artifact["features"]

except Exception as e:
    st.error("Could not load the trained model.")
    st.code(str(e))
    st.stop()


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

try:
    df = pd.read_csv(DATA_FILE)
except Exception:
    df = None


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("💳 Bank Fraud Detection System")

st.write(
    "Machine Learning based fraud detection using XGBoost."
)

st.info(
    f"Fraud decision threshold: {threshold:.2f}"
)


# --------------------------------------------------
# SAMPLE TRANSACTION BUTTONS
# --------------------------------------------------

st.subheader("🧪 Quick Test")

col1, col2 = st.columns(2)


def load_transaction(row):
    """Load a dataset transaction into the input fields."""

    for feature in features:
        st.session_state[f"input_{feature}"] = float(row[feature])

    st.rerun()


with col1:

    if st.button(
        "✅ Load Legitimate Sample",
        use_container_width=True
    ):

        if df is not None:

            legitimate = df[df["Class"] == 0]

            if len(legitimate) > 0:
                row = legitimate.iloc[0]
                load_transaction(row)

        else:
            st.warning("creditcard.csv was not found.")


with col2:

    if st.button(
        "🚨 Load Fraud Sample",
        use_container_width=True
    ):

        if df is not None:

            fraud = df[df["Class"] == 1]

            if len(fraud) > 0:
                row = fraud.iloc[0]
                load_transaction(row)

        else:
            st.warning("creditcard.csv was not found.")


st.divider()


# --------------------------------------------------
# TRANSACTION INPUTS
# --------------------------------------------------

st.subheader("Enter Transaction Details")

input_data = {}

for feature in features:

    key = f"input_{feature}"

    if key not in st.session_state:
        st.session_state[key] = 0.0

    input_data[feature] = st.number_input(
        feature,
        value=float(st.session_state[key]),
        format="%.6f",
        key=key
    )


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

st.divider()

if st.button(
    "🔍 Check Transaction",
    type="primary",
    use_container_width=True
):

    try:

        # Create dataframe in the exact feature order
        input_df = pd.DataFrame(
            [input_data],
            columns=features
        )

        # Get fraud probability
        fraud_probability = model.predict_proba(
            input_df
        )[0][1]

        # Apply optimized threshold
        prediction = (
            fraud_probability >= threshold
        )

        st.subheader("Prediction Result")

        # ------------------------------------------
        # FRAUD RESULT
        # ------------------------------------------

        if prediction:

            st.error(
                "🚨 FRAUDULENT TRANSACTION DETECTED"
            )

            st.metric(
                "Fraud Probability",
                f"{fraud_probability * 100:.2f}%"
            )

            st.warning(
                "This transaction should be reviewed."
            )

        # ------------------------------------------
        # LEGITIMATE RESULT
        # ------------------------------------------

        else:

            st.success(
                "✅ LEGITIMATE TRANSACTION"
            )

            st.metric(
                "Fraud Probability",
                f"{fraud_probability * 100:.2f}%"
            )

            st.info(
                "The transaction is below the fraud decision threshold."
            )

        # ------------------------------------------
        # ADDITIONAL INFORMATION
        # ------------------------------------------

        st.write(
            f"Decision threshold: "
            f"{threshold:.2f}"
        )

        st.write(
            f"Model probability: "
            f"{fraud_probability:.4f}"
        )

    except Exception as e:

        st.error("Prediction failed.")

        st.code(str(e))


# --------------------------------------------------
# PROJECT INFORMATION
# --------------------------------------------------

st.divider()

st.subheader("📊 About This Project")

st.write(
    """
    This project uses machine learning to identify potentially
    fraudulent credit card transactions.

    The trained XGBoost model handles the highly imbalanced
    fraud detection problem and uses an optimized decision
    threshold to make the final prediction.
    """
)

st.caption(
    "Bank Fraud Detection | Machine Learning | XGBoost"
)