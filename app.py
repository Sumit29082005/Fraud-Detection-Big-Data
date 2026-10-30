
import json
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Fraud Detection Using Big Data")
st.write("Transaction analysis and fraud detection dashboard")

def load_metrics():
    path = OUTPUT_DIR / "metrics.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def load_spark_csv(folder_name):
    folder = OUTPUT_DIR / folder_name
    if not folder.exists():
        return pd.DataFrame()

    files = list(folder.glob("part-*.csv"))
    if not files:
        return pd.DataFrame()

    frames = [pd.read_csv(file) for file in files]
    return pd.concat(frames, ignore_index=True)

metrics = load_metrics()

if metrics:
    st.subheader("📊 Model Performance")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Accuracy", f"{metrics.get('accuracy', 0) * 100:.2f}%")
    c2.metric("Precision", f"{metrics.get('precision', 0) * 100:.2f}%")
    c3.metric("Recall", f"{metrics.get('recall', 0) * 100:.2f}%")
    c4.metric("F1 Score", f"{metrics.get('f1_score', 0) * 100:.2f}%")

    st.metric("AUC Score", f"{metrics.get('auc', 0):.4f}")

    st.subheader("📌 Confusion Matrix")

    confusion = pd.DataFrame({
        "Result": [
            "True Positive",
            "True Negative",
            "False Positive",
            "False Negative"
        ],
        "Count": [
            metrics.get("true_positive", 0),
            metrics.get("true_negative", 0),
            metrics.get("false_positive", 0),
            metrics.get("false_negative", 0)
        ]
    })

    st.bar_chart(confusion.set_index("Result"))

else:
    st.warning("metrics.json not found. Run the training script first.")

st.divider()

st.subheader("🚨 Fraud Summary")

fraud_summary = load_spark_csv("fraud_summary")

if not fraud_summary.empty:
    st.dataframe(fraud_summary, use_container_width=True)
    st.bar_chart(
        fraud_summary.set_index("is_fraud")["total_transactions"]
    )
else:
    st.info("Fraud summary data is not available yet.")

st.divider()

st.subheader("💳 Transaction Predictions")

predictions = load_spark_csv("predictions")

if not predictions.empty:
    st.write(f"Total predictions: {len(predictions)}")

    search = st.text_input("Search Transaction ID")

    if search:
        predictions = predictions[
            predictions["transaction_id"].astype(str).str.contains(
                search, case=False, na=False
            )
        ]

    st.dataframe(predictions, use_container_width=True)

    if "prediction" in predictions.columns:
        st.write("Prediction distribution")
        st.bar_chart(predictions["prediction"].value_counts())

else:
    st.info("Prediction data is not available yet.")

st.caption("Fraud Detection Using Big Data | Built with PySpark and Streamlit")
