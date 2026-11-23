"""
STEP 3 of the project: a small web app (Streamlit) to try the model.

Run:  streamlit run app.py
"""

import os

import joblib
import pandas as pd
import streamlit as st

from train import MODEL_PATH, TYPE_VALUES, build_features

st.set_page_config(page_title="Financial Fraud Detection", page_icon="🛡️")
st.title("🛡️ Financial Fraud Detection")
st.write("Enter the details of a transaction and the model will estimate how likely it is to be fraud.")


@st.cache_resource  # load the model once, not on every click
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


model = load_model()
if model is None:
    st.error("No trained model found. Run `python train.py` first.")
    st.stop()

tab_predict, tab_charts = st.tabs(["Check a transaction", "Data charts"])

# ---------------- Tab 1: prediction ----------------
with tab_predict:
    col1, col2 = st.columns(2)
    with col1:
        tx_type = st.selectbox("Transaction type", TYPE_VALUES, index=4)
        amount = st.number_input("Amount", min_value=0.0, value=181.0)
        step = st.number_input("Hour of simulation (step)", min_value=1, value=1)
    with col2:
        old_org = st.number_input("Sender balance BEFORE", min_value=0.0, value=181.0)
        new_org = st.number_input("Sender balance AFTER", min_value=0.0, value=0.0)
        old_dest = st.number_input("Receiver balance BEFORE", min_value=0.0, value=0.0)
        new_dest = st.number_input("Receiver balance AFTER", min_value=0.0, value=0.0)

    if st.button("Check for fraud", type="primary"):
        row = pd.DataFrame([{
            "step": step, "type": tx_type, "amount": amount,
            "oldbalanceOrg": old_org, "newbalanceOrig": new_org,
            "oldbalanceDest": old_dest, "newbalanceDest": new_dest,
        }])
        # Same feature function used in training -> consistent results
        prob = model.predict_proba(build_features(row))[0, 1]

        st.metric("Fraud probability", f"{prob:.1%}")
        st.progress(min(float(prob), 1.0))
        if prob >= 0.5:
            st.error("⚠️ Likely FRAUD")
        else:
            st.success("✅ Looks legitimate")

# ---------------- Tab 2: the charts made by main.py ----------------
with tab_charts:
    st.write("Charts created by `main.py` (data exploration).")
    charts = {
        "Fraud vs non-fraud": "reports/class_distribution.png",
        "Transaction types": "reports/transactions.png",
        "Fraud by type": "reports/fraud_by_type.png",
        "Amount distribution": "reports/amount_distribution.png",
        "Model comparison": "reports/model_comparison.png",
    }
    for title, path in charts.items():
        if os.path.exists(path):
            st.subheader(title)
            st.image(path)
