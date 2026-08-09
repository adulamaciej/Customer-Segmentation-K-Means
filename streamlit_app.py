import streamlit as st
import pandas as pd
from datetime import date
from src.inference import predict_segment, load_artifacts

st.set_page_config(page_title="Customer Segment Predictor", page_icon="🎯")
st.title("Customer Segment Predictor")
st.write("Enter a customer's purchase history to predict their RFM segment.")

artifacts = load_artifacts()

TRAINING_MIN_DATE = date(2009, 12, 1)
TRAINING_MAX_DATE = date(2011, 12, 9)  # model's reference_date

col1, col2 = st.columns(2)
with col1:
    first_purchase = st.date_input(
        "First purchase date",
        value=date(2011, 1, 1),
        min_value=TRAINING_MIN_DATE,
        max_value=TRAINING_MAX_DATE,
    )
    monetary_value = st.number_input("Total spend (£)", min_value=0.0, value=500.0)
with col2:
    last_purchase = st.date_input(
        "Last purchase date",
        value=date(2011, 11, 1),
        min_value=TRAINING_MIN_DATE,
        max_value=TRAINING_MAX_DATE,
    )
    frequency = st.number_input("Number of orders", min_value=1, value=3)

aov = monetary_value / frequency if frequency else 0
st.metric("Average Order Value (AOV)", f"£{aov:.2f}")

st.caption(
    f"Dates are limited to the model's training window "
    f"({TRAINING_MIN_DATE} to {TRAINING_MAX_DATE}) — predictions outside "
    f"this range are unreliable extrapolations."
)

if st.button("Predict Segment"):
    if first_purchase > last_purchase:
        st.error("First purchase date must be before last purchase date.")
        st.stop()

    customer_df = pd.DataFrame([{
        "last_purchase": pd.Timestamp(last_purchase),
        "first_purchase": pd.Timestamp(first_purchase),
        "MonetaryValue": monetary_value,
        "Frequency": frequency,
        "AOV": aov,
    }])

    result = predict_segment(customer_df, artifacts=artifacts)
    row = result.iloc[0]

    st.success(f"Segment: **{row['Segment']}**")
    if row["IsWhale"]:
        st.info("🐋 Flagged as a whale (B2B/wholesale outlier)")
    elif row["IsOutlier"]:
        st.info("⚠️ Flagged as a statistical outlier")