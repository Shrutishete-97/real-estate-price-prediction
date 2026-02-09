import streamlit as st
import joblib
import os
import numpy as np

# -------------------------------
# Load model safely (no path issues)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "price_model.pkl")

model = joblib.load(MODEL_PATH)

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(
    page_title="Real Estate Price Prediction",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 Real Estate Price Prediction")
st.write("Enter property details to estimate the house price")

# -------------------------------
# User Inputs
# ⚠️ These MUST match training features order
# -------------------------------
area = st.number_input("Area (sq ft)", min_value=100, max_value=10000, value=500)
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=2)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
location_score = st.slider("Location Score (1 = Poor, 5 = Excellent)", 1, 5, 3)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Price"):
    try:
        input_data = np.array([[area, bedrooms, bathrooms, location_score]])
        prediction = model.predict(input_data)

        st.success(f"💰 Estimated House Price: ₹ {prediction[0]:,.2f}")

    except Exception as e:
        st.error("⚠️ Prediction failed")
        st.error(e)

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Built using Machine Learning & Streamlit")

