"""Streamlit inference app for Tourism package purchase prediction."""

import os
import json
import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Tourism Purchase Predictor", layout="wide")
st.title("Tourism Package Purchase Prediction")
st.write("This app loads the best model from Hugging Face Model Hub and predicts purchase likelihood.")

# Non-secret app configuration is defined in code (not in .env).
HF_MODEL_REPO = "tourism-best-model"
HF_USERNAME = os.getenv("HF_USERNAME", "vinnie16")
REPO_ID = f"{HF_USERNAME}/{HF_MODEL_REPO}"

@st.cache_resource
def load_model_and_metadata():
    """Download and load model + metadata from Hugging Face Model Hub."""
    model_file = hf_hub_download(repo_id=REPO_ID, filename="best_model.joblib", repo_type="model")
    metadata_file = hf_hub_download(repo_id=REPO_ID, filename="model_metadata.json", repo_type="model")
    model = joblib.load(model_file)
    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return model, metadata

try:
    model, metadata = load_model_and_metadata()
    st.success(f"Loaded model from: {REPO_ID}")
except Exception as exc:
    st.error(f"Model loading failed: {exc}")
    st.stop()

st.subheader("Input Features")
st.caption("Provide customer details and click Predict.")

# Build a simple form with assignment-relevant features.
age = st.number_input("Age", min_value=18, max_value=80, value=35)
type_of_contact = st.selectbox("TypeofContact", ["Self Enquiry", "Company Invited"] )
city_tier = st.selectbox("CityTier", [1, 2, 3])
duration_of_pitch = st.number_input("DurationOfPitch", min_value=1, max_value=60, value=15)
occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"] )
gender = st.selectbox("Gender", ["Male", "Female"] )
num_person_visiting = st.number_input("NumberOfPersonVisiting", min_value=1, max_value=10, value=2)
num_followups = st.number_input("NumberOfFollowups", min_value=0, max_value=10, value=3)
product_pitched = st.selectbox("ProductPitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"] )
preferred_star = st.selectbox("PreferredPropertyStar", [3, 4, 5])
marital_status = st.selectbox("MaritalStatus", ["Single", "Married", "Divorced", "Unmarried"] )
num_trips = st.number_input("NumberOfTrips", min_value=0, max_value=15, value=2)
passport = st.selectbox("Passport", [0, 1])
pitch_satisfaction = st.number_input("PitchSatisfactionScore", min_value=1, max_value=5, value=3)
own_car = st.selectbox("OwnCar", [0, 1])
num_children = st.number_input("NumberOfChildrenVisiting", min_value=0, max_value=5, value=0)
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"] )
monthly_income = st.number_input("MonthlyIncome", min_value=1000, max_value=1000000, value=25000)

if st.button("Predict"):
    input_df = pd.DataFrame([{
        "Age": age,
        "TypeofContact": type_of_contact,
        "CityTier": city_tier,
        "DurationOfPitch": duration_of_pitch,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": num_person_visiting,
        "NumberOfFollowups": num_followups,
        "ProductPitched": product_pitched,
        "PreferredPropertyStar": preferred_star,
        "MaritalStatus": marital_status,
        "NumberOfTrips": num_trips,
        "Passport": passport,
        "PitchSatisfactionScore": pitch_satisfaction,
        "OwnCar": own_car,
        "NumberOfChildrenVisiting": num_children,
        "Designation": designation,
        "MonthlyIncome": monthly_income,
    }])

    # Save inputs into a dataframe file for traceability.
    os.makedirs("inputs", exist_ok=True)
    input_df.to_csv("inputs/latest_inputs.csv", index=False)

    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0][1]) if hasattr(model, "predict_proba") else None

    st.subheader("Prediction Result")
    st.write("Predicted class (1 = likely to buy):", prediction)
    if probability is not None:
        st.write("Purchase probability:", f"{probability:.4f}")

st.subheader("Model Details")
st.json(metadata)
