# Streamlit inference app for Tourism Wellness Package prediction
import json
import os
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Tourism Package Predictor", layout="wide")
st.title("Tourism Wellness Package Prediction")
st.write("Provide customer details to estimate purchase probability.")

HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "")
HF_TOKEN = os.getenv("HF_TOKEN", None)

@st.cache_resource
def load_model_and_schema():
    """Load model + schema from HF Hub if configured, otherwise local fallback."""
    model_local = Path("best_model.pkl")
    schema_local = Path("inference_schema.json")

    if HF_MODEL_REPO:
        model_path = hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename="best_model.pkl",
            repo_type="model",
            token=HF_TOKEN,
        )
        schema_path = hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename="inference_schema.json",
            repo_type="model",
            token=HF_TOKEN,
        )
    else:
        model_path = model_local
        schema_path = schema_local

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    return model, schema

model, schema = load_model_and_schema()
feature_columns = schema["feature_columns"]

default_numeric = {
    "Age": 35,
    "CityTier": 1,
    "DurationOfPitch": 12,
    "NumberOfPersonVisiting": 2,
    "NumberOfFollowups": 3,
    "PreferredPropertyStar": 3,
    "NumberOfTrips": 2,
    "Passport": 0,
    "PitchSatisfactionScore": 3,
    "OwnCar": 1,
    "NumberOfChildrenVisiting": 0,
    "MonthlyIncome": 22000,
    "FamilySize": 2,
    "IncomePerTrip": 11000,
}

default_categorical = {
    "TypeofContact": "Self Inquiry",
    "Occupation": "Salaried",
    "Gender": "Male",
    "ProductPitched": "Basic",
    "MaritalStatus": "Single",
    "Designation": "Executive",
}

user_input = {}
cols = st.columns(2)
for i, col_name in enumerate(feature_columns):
    c = cols[i % 2]
    if col_name in default_numeric:
        user_input[col_name] = c.number_input(col_name, value=float(default_numeric[col_name]))
    else:
        user_input[col_name] = c.text_input(col_name, value=str(default_categorical.get(col_name, "Unknown")))

if st.button("Predict Purchase Likelihood", type="primary"):
    input_df = pd.DataFrame([user_input])
    pred = int(model.predict(input_df)[0])
    proba = float(model.predict_proba(input_df)[0, 1])

    st.subheader("Prediction Result")
    st.write(f"Predicted class: **{pred}** (1 = likely to purchase)")
    st.write(f"Purchase probability: **{proba:.2%}**")
    st.progress(min(max(proba, 0.0), 1.0))
