"""
Heart Disease Prediction System - Streamlit Web App
----------------------------------------------------
Student: Fatima Farzad (SP24-BSE-025) - AIC354 Assignment 2, Task 2

Loads the Support Vector Classifier trained in
Task2-Heart-Disease-Prediction/Heart_Disease_Prediction_System.ipynb and recreates
the exact same LabelEncoders (fit on the same fixed category lists), so the
predictions match the notebook exactly.
"""

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

MODEL_PATH = Path(__file__).parent / "heart_svc_trained_model.pkl"

CHEST_PAIN = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"]
EXERCISE_ANGINA = ["Yes", "No"]
VESSELS = ["Zero", "One", "Two", "Three"]
THAL = ["Normal", "Fixed Defect", "Reversible Defect"]


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def build_label_encoders():
    """Same encoders as Step 5.1 of the notebook (LabelEncoder sorts the
    categories, so fitting on the same lists gives the same codes)."""
    return {
        "ChestPain": LabelEncoder().fit(CHEST_PAIN),
        "ExerciseAngina": LabelEncoder().fit(EXERCISE_ANGINA),
        "Vessels": LabelEncoder().fit(VESSELS),
        "Thal": LabelEncoder().fit(THAL),
    }


model = load_model()
encoders = build_label_encoders()

st.title("❤️ Heart Disease Prediction System")
st.markdown(
    """
    Enter a patient's clinical test results below and the trained Support Vector
    Classifier (SVC) will predict whether the patient has **heart disease**.

    *Trained on the UCI Heart Disease (Cleveland) dataset — 297 patients.
    Accuracy on held-out test data: **75%**.*
    """
)
st.caption("Developed by **Fatima Farzad** (SP24-BSE-025) · AIC354 Machine Learning Fundamentals")

st.divider()

col1, col2 = st.columns(2)
with col1:
    chest_pain = st.selectbox("Chest Pain Type", CHEST_PAIN, index=3)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
with col2:
    vessels = st.selectbox("Major Vessels Coloured by Fluoroscopy", VESSELS)
    thal = st.selectbox("Thallium Stress Test (Thal)", THAL)

if st.button("Predict Heart Disease", type="primary", use_container_width=True):
    user_input = pd.DataFrame({
        "ChestPain": [chest_pain],
        "ExerciseAngina": [exercise_angina],
        "Vessels": [vessels],
        "Thal": [thal],
    })
    encoded = pd.DataFrame({col: encoders[col].transform(user_input[col]) for col in user_input.columns})

    prediction = model.predict(encoded)[0]

    st.divider()
    if prediction == 1:
        st.error("### ⚠️ HEART DISEASE")
    else:
        st.success("### ✅ NO HEART DISEASE")

    with st.expander("See the encoded feature vector sent to the model"):
        st.dataframe(encoded, hide_index=True)

st.divider()
st.caption(
    "Built for an ML course assignment using only 4 input attributes. "
    "This is not a medical diagnosis tool — always consult a doctor."
)
