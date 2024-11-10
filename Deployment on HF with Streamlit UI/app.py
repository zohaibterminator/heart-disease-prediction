import streamlit as st
import pandas as pd
import numpy as np
from pydantic import BaseModel
from typing import List
from fastapi import HTTPException
from model_load_save import load_model
import dill

def load_preprocessing_components():
    with open("encoder.pkl", "rb") as f:
        encoder = dill.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = dill.load(f)
    return encoder, scaler

model = load_model()
encoder, scaler = load_preprocessing_components()

class InferenceData(BaseModel):
    Age: float
    Sex: str
    ChestPainType: str
    RestingBP: float
    Cholesterol: float
    FastingBS: int
    RestingECG: str
    MaxHR: float
    ExerciseAngina: str
    Oldpeak: float
    ST_Slope: str


def preprocess_data(df: pd.DataFrame) -> np.ndarray:
    # Encode categorical variables
    encoded = encoder.transform(df[encoder.feature_names_in_])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(), index=df.index)

    # Extracting features
    df = pd.concat([df.drop(encoder.feature_names_in_, axis=1), encoded_df], axis=1)

    # Combine and scale features
    df_selected = pd.concat([df[['Oldpeak', 'MaxHR', 'Age']], df[['ExerciseAngina_Y', 'ST_Slope_Flat', 'ST_Slope_Up']]], axis=1) # directly extracted selected features

    # Scale features
    df = scaler.transform(df_selected)

    return df


def predict(data: InferenceData):
    try:
        # Convert input data to DataFrame
        df = pd.DataFrame([data.dict()])

        # Preprocess data
        processed_data = preprocess_data(df)

        # Make prediction
        prediction = model.predict(processed_data)

        # Return prediction result
        return {"prediction": int(prediction[0])}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")


# Define the user input form for prediction
st.title("Heart Disease Prediction")

st.subheader("Enter patient information below:")
age = st.number_input("Age", min_value=0, max_value=120, step=1)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain_type = st.selectbox("Chest Pain Type", ["TA", "ATA", "NAP", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure", min_value=0, max_value=300)
cholesterol = st.number_input("Cholesterol", min_value=0, max_value=600)
fasting_bs = st.selectbox("Fasting Blood Sugar", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.number_input("Maximum Heart Rate", min_value=0, max_value=220)
exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, step=0.1)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

if st.button("Predict"):
    data = InferenceData(
        Age=age,
        Sex=sex,
        ChestPainType=chest_pain_type,
        RestingBP=resting_bp,
        Cholesterol=cholesterol,
        FastingBS=fasting_bs,
        RestingECG=resting_ecg,
        MaxHR=max_hr,
        ExerciseAngina=exercise_angina,
        Oldpeak=oldpeak,
        ST_Slope=st_slope
    )

    result = predict(data)

    st.write("### Prediction:")
    if result == 1:
        st.write("The model predicts a high risk of heart disease.")
    else:
        st.write("The model predicts a low risk of heart disease.")


st.subheader("Batch Prediction")
uploaded_file = st.file_uploader("Upload CSV for batch prediction", type="csv")

if uploaded_file:
    # Load the CSV file
    batch_data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.write(batch_data)

    # Prepare batch data for the API
    batch_data = batch_data.to_dict(orient="records")

    if st.button("Predict Batch"):
        # Send batch data to the API
        batch_response = requests.post(f"{API_URL}/batch_predict", json=batch_data)

        # Display batch prediction results
        if batch_response.status_code == 200:
            predictions = batch_response.json()["predictions"]
            results_df = pd.DataFrame(predictions)
            st.write("Batch Prediction Results:")
            st.write(results_df)
        else:
            st.error("Error: Unable to get batch predictions from API. Please try again later.")