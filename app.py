import streamlit as st
import requests
import pandas as pd

# Set the FastAPI URL
API_URL = "http://127.0.0.1:8000"  # Replace with your FastAPI URL if different

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

# Button to submit the form
if st.button("Predict"):
    # Prepare the data payload
    data = {
        "Age": age,
        "Sex": sex,
        "ChestPainType": chest_pain_type,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "RestingECG": resting_ecg,
        "MaxHR": max_hr,
        "ExerciseAngina": exercise_angina,
        "Oldpeak": oldpeak,
        "ST_Slope": st_slope
    }
    
    # Send a request to the FastAPI server
    response = requests.post(f"{API_URL}/predict", json=data)

    # Display the result
    if response.status_code == 200:
        prediction = response.json()["prediction"]
        result = "Positive for heart disease" if prediction == 1 else "Negative for heart disease"
        st.success(f"Prediction: {result}")
    else:
        st.error("Error: Unable to get prediction from API. Please try again later.")

# Batch Prediction Section
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