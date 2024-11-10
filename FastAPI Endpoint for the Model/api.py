from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
from model_load_save import load_model
import dill

def load_preprocessing_components():
    with open("encoder.pkl", "rb") as f:
        encoder = dill.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = dill.load(f)
    return encoder, scaler

app = FastAPI()

# Load trained model
model = load_model()
encoder, scaler = load_preprocessing_components()

# Define input schema
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


# Health check endpoint
@app.get("/")
def read_root():
    return {"message": "Inference API is up and running"}


# Helper function for preprocessing
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

# Endpoint for single prediction
@app.post("/predict")
def predict(data: InferenceData):
    try:
        # Convert input data to DataFrame
        df = pd.DataFrame([data.model_dump()])

        # Preprocess data
        processed_data = preprocess_data(df)

        # Make prediction
        prediction = model.predict(processed_data)

        # Return prediction result
        return {"prediction": int(prediction[0])}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")


# Endpoint for batch prediction
@app.post("/batch_predict")
def batch_predict(data: List[InferenceData]):
    try:
        # Convert list of inputs to DataFrame
        df = pd.DataFrame([item.model_dump() for item in data])

        # Preprocess data
        processed_data = preprocess_data(df)

        # Make batch predictions
        predictions = model.predict(processed_data)

        # Format and return predictions
        results = [{"input": item.model_dump(), "prediction": int(pred)} for item, pred in zip(data, predictions)]
        return {"predictions": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during batch prediction: {str(e)}")