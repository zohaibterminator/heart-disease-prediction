# Heart Disease Prediction Application

This repository contains a FastAPI backend that serves a machine learning model for heart disease prediction, along with a Streamlit frontend to interact with the model locally. Follow the instructions below to set up each component.

## Requirements

Ensure you have the following installed on your machine:

* Python 3.10+

* pip for package management

## Repository Structure

```plaintext
.
├── 1. Data Cleaning/data_cleaning.py                                # Data cleaning and preprocessing pipeline
├── 2. Model Building/model_building.py                              # Model training and saving
├── 3. Model Saving/model_load_save.py                               # Model saving and loading script
├── 4. FastAPI Endpoint for the model/api.py                         # FastAPI application
├── 5. Deployment on HF with Streamlit UIapp.py                      # Streamlit frontend
├── requirements.txt                                                 # Dependencies
├── model_load_save.py                                               # Utilities for loading model and preprocessing components
└── README.md                                                        # Documentation
```

## Instructions

### Step 1: Install Dependencies

First, clone this repository and install all required Python packages:

```bash
git clone <repository-url>
cd <repository-name>
pip install -r requirements.txt
```

### Step 2: Run the ML Pipeline (Data Cleaning to Model Building)

To generate a trained model, follow these steps:

#### 1. Data Cleaning and Preprocessing

* Run `data_cleaning_pipeline.py` to clean and preprocess the data.
* This script will load your dataset, perform feature selection, handle encoding, scaling, and save the necessary preprocessing components (e.g., encoders, scalers) required for inference.

```bash
python data_cleaning_pipeline.py
```

#### 2. Model Training

* Run `model_training.py` to train and save the model using the preprocessed data.
* TThe model and preprocessing components will be saved for later use in the FastAPI app.

```bash
python model_training.py
```

#### 3. Run the FastAPI Backend

* To start the FastAPI app locally, run:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

* The backend will be accessible at `http://localhost:8000`

```bash
python data_cleaning_pipeline.py
```

### 4. Run the Streamlit Frontend

* To start the Streamlit app, run:

```bash
streamlit run app.py
```

* Open a browser and go to `http://localhost:8501` to access the frontend
