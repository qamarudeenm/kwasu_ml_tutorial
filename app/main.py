from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(title="Student Performance Prediction API")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Global variables to store models and scalers
models = {}

@app.on_event("startup")
def load_artifacts():
    # Load models and scalers
    try:
        models['lin_reg'] = joblib.load('models/linear_regression.pkl')
        models['log_reg'] = joblib.load('models/logistic_regression.pkl')
        models['scaler'] = joblib.load('models/scaler.pkl')
        models['label_encoder'] = joblib.load('models/label_encoder.pkl')
        print("All artifacts loaded successfully.")
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        # In production, we might want to crash if models fail to load
        pass

class StudentData(BaseModel):
    weekly_self_study_hours: float
    attendance_percentage: float
    class_participation: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the Student Performance Prediction API"}

@app.post("/predict/score")
def predict_score(data: StudentData):
    if 'lin_reg' not in models:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Prepare input data
    input_df = pd.DataFrame([data.dict()])
    
    # Scale features
    scaled_features = models['scaler'].transform(input_df)
    
    # Predict
    prediction = models['lin_reg'].predict(scaled_features)
    
    return {"predicted_total_score": float(prediction[0])}

@app.post("/predict/grade")
def predict_grade(data: StudentData):
    if 'log_reg' not in models:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Prepare input data
    input_df = pd.DataFrame([data.dict()])
    
    # Scale features
    scaled_features = models['scaler'].transform(input_df)
    
    # Predict
    prediction_idx = models['log_reg'].predict(scaled_features)[0]
    
    # Decode label
    prediction_label = models['label_encoder'].inverse_transform([prediction_idx])[0]
    
    return {"predicted_grade": prediction_label}
