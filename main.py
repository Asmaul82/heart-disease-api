from fastapi import FastAPI
import joblib
import numpy as np
from schemas import HeartData

model = joblib.load("model/heart_model.joblib")

app = FastAPI(title="Heart Disease Prediction API")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/info")
def model_info():
    return {
        "model": "Logistic Regression",
        "features": ["age", "sex", "cp", "trestbps", "chol", "fbs",
                     "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
    }

@app.post("/predict")
def predict(data: HeartData):
    input_data = np.array([[data.age, data.sex, data.cp, data.trestbps, data.chol,
                            data.fbs, data.restecg, data.thalach, data.exang,
                            data.oldpeak, data.slope, data.ca, data.thal]])
    prediction = model.predict(input_data)
    return {"heart_disease": bool(prediction[0])}
