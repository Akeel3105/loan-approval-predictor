# api/app.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load model and encoders
model = joblib.load("model/loan_model.pkl")
encoders = joblib.load("model/encoders.pkl")

# Input schema
class LoanInput(BaseModel):
    Gender: str
    Married: str
    Education: str
    Self_Employed: str
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: str

@app.get("/")
def root():
    return {"message": "Loan Approval Predictor API is up!"}

@app.post("/predict")
def predict_loan(data: LoanInput):
    try:
        # Apply encoding
        inputs = [
            encoders['Gender'].transform([data.Gender])[0],
            encoders['Married'].transform([data.Married])[0],
            encoders['Education'].transform([data.Education])[0],
            encoders['Self_Employed'].transform([data.Self_Employed])[0],
            data.ApplicantIncome,
            data.CoapplicantIncome,
            data.LoanAmount,
            data.Loan_Amount_Term,
            data.Credit_History,
            encoders['Property_Area'].transform([data.Property_Area])[0]
        ]

        # Prediction
        prediction = model.predict([inputs])[0]
        result = "Loan Approved ✅" if prediction == 1 else "Loan Rejected ❌"
        return {"prediction": result}

    except Exception as e:
        return {"error": str(e)}
