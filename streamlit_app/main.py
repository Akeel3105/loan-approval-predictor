# streamlit_app/main.py

import streamlit as st
import joblib

# Load model and encoders
model = joblib.load("model/loan_model.pkl")
encoders = joblib.load("model/encoders.pkl")

# UI
st.title("🏦 Loan Approval Predictor")
st.write("Fill the form below to check loan eligibility.")

# Form inputs
gender = st.selectbox("Gender", encoders["Gender"].classes_)
married = st.selectbox("Married", encoders["Married"].classes_)
education = st.selectbox("Education", encoders["Education"].classes_)
self_employed = st.selectbox("Self Employed", encoders["Self_Employed"].classes_)
applicant_income = st.number_input("Applicant Income", value=5000)
coapplicant_income = st.number_input("Coapplicant Income", value=2000)
loan_amount = st.number_input("Loan Amount (in thousands)", value=120)
loan_term = st.number_input("Loan Term (in days)", value=360)
credit_history = st.selectbox("Credit History", [0.0, 1.0])
property_area = st.selectbox("Property Area", encoders["Property_Area"].classes_)

if st.button("Predict"):
    try:
        # Prepare input
        input_data = [
            encoders["Gender"].transform([gender])[0],
            encoders["Married"].transform([married])[0],
            encoders["Education"].transform([education])[0],
            encoders["Self_Employed"].transform([self_employed])[0],
            applicant_income,
            coapplicant_income,
            loan_amount,
            loan_term,
            credit_history,
            encoders["Property_Area"].transform([property_area])[0],
        ]

        prediction = model.predict([input_data])[0]
        result = "✅ Loan Approved" if prediction == 1 else "❌ Loan Rejected"
        st.success(result)

    except Exception as e:
        st.error(f"Error: {str(e)}")
