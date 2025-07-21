# model/train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import os

# Load Data
df = pd.read_csv("data/train.csv")

# Basic preprocessing
df.dropna(inplace=True)

# Encoding categorical variables
categorical_cols = ["Gender", "Married", "Education", "Self_Employed", "Property_Area"]
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Features and target
X = df[
    [
        "Gender",
        "Married",
        "Education",
        "Self_Employed",
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
        "Property_Area",
    ]
]

y = df["Loan_Status"].map({"Y": 1, "N": 0})  # Target encoding

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc:.2f}")

# Save model and encoders
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/loan_model.pkl")
joblib.dump(encoders, "model/encoders.pkl")
