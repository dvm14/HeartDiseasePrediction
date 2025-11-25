from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Heart Disease Predictor (Pipeline)")

# Load the pipeline you saved
model = joblib.load("models/best_rf_model.pkl")

# Pydantic schema: use natural types found in the CSV
# Sex: "M"/"F"
# ChestPainType: "TA","ATA","NAP","ASY"
# RestingECG: "Normal","ST","LVH"
# ExerciseAngina: "Y"/"N"
# ST_Slope: "Up","Flat","Down"
class HeartData(BaseModel):
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

@app.post("/predict")
def predict(payload: HeartData):
    # Build a single-row DataFrame with the exact CSV column names
    df = pd.DataFrame([{
        "Age": payload.Age,
        "Sex": payload.Sex,
        "ChestPainType": payload.ChestPainType,
        "RestingBP": payload.RestingBP,
        "Cholesterol": payload.Cholesterol,
        "FastingBS": payload.FastingBS,
        "RestingECG": payload.RestingECG,
        "MaxHR": payload.MaxHR,
        "ExerciseAngina": payload.ExerciseAngina,
        "Oldpeak": payload.Oldpeak,
        "ST_Slope": payload.ST_Slope,
    }])
    pred = model.predict(df)[0]
    # If you want probabilities:
    # proba = float(model.predict_proba(df)[0][1])
    return {"prediction": int(pred)}
