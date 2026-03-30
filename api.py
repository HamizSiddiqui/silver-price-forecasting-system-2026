from fastapi import FastAPI
from prophet import Prophet
import pandas as pd

app = FastAPI()
handler = app  # required for Vercel

def load_model():
    model = Prophet()
    model = model.load("models/latest_model.json")
    return model

@app.get("/")
def home():
    return {"message": "Silver Forecast API Running"}

@app.get("/predict")
def predict(days: int = 10):
    model = load_model()

    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    result = forecast.tail(days)[["ds", "yhat"]]
    return result.to_dict(orient="records")