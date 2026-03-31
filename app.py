from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from prophet.serialize import model_from_json
import pandas as pd

from fastapi.staticfiles import StaticFiles

app = FastAPI()
handler = app  # required for Vercel

# Serve static files (CSS, JS) from the current directory
app.mount("/static", StaticFiles(directory="."), name="static")

MODEL_PATH = Path(__file__).resolve().parent / "models" / "latest_model.json"
PLOT_HTML_PATH = Path(__file__).resolve().parent / "forecast_plot.html"
_model_cache = None


def load_model():
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    with MODEL_PATH.open("r", encoding="utf-8") as f:
        _model_cache = model_from_json(f.read())

    return _model_cache


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@app.get("/")
def home():
    if PLOT_HTML_PATH.exists():
        return FileResponse(str(PLOT_HTML_PATH), media_type="text/html")
    return {"message": "Silver Forecast API Running"}


@app.get("/plot")
def plot():
    if not PLOT_HTML_PATH.exists():
        raise HTTPException(status_code=404, detail="Plot HTML file not found")
    return FileResponse(str(PLOT_HTML_PATH), media_type="text/html")


@app.get("/predict")
def predict(days: int = 10):
    if days < 1:
        raise HTTPException(status_code=400, detail="days must be a positive integer")

    try:
        model = load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to load forecast model")

    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    result = forecast.tail(days)[["ds", "yhat"]]
    return result.to_dict(orient="records")