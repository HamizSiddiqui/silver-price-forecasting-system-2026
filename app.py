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
INDEX_PATH = Path(__file__).resolve().parent / "index.html"
DATA_PATH = Path(__file__).resolve().parent / "data" / "silver_prices.csv"
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
    if INDEX_PATH.exists():
        return FileResponse(str(INDEX_PATH), media_type="text/html")
    return {"message": "Silver Forecast System Running"}


@app.get("/graph")
def graph_plot():
    if not PLOT_HTML_PATH.exists():
        raise HTTPException(status_code=404, detail="Forecast plot not found")
    return FileResponse(str(PLOT_HTML_PATH), media_type="text/html")


@app.get("/metrics")
def get_metrics():
    try:
        # Load latest data
        df = pd.read_csv(DATA_PATH)
        current_price = float(df.iloc[-1]["Silver_PKR_per_Ounce"])
        
        # Load model & forecast
        model = load_model()
        future = model.make_future_dataframe(periods=1)
        forecast = model.predict(future)
        pred_price = float(forecast.iloc[-1]["yhat"])
        
        # Get last update time
        last_update = Path(DATA_PATH).stat().st_mtime
        from datetime import datetime
        last_update_str = datetime.fromtimestamp(last_update).strftime("%B %d, %Y, %I:%M %p")
        
        return {
            "current_price": current_price,
            "predicted_price": pred_price,
            "training_time": last_update_str
        }
    except Exception as e:
        return {"error": str(e)}


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