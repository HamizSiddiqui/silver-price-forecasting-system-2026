from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from prophet import Prophet

try:
    from prophet.serialize import model_to_json
except ImportError:  # type: ignore
    from fbprophet.serialize import model_to_json  # type: ignore

DATA_FILE = Path("data") / "silver_prices.csv"
MODEL_DIR = Path("models")
MODEL_FILE = MODEL_DIR / "latest_model.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def ensure_artifacts() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_training_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Training CSV not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    if df.empty:
        raise ValueError("Training CSV is empty.")

    expected_columns = {"Date", "Silver_PKR_per_Ounce"}
    if not expected_columns.issubset(df.columns):
        raise ValueError(f"Training CSV must contain columns: {expected_columns}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Silver_PKR_per_Ounce"])
    df["Silver_PKR_per_Ounce"] = pd.to_numeric(df["Silver_PKR_per_Ounce"], errors="coerce")
    df = df.dropna(subset=["Silver_PKR_per_Ounce"])

    if df.empty:
        raise ValueError("No valid rows found after cleaning training data.")

    return df.sort_values("Date")


def save_model(model: Prophet) -> None:
    ensure_artifacts()
    model_json = model_to_json(model)
    MODEL_FILE.write_text(model_json, encoding="utf-8")
    logger.info("Saved trained model to %s", MODEL_FILE)


def train_model() -> Prophet:
    df = load_training_data()
    prophet_df = df[["Date", "Silver_PKR_per_Ounce"]].rename(
        columns={"Date": "ds", "Silver_PKR_per_Ounce": "y"}
    )

    logger.info("Training Prophet model on %d historic rows.", len(prophet_df))
    model = Prophet()
    model.fit(prophet_df)

    save_model(model)
    logger.info("Training completed successfully.")
    return model


if __name__ == "__main__":
    train_model()
