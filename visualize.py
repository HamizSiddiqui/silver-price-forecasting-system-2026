from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.metrics import r2_score

try:
    from prophet.serialize import model_from_json
except ImportError:  # type: ignore
    from fbprophet.serialize import model_from_json  # type: ignore

DATA_FILE = Path("data") / "silver_prices.csv"
MODEL_FILE = Path("models") / "latest_model.json"


def load_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Silver_PKR_per_Ounce"])
    df["Silver_PKR_per_Ounce"] = pd.to_numeric(df["Silver_PKR_per_Ounce"], errors="coerce")
    df = df.dropna(subset=["Silver_PKR_per_Ounce"])
    df = df.sort_values("Date")
    return df


def load_model() -> Prophet:
    if MODEL_FILE.exists():
        model_text = MODEL_FILE.read_text(encoding="utf-8")
        return model_from_json(model_text)

    df = load_data()
    prophet_df = df[["Date", "Silver_PKR_per_Ounce"]].rename(
        columns={"Date": "ds", "Silver_PKR_per_Ounce": "y"}
    )
    model = Prophet()
    model.fit(prophet_df)
    return model


def add_unit_conversions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["gram"] = df["Silver_PKR_per_Ounce"] / 31.1035
    df["tola"] = df["gram"] * 11.6638
    df["kg"] = df["gram"] * 1000
    return df


def create_forecast_plot(df: pd.DataFrame, forecast: pd.DataFrame) -> go.Figure:
    df = add_unit_conversions(df)
    forecast_units = add_unit_conversions(forecast.rename(columns={"yhat": "Silver_PKR_per_Ounce"}))
    forecast_units["Date"] = forecast_units["ds"]

    merged = pd.merge(
        df[["Date", "Silver_PKR_per_Ounce"]].rename(columns={"Date": "ds"}),
        forecast[["ds", "yhat"]],
        on="ds",
        how="inner",
    )
    r2 = r2_score(merged["Silver_PKR_per_Ounce"], merged["yhat"]) if not merged.empty else float("nan")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["tola"],
            mode="lines",
            name="Tola (Historical)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["gram"],
            mode="lines",
            name="Gram (Historical)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["kg"],
            mode="lines",
            name="Kilogram (Historical)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_units["Date"],
            y=forecast_units["tola"],
            mode="lines",
            name="Tola (Forecast)",
            line=dict(dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_units["Date"],
            y=forecast_units["gram"],
            mode="lines",
            name="Gram (Forecast)",
            line=dict(dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_units["Date"],
            y=forecast_units["kg"],
            mode="lines",
            name="Kilogram (Forecast)",
            line=dict(dash="dash"),
        )
    )

    fig.update_layout(
        title=f"Silver Price Pakistan — Prophet Forecast (R² = {r2:.2f})",
        xaxis_title="Date",
        yaxis_title="Price / Converted Units",
        hovermode="x unified",
        template="plotly_white",
    )
    return fig


def main() -> None:
    df = load_data()
    model = load_model()

    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)

    fig = create_forecast_plot(df, forecast)
    output_file = Path("models") / "forecast_plot.html"
    fig.write_html(output_file, include_plotlyjs="cdn")
    print(f"Forecast plot saved to {output_file.resolve()}")
    fig.show()


if __name__ == "__main__":
    main()
