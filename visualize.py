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

    # Add traces for Tola, Gram, and KG (matching requested colors)
    # Tola: Blue/Indigo
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["tola"],
            mode="lines",
            name="Tola (Historical)",
            line=dict(color="#636efa", width=2),
        )
    )
    # Gram: Red/Orange
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["gram"],
            mode="lines",
            name="Gram (Historical)",
            line=dict(color="#ef553b", width=2),
            visible="legendonly",
        )
    )
    # Kilogram: Green
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["kg"],
            mode="lines",
            name="Kilogram (Historical)",
            line=dict(color="#00cc96", width=2),
            visible="legendonly",
        )
    )

    # Forecast traces (using dashed lines with corresponding colors)
    # Tola Forecast: Purple
    fig.add_trace(
        go.Scatter(
            x=forecast_units["Date"],
            y=forecast_units["tola"],
            mode="lines",
            name="Tola (Forecast)",
            line=dict(dash="dash", color="#ab63fa", width=2),
        )
    )
    # Gram Forecast: Light Orange/Red
    fig.add_trace(
        go.Scatter(
            x=forecast_units["Date"],
            y=forecast_units["gram"],
            mode="lines",
            name="Gram (Forecast)",
            line=dict(dash="dash", color="#ffa15a", width=2),
            visible="legendonly",
        )
    )
    # Kilogram Forecast: Light Blue/Cyan
    fig.add_trace(
        go.Scatter(
            x=forecast_units["Date"],
            y=forecast_units["kg"],
            mode="lines",
            name="Kilogram (Forecast)",
            line=dict(dash="dash", color="#19d3f3", width=2),
            visible="legendonly",
        )
    )

    fig.update_layout(
        title=dict(
            text=f"Silver Price Pakistan — Prophet Forecast (R² = {r2:.2f})",
            x=0.5,
            xanchor='center',
            font=dict(color="white", size=20)
        ),
        xaxis_title="Date",
        yaxis_title="Price / Converted Units",
        hovermode="x unified",
        template="plotly_dark",
        paper_bgcolor="black",
        plot_bgcolor="black",
        xaxis=dict(
            showgrid=True,
            gridcolor="#2c2c2c",
            title_font=dict(color="#94a3b8")
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#2c2c2c",
            title_font=dict(color="#94a3b8")
        ),
        margin=dict(l=60, r=180, t=100, b=60),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(0,0,0,0.6)",
            bordercolor="rgba(255,255,255,0.2)",
            borderwidth=1,
            font=dict(size=11, color="white")
        )
    )
    return fig


def main() -> None:
    df = load_data()
    model = load_model()

    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)

    fig = create_forecast_plot(df, forecast)
    output_file = Path("forecast_plot.html")
    
    # Enhanced config for mobile stability and interactivity
    config = {
        'displayModeBar': False,
        'responsive': True,
        'scrollZoom': False
    }
    
    # Pure JavaScript for post_script (No <script> or <style> tags allowed here)
    post_script = """
    // Inject CSS
    var style = document.createElement('style');
    style.innerHTML = `
        html, body {
            margin: 0;
            padding: 0;
            height: 100vh !important;
            width: 100vw !important;
            background-color: #1e293b !important;
            overflow: hidden !important;
        }
        .plotly-graph-div {
            height: 100vh !important;
            width: 100vw !important;
        }
        .js-plotly-plot .plotly .main-svg {
            touch-action: pan-y !important;
        }
    `;
    document.head.appendChild(style);

    // Force redraw on window resize
    window.addEventListener('resize', function() {
        var plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
        if (plotDiv) {
            Plotly.Plots.resize(plotDiv);
        }
    });

    // Initial resize trigger
    setTimeout(function() {
        var plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
        if (plotDiv) Plotly.Plots.resize(plotDiv);
    }, 100);
    """
    
    fig.write_html(
        output_file, 
        include_plotlyjs="cdn",
        config=config,
        post_script=post_script,
        full_html=True
    )
    print(f"Forecast plot saved to {output_file.resolve()}")


if __name__ == "__main__":
    main()
