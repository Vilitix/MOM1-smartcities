from flask import Flask, render_template, jsonify, Response
from weather import get_weather_data
from data_handler import load_and_clean_data, get_latest_sensor_metrics, get_resampled_sensor_data
from datetime import datetime, date
import pandas as pd
import numpy as np
import json
import io

app = Flask(__name__)

# Nancy / ECHO coordinates
LAT = 48.693033
LON = 6.204775

# Global cache for sensor data
_df_sensor = None
_numeric_cols = []

def get_processed_sensor_data():
    """Load and return processed sensor data from data_handler."""
    global _df_sensor, _numeric_cols
    # Re-loading every time for now as requested or until we add a proper trigger
    _df_sensor, _numeric_cols = load_and_clean_data("data.csv")
    return _df_sensor, _numeric_cols

def get_cached_weather():
    """Fetch weather data (cached by requests_cache in weather.py)."""
    return get_weather_data(lat=LAT, lon=LON, days=365)

# --- Pages ---

@app.route("/")
def dashboard():
    return render_template("dashboard.html", active_page="dashboard")

@app.route("/analysis")
def analysis():
    return render_template("analysis.html", active_page="analysis")

@app.route("/correlation")
def correlation():
    return render_template("correlation.html", active_page="correlation")

@app.route("/connectors")
def connectors():
    return render_template("connectors.html", active_page="connectors")

@app.route("/calendar")
def calendar():
    return render_template("calendar.html", active_page="calendar")

# --- API ---

@app.route("/api/weather")
def api_weather():
    """Return weather data as JSON for Chart.js."""
    df = get_cached_weather()
    
    # Latest values
    latest = df.iloc[-1]
    
    # Stats
    stats = {
        "temperature": {
            "current": round(float(latest["temperature_2m"]), 1),
            "mean": round(float(df["temperature_2m"].mean()), 1),
            "min": round(float(df["temperature_2m"].min()), 1),
            "max": round(float(df["temperature_2m"].max()), 1),
            "std": round(float(df["temperature_2m"].std()), 1),
        },
        "precipitation": {
            "current": round(float(latest["precipitation"]), 2),
            "total_30d": round(float(df["precipitation"].tail(90).sum()), 1),
            "mean": round(float(df["precipitation"].mean()), 2),
            "max": round(float(df["precipitation"].max()), 2),
        },
        "wind_speed": {
            "current": round(float(latest["wind_speed_10m"]), 1),
            "mean": round(float(df["wind_speed_10m"].mean()), 1),
            "max": round(float(df["wind_speed_10m"].max()), 1),
            "std": round(float(df["wind_speed_10m"].std()), 1),
        },
    }
    
    # Daily aggregation for yearly chart
    df_daily = df.resample("D").agg({
        "temperature_2m": "mean",
        "precipitation": "sum",
        "wind_speed_10m": "mean",
        "snowfall": "sum"
    })
    df_daily.index = df_daily.index.strftime("%Y-%m-%d")
    
    # Last 7 days at 8h resolution
    df_week = df.tail(21)
    week_labels = df_week.index.strftime("%m-%d %H:%M").tolist()
    
    # Last 30 days at 8h resolution
    df_month = df.tail(90)
    month_labels = df_month.index.strftime("%m-%d").tolist()
    
    # Last 30 data points for the table
    df_table = df.tail(30)
    table_data = []
    for idx, row in df_table.iterrows():
        table_data.append({
            "date": idx.strftime("%Y-%m-%d %H:%M"),
            "temperature": round(float(row["temperature_2m"]), 1),
            "precipitation": round(float(row["precipitation"]), 2),
            "wind_speed": round(float(row["wind_speed_10m"]), 1),
        })
    
    return jsonify({
        "stats": stats,
        "daily": {
            "labels": df_daily.index.tolist(),
            "temperature": [round(v, 1) if pd.notna(v) else None for v in df_daily["temperature_2m"].tolist()],
            "precipitation": [round(v, 1) if pd.notna(v) else None for v in df_daily["precipitation"].tolist()],
            "wind_speed": [round(v, 1) if pd.notna(v) else None for v in df_daily["wind_speed_10m"].tolist()],
            "snowfall": [round(v, 1) if pd.notna(v) else None for v in df_daily["snowfall"].tolist()],
        },
        "weekly": {
            "labels": week_labels,
            "temperature": [round(v, 1) if pd.notna(v) else None for v in df_week["temperature_2m"].tolist()],
            "precipitation": [round(v, 1) if pd.notna(v) else None for v in df_week["precipitation"].tolist()],
            "wind_speed": [round(v, 1) if pd.notna(v) else None for v in df_week["wind_speed_10m"].tolist()],
            "snowfall": [round(v, 1) if pd.notna(v) else None for v in df_week["snowfall"].tolist()],
        },
        "monthly": {
            "labels": month_labels,
            "temperature": [round(v, 1) if pd.notna(v) else None for v in df_month["temperature_2m"].tolist()],
            "precipitation": [round(v, 1) if pd.notna(v) else None for v in df_month["precipitation"].tolist()],
            "wind_speed": [round(v, 1) if pd.notna(v) else None for v in df_month["wind_speed_10m"].tolist()],
            "snowfall": [round(v, 1) if pd.notna(v) else None for v in df_month["snowfall"].tolist()],
        },
        "table": table_data,
    })

@app.route("/api/export")
def api_export():
    """Export all weather data as CSV."""
    df = get_cached_weather()
    buf = io.StringIO()
    df.to_csv(buf)
    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=hydrolens_weather_data.csv"}
    )

@app.route("/api/correlation")
def api_correlation():
    """Return correlation matrix of all numeric sensor parameters."""
    df, numeric_cols = get_processed_sensor_data()
    if df.empty or len(numeric_cols) == 0:
        return jsonify({"error": "No data found"}), 404
        
    num_df = df[numeric_cols].dropna()
    # Replace NaN with None in the correlation matrix so it parses to valid JSON (null)
    corr_matrix = num_df.corr().round(2)
    # where correlation is undefined (like constant battery level), fill with None
    corr_matrix = corr_matrix.replace({np.nan: None})
    corr_matrix = corr_matrix.to_dict()
    
    return jsonify({
        "columns": numeric_cols.tolist(),
        "matrix": corr_matrix
    })

@app.route("/api/sensor-data")
def api_sensor_data():
    """Return processed sensor data from data.csv."""
    df, _ = get_processed_sensor_data()
    if df.empty:
        return jsonify({"error": "No data found"}), 404
    
    # Latest readings
    metrics = get_latest_sensor_metrics(df)
    
    # Resampled data for trends (last 365 days)
    res_df = get_resampled_sensor_data(df, interval='24h') # Daily average for better speed/overview
    res_tail = res_df.tail(365).copy()
    
    def safe_get_col(col_name, ndigits=2):
        if col_name in res_tail:
            return [round(float(v), ndigits) if pd.notna(v) else None for v in res_tail[col_name]]
        return [None] * len(res_tail)

    res_data = {
        "labels": res_tail.index.strftime("%Y-%m-%d").tolist(),
        "ph": safe_get_col("pH Test", 2),
        "turbidity": safe_get_col("Turbidité", 3),
        "conductivity": safe_get_col("Conductivité", 1),
        "temp": safe_get_col("O2 Temperature", 1),
        "all_params": {}
    }
    
    # Store all numeric columns for potential plotting
    _, numeric_cols = load_and_clean_data("data.csv")
    for col in numeric_cols:
        res_data["all_params"][col] = safe_get_col(col, 3)
    
    return jsonify({
        "metrics": metrics,
        "trends": res_data,
        "numeric_columns": numeric_cols.tolist(),
        "total_records": len(df)
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
