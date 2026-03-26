from flask import Flask, render_template, jsonify, Response
from weather import get_weather_data
from datetime import datetime, date
import pandas as pd
import json
import io

app = Flask(__name__)

# Nancy / ECHO coordinates
LAT = 48.693033
LON = 6.204775

def get_cached_data():
    """Fetch weather data (cached by requests_cache in weather.py)."""
    return get_weather_data(lat=LAT, lon=LON, days=365)

# --- Pages ---

@app.route("/")
def dashboard():
    return render_template("dashboard.html", active_page="dashboard")

@app.route("/analysis")
def analysis():
    return render_template("analysis.html", active_page="analysis")

@app.route("/sensors")
def sensors():
    return render_template("sensors.html", active_page="sensors")

@app.route("/calendar")
def calendar():
    return render_template("calendar.html", active_page="calendar")

# --- API ---

@app.route("/api/weather")
def api_weather():
    """Return weather data as JSON for Chart.js."""
    df = get_cached_data()
    
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
        "wind_speed_10m": "mean"
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
        },
        "weekly": {
            "labels": week_labels,
            "temperature": [round(v, 1) if pd.notna(v) else None for v in df_week["temperature_2m"].tolist()],
            "precipitation": [round(v, 1) if pd.notna(v) else None for v in df_week["precipitation"].tolist()],
            "wind_speed": [round(v, 1) if pd.notna(v) else None for v in df_week["wind_speed_10m"].tolist()],
        },
        "monthly": {
            "labels": month_labels,
            "temperature": [round(v, 1) if pd.notna(v) else None for v in df_month["temperature_2m"].tolist()],
            "precipitation": [round(v, 1) if pd.notna(v) else None for v in df_month["precipitation"].tolist()],
            "wind_speed": [round(v, 1) if pd.notna(v) else None for v in df_month["wind_speed_10m"].tolist()],
        },
        "table": table_data,
    })

@app.route("/api/export")
def api_export():
    """Export all weather data as CSV."""
    df = get_cached_data()
    buf = io.StringIO()
    df.to_csv(buf)
    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=hydrolens_weather_data.csv"}
    )

if __name__ == "__main__":
    app.run(debug=True, port=5000)
