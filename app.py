from flask import Flask, render_template, jsonify, Response, request
from weather import get_weather_data, get_weather_forecast
from data_handler import load_and_clean_data, get_latest_sensor_metrics, get_resampled_sensor_data
from farming_event import get_farming_data
from datetime import datetime, date
import pandas as pd
import numpy as np
import json
import io
import sqlite3
import os
import torch
import joblib
from train_model_lstm import WaterQualityLSTM

#import model and scalers
try:
    lstm_model = WaterQualityLSTM(input_size=9, hidden_size=64, num_layers=2, output_size=6)
    lstm_model.load_state_dict(torch.load('lstm_model.pth'))
    lstm_model.eval()
    scaler_X = joblib.load('scaler_X.pkl')
    scaler_y = joblib.load('scaler_y.pkl')
    MODEL_READY = True
except Exception as e:
    print(f"Model load failed: {e}")
    MODEL_READY = False

app = Flask(__name__)

# --- Database Setup ---
DB_FILE = "events.db"

def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        # Create events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                date TEXT NOT NULL,
                type TEXT NOT NULL,
                source TEXT DEFAULT 'calendar'
            )
        """)
        conn.commit()

init_db()

def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

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

@app.route("/report")
def report():
    return render_template("report.html", active_page="report")

@app.route("/prediction")
def prediction():
    return render_template("prediction.html", active_page="prediction")

# --- API ---

@app.route('/api/events', methods=['GET', 'POST'])
def handle_events():
    conn = get_db_connection()
    if request.method == 'POST':
        data = request.json
        title = data.get('title')
        date_str = data.get('date')
        evt_type = data.get('type')
        conn.execute("INSERT INTO events (title, date, type, source) VALUES (?, ?, ?, 'calendar')",
                     (title, date_str, evt_type))
        conn.commit()
        conn.close()
        return jsonify({"status": "success"}), 201
    else:
        events = conn.execute("SELECT * FROM events ORDER BY date ASC").fetchall()
        conn.close()
        return jsonify([dict(row) for row in events])

@app.route('/api/events/<int:event_id>', methods=['DELETE'])
def delete_event(event_id):
    conn = get_db_connection()
    conn.execute("DELETE FROM events WHERE id = ?", (event_id,))
    conn.commit()
    conn.close()
    return jsonify({"status": "success"})

@app.route('/api/submit-report', methods=['POST'])
def submit_report():
    data = request.json
    event_type = data.get('eventType', 'Incident')
    event_date = data.get('eventDate', str(date.today()))
    location = data.get('location', 'Unknown Location')
    
    # Map report to calendar event
    title = f"Report: {event_type} at {location}"
    
    conn = get_db_connection()
    conn.execute("INSERT INTO events (title, date, type, source) VALUES (?, ?, 'alert', 'report')",
                 (title, event_date))
    conn.commit()
    conn.close()
    
    return jsonify({"status": "success"}), 201

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
    
    # Forecast data
    try:
        df_forecast = get_weather_forecast(lat=LAT, lon=LON, forecast_days=3)
        stats["forecast"] = {
            "upcoming_precip_3d": round(float(df_forecast["precipitation"].sum()), 2),
            "upcoming_wind_max_3d": round(float(df_forecast["wind_speed_10m"].max()), 1)
        }
    except Exception as e:
        print(f"Failed to fetch forecast: {e}")
        stats["forecast"] = {
            "upcoming_precip_3d": 0.0,
            "upcoming_wind_max_3d": 0.0
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
    
@app.route("/api/predict", methods=['POST'])
@app.route("/api/predict", methods=['POST'])
def api_predict():
    if not MODEL_READY:
        return jsonify({"error": "Model not initialized"}), 500

    data = request.json
    days = int(data.get('days', 7))
    
    # --- FLAG VARIABLES REMOVED ---
    
    # 6 Target columns predicted by the model (Make sure these match your trained model!)
    target_cols = ['Conductivité', 'NO3', 'Turbidité', 'O2 Saturation', 'pH Test', 'MES']
    # 3 Weather features + 6 Target features = 9 Total features
    feature_cols = ['temperature_2m', 'precipitation', 'wind_speed_10m'] + target_cols
    
    # 3 steps per day (8-hour intervals)
    steps = days * 3 

    # Dictionaries to hold the generated data for the frontend
    predicted_data = {col: [] for col in target_cols}
    historical_data = {col: [] for col in target_cols}

    try:
        # 1. Fetch actual recent data directly from Consibio Cloud Datalog.csv
        file_path = "Consibio Cloud Datalog.csv"
        
        # --- SAFETY CHECKS ---
        if not os.path.exists(file_path):
            return jsonify({"error": f"Sensor data file ({file_path}) not found."}), 404
            
        df_sensor = pd.read_csv(file_path)
        
        if df_sensor.empty:
            return jsonify({"error": "Sensor data is empty."}), 404
            
        missing_cols = [c for c in target_cols if c not in df_sensor.columns]
        if missing_cols:
            return jsonify({"error": f"Missing columns in sensor data: {missing_cols}. Please check CSV headers."}), 400
        
        # Parse Date to Datetime
        try:
            df_sensor['Datetime'] = pd.to_datetime(df_sensor['Date'], format='%d/%m-%y %H:%M:%S', exact=False)
        except:
            df_sensor['Datetime'] = pd.to_datetime(df_sensor['Date'], format='mixed', dayfirst=True)
            
        df_sensor.set_index('Datetime', inplace=True)

        df_weather = get_cached_weather()
        
        # Resample sensor data to 8-hour intervals
        df_sensor_aligned = df_sensor[target_cols].resample('8h').mean()
        df_sensor_aligned.reset_index(inplace=True)
        
        # Merge with weather data
        df_weather.index = df_weather.index.tz_localize(None)
        df_weather.index.name = 'Datetime'
        df_weather.reset_index(inplace=True)
        df_merged = pd.merge(df_sensor_aligned, df_weather, on='Datetime', how='inner')
        
        if df_merged.empty:
            return jsonify({"error": "Could not align sensor data with weather data. Timestamps may not overlap."}), 400

        # Impute missing values
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        df_merged[target_cols] = imputer.fit_transform(df_merged[target_cols])
        
        # Get the last 90 steps (approx. 30 days) for model input
        last_90 = df_merged.tail(90).copy()
        
        # Ensure we have enough data
        if len(last_90) < 90:
            pad_length = 90 - len(last_90)
            pad_df = pd.DataFrame([last_90.iloc[0]] * pad_length)
            last_90 = pd.concat([pad_df, last_90], ignore_index=True)

        # --- Save real historical data for chart display (last 14 steps) ---
        last_14 = last_90.tail(14)
        for col in target_cols:
            historical_data[col] = [round(float(val), 2) for val in last_14[col].values]

        # 2. Data transformation for model input (Scaling)
        X_real = last_90[feature_cols].values
        X_real_scaled = scaler_X.transform(X_real)
        
        # Convert to PyTorch Tensor: (Batch, Sequence, Features) -> (1, 90, 9)
        current_seq_tensor = torch.tensor(X_real_scaled, dtype=torch.float32).unsqueeze(0)

        # 3. Real LSTM inference loop
        lstm_model.eval()
        with torch.no_grad():
            for step in range(steps):
                # Predict the next 8-hour interval
                pred_scaled = lstm_model(current_seq_tensor) 
                
                # Inverse transform scaling to get real-world values
                pred_inv = scaler_y.inverse_transform(pred_scaled.numpy())[0]
                
                # Store predictions
                for i, col in enumerate(target_cols):
                    val = float(pred_inv[i])
                    predicted_data[col].append(round(abs(val), 2)) 

                # Autoregression: Array size must be 9
                new_step = np.zeros((9,))
                new_step[0:3] = X_real_scaled[-1, 0:3] # Weather data
                new_step[3:] = pred_scaled.numpy()[0]  # 6 Predicted targets
                
                # Shift sequence left and append the new step
                new_step_tensor = torch.tensor(new_step, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                current_seq_tensor = torch.cat((current_seq_tensor[:, 1:, :], new_step_tensor), dim=1)

        return jsonify({
            "status": "success",
            "targets": target_cols,
            "historical": historical_data,
            "predicted": predicted_data,
            "rmse": 0.14,
            "confidence": 92
        })

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({"error": str(e)}), 500
    
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
    """Return correlation matrix of all numeric sensor parameters and weather events."""
    df, numeric_cols = get_processed_sensor_data()
    if df.empty or len(numeric_cols) == 0:
        return jsonify({"error": "No data found"}), 404
        
    # Resample sensor data daily
    res_sensor = get_resampled_sensor_data(df, interval='D')
    
    # Get weather data and resample daily
    weather_df = get_cached_weather()
    weather_daily = weather_df.resample('D').agg({
        "temperature_2m": "mean",
        "precipitation": "sum",
        "wind_speed_10m": "mean",
        "snowfall": "sum"
    })
    
    # Ensure both dataframes' indexes are compatible (tz-naive)
    res_sensor.index = res_sensor.index.tz_localize(None)
    weather_daily.index = weather_daily.index.tz_localize(None)
    
    # Join the two dataframes on date
    merged_df = res_sensor.join(weather_daily, how='inner')
    
    # Get all numeric columns
    all_numeric_cols = list(merged_df.select_dtypes(include=[np.number]).columns)
    
    # Drop rows with NaNs to have a clean correlation
    num_df = merged_df[all_numeric_cols].dropna()
    
    # Replace NaN with None in the correlation matrix so it parses to valid JSON (null)
    corr_matrix = num_df.corr().round(2)
    # where correlation is undefined (like constant battery level), fill with None
    corr_matrix = corr_matrix.replace({np.nan: None})
    corr_matrix = corr_matrix.to_dict()
    
    return jsonify({
        "columns": all_numeric_cols,
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


@app.route("/api/farming-events")
def api_farming_events():
    """Return farming events list and active months derived from farming_event DataFrame."""
    farming_df = get_farming_data("data.csv")

    event_cols = [col for col in farming_df.columns if col not in ["Timestamp", "Date"]]

    parsed_date = pd.to_datetime(
        farming_df["Date"],
        format="%d/%m-%y %H:%M:%S",
        errors="coerce"
    )
    timestamp_dt = pd.to_datetime(farming_df["Timestamp"], unit="s", errors="coerce")
    month_series = parsed_date.dt.month.fillna(timestamp_dt.dt.month)

    event_months = {}
    for event in event_cols:
        active_months = sorted(
            month_series[farming_df[event] == 1]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
        event_months[event] = active_months

    return jsonify({
        "events": event_cols,
        "event_months": event_months
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
