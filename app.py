from flask import Flask, render_template, jsonify, Response, request, session, redirect, url_for
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
import sys
import time
import threading

# Add scripts directory to path for model imports (e.g., train_model_lstm)
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

# Create a global lock for data loading
data_lock = threading.Lock()

# --- Lite Mode Detection ---
# Enable if LITE_MODE environment variable is set OR if "lite" is in command line arguments
LITE_MODE = (os.environ.get("LITE_MODE", "false").lower() == "true" or 
             "lite" in sys.argv)

# Placeholder variables for AI model
MODEL_READY = False
lstm_model = None
scaler_X = None
scaler_y = None

# Conditionally import Heavy AI libraries
if not LITE_MODE:
    try:
        import torch
        import joblib
        from train_model_lstm import WaterQualityLSTM

        #import model and scalers
        lstm_model = WaterQualityLSTM(input_size=11, hidden_size=128, num_layers=2, output_size=11)
        lstm_model.load_state_dict(torch.load('models/lstm_model.pth', map_location=torch.device('cpu')))
        lstm_model.eval()
        scaler_X = joblib.load('models/scaler_X.pkl')
        scaler_y = joblib.load('models/scaler_y.pkl')
        MODEL_READY = True
    except Exception as e:
        print(f"AI Model load failed: {e}")
        MODEL_READY = False

app = Flask(__name__)
app.secret_key = "hydrolens_secret_key" # Needed for sessions

@app.context_processor
def inject_lang():
    lang = session.get('lang', 'fr')
    return dict(lang=lang)

@app.route("/set_language/<lang>")
def set_language(lang):
    if lang in ['en', 'fr']:
        session['lang'] = lang
    return redirect(request.referrer or url_for('dashboard'))

@app.context_processor
def inject_lite_mode():
    return dict(lite_mode=LITE_MODE)

# --- Database Setup ---
DB_FILE = "data/events.db"

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
_sensor_data_mtime = None
_farming_cache = None # Cache for farming metadata

def get_processed_sensor_data():
    """Load and return processed sensor data from data_handler."""
    global _df_sensor, _numeric_cols, _sensor_data_mtime
    file_path = "data/data.csv"
    
    # Use lock to prevent multiple threads from loading data at once
    with data_lock:
        current_mtime = os.path.getmtime(file_path) if os.path.exists(file_path) else None

        # Reload only if cache is empty or source file changed.
        if _df_sensor is None or _numeric_cols is None or _sensor_data_mtime != current_mtime:
            start = time.time()
            print(f"[DEBUG] Sensor reload starting from {file_path}...")
            _df_sensor, _numeric_cols = load_and_clean_data(file_path)
            _sensor_data_mtime = current_mtime
            print(f"[DEBUG] Sensor reload COMPLETED in {time.time() - start:.3f}s (Rows: {len(_df_sensor)})")

    return _df_sensor, _numeric_cols

def get_cached_weather():
    """Fetch weather data (cached by requests_cache in weather.py)."""
    start = time.time()
    res = get_weather_data(lat=LAT, lon=LON, days=365)
    print(f"[DEBUG] Weather Fetch took {time.time() - start:.3f}s")
    return res

# --- Cache Preloading on Startup ---
def warm_up_caches():
    """Runs a background warm-up of sensor, weather and farming data."""
    def _run():
        print("[DEBUG] STARTING CACHE WARM-UP (Background Thread)...")
        get_processed_sensor_data()
        get_cached_weather()
        
        # Pre-load farming metadata
        global _farming_cache
        from farming_event import get_event_metadata
        _farming_cache = get_event_metadata()
        
        print("[DEBUG] CACHE WARM-UP COMPLETED.")

    # Start the warm-up in a daemon thread so it doesn't block app startup
    threading.Thread(target=_run, daemon=True).start()

# Execute warm-up
warm_up_caches()

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
    start = time.time()
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
def api_predict():
    if not MODEL_READY:
        return jsonify({"error": "Model not initialized"}), 500

    start_total = time.time()
    data = request.json
    days = int(data.get('days', 7))
    
    water_cols = ['Conductivité', 'NO3', 'Turbidité', 'O2 Saturation', 'pH Test', 'MES', 'DBOeq', 'Phycocyanine scaled']
    target_cols = ['temperature_2m', 'precipitation', 'wind_speed_10m'] + water_cols
    feature_cols = target_cols
    
    steps = days * 3 
    predicted_data = {col: [] for col in target_cols}
    historical_data = {col: [] for col in target_cols}

    try:
        # 1. Fetch Forecast
        df_forecast = get_weather_forecast(lat=LAT, lon=LON, forecast_days=days)
        df_forecast_res = df_forecast.resample('8h').mean().head(steps)

        # 2. Prepare Sensor Data aligned to 8h
        df_sensor_cached, _ = get_processed_sensor_data()
        df_sensor = df_sensor_cached.copy()
        df_sensor_aligned = df_sensor[water_cols].resample('8h').mean()
        
        # 3. Prepare Weather and Merge on Index
        df_weather = get_cached_weather()
        df_weather.index = df_weather.index.tz_localize(None)
        
        # Inner Join: Keeps only overlapping timestamps as a DatetimeIndex
        df_merged = df_sensor_aligned.join(df_weather, how='inner')

        if df_merged.empty:
            return jsonify({"error": "Data alignment failed"}), 400

        # 4. Safe Imputation while preserving the Index
        from sklearn.impute import SimpleImputer
        for col in target_cols:
            if col not in df_merged.columns: df_merged[col] = np.nan
        
        # FIX: Ensure no column is 100% NaN (avoids SimpleImputer dropping columns)
        df_merged[target_cols] = df_merged[target_cols].fillna(0) if df_merged[target_cols].isna().all().any() else df_merged[target_cols]
        # Or better: fill completely empty columns with a single 0
        for col in target_cols:
            if df_merged[col].isna().all():
                df_merged[col] = 0.0

        imputer = SimpleImputer(strategy='mean', fill_value=0)
        imputed_data = imputer.fit_transform(df_merged[target_cols])
        # REBUILD: Preserve the DatetimeIndex explicitly
        df_merged = pd.DataFrame(imputed_data, columns=target_cols, index=df_merged.index)
        
        # 5. Prepare Sequence
        last_90 = df_merged.tail(90).copy()
        if len(last_90) < 90:
            pad_df = pd.DataFrame([last_90.iloc[0]] * (90 - len(last_90)), columns=target_cols)
            # Handle padding carefully to maintain index (though mostly for safety)
            last_90 = pd.concat([pad_df, last_90])

        # Labels from Index
        last_14 = last_90.tail(14)
        historical_labels = [dt.strftime("%b %d, %H:%M") for dt in last_14.index]
        
        last_ts = last_14.index[-1]
        predicted_labels = []
        for i in range(1, steps + 1):
            future_dt = last_ts + pd.Timedelta(hours=i * 8)
            predicted_labels.append(future_dt.strftime("%b %d, %H:%M"))

        historical_data = {col: [round(float(v), 2) for v in last_14[col]] for col in target_cols}

        # Last month comparison 
        try:
            total_steps = 14 + steps
            one_month_ago_start = last_ts - pd.Timedelta(days=30) - pd.Timedelta(hours=14 * 8)
            df_last_month = df_merged.loc[df_merged.index >= one_month_ago_start].head(total_steps)
            last_month_data = {col: [round(float(v), 2) if pd.notna(v) else None for v in df_last_month[col]] for col in target_cols}
            for col in target_cols:
                if len(last_month_data[col]) < total_steps:
                    last_month_data[col] += [None] * (total_steps - len(last_month_data[col]))
        except:
            last_month_data = {col: [None] * (14 + steps) for col in target_cols}
        
        # 6. Inference Loop
        X_real_scaled = scaler_X.transform(last_90[feature_cols].values)
        current_seq_tensor = torch.tensor(X_real_scaled, dtype=torch.float32).unsqueeze(0)

        predicted_data = {col: [] for col in target_cols}
        lstm_model.eval()
        with torch.no_grad():
            for step in range(steps):
                pred_scaled = lstm_model(current_seq_tensor)
                vals_scaled = pred_scaled.detach().cpu().numpy()[0]
                
                if step < len(df_forecast_res):
                    fc_row = df_forecast_res.iloc[[step]].copy()
                    for col in water_cols: fc_row[col] = 0 
                    fc_scaled = scaler_X.transform(fc_row[feature_cols].values)[0]
                    vals_scaled[0:3] = fc_scaled[0:3] # Replace weather indices
                
                pred_inv = scaler_y.inverse_transform(vals_scaled.reshape(1, -1))[0]
                for i, col in enumerate(target_cols):
                    predicted_data[col].append(round(max(0, float(pred_inv[i])), 2))

                new_step_tensor = torch.tensor(vals_scaled, dtype=torch.float32).view(1, 1, -1)
                current_seq_tensor = torch.cat((current_seq_tensor[:, 1:, :], new_step_tensor), dim=1)

        return jsonify({
            "status": "success",
            "targets": target_cols,
            "historical": historical_data,
            "predicted": predicted_data,
            "last_month": last_month_data,
            "historical_labels": historical_labels,
            "predicted_labels": predicted_labels
        })

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({"error": str(e)}), 500
        
@app.route("/api/validate", methods=['GET'])
def api_validate():
    if not MODEL_READY:
        return jsonify({"error": "Model not initialized"}), 500

    water_cols = ['Conductivité', 'NO3', 'Turbidité', 'O2 Saturation', 'pH Test', 'MES', 'DBOeq', 'Phycocyanine scaled']
    target_cols = ['temperature_2m', 'precipitation', 'wind_speed_10m'] + water_cols
    feature_cols = target_cols
    
    try:
        # Start of March
        target_start = pd.Timestamp('2026-03-01')
        target_end = pd.Timestamp('2026-03-27') # Data ends Mar 26
        input_start = target_start - pd.Timedelta(days=30)

        df_sensor_cached, _ = get_processed_sensor_data()
        df_sensor_aligned = df_sensor_cached[water_cols].resample('8h').mean()
        
        df_weather = get_cached_weather()
        df_weather.index = df_weather.index.tz_localize(None)
        
        df_merged = df_sensor_aligned.join(df_weather, how='inner')
        
        df_input_raw = df_merged.loc[(df_merged.index >= input_start) & (df_merged.index < target_start)].copy()
        df_actual_raw = df_merged.loc[(df_merged.index >= target_start) & (df_merged.index < target_end)].copy()
        
        if len(df_input_raw) < 90:
             return jsonify({"error": "Insufficient history before March"}), 400

        from sklearn.impute import SimpleImputer
        
        # Ensure all columns exist and are NOT 100% NaN
        for col in target_cols:
            if col not in df_input_raw.columns: df_input_raw[col] = 0.0
            if df_input_raw[col].isna().all(): df_input_raw[col] = 0.0
            
            if col not in df_actual_raw.columns: df_actual_raw[col] = 0.0
            if df_actual_raw[col].isna().all(): df_actual_raw[col] = 0.0

        imputer = SimpleImputer(strategy='mean', fill_value=0)
        
        # Impute and rebuild while keeping index
        df_input = pd.DataFrame(imputer.fit_transform(df_input_raw[target_cols]), 
                                columns=target_cols, index=df_input_raw.index)
        df_actual = pd.DataFrame(imputer.transform(df_actual_raw[target_cols]), 
                                 columns=target_cols, index=df_actual_raw.index)

        # Inference initialization
        last_90 = df_input.tail(90)
        X_scaled = scaler_X.transform(last_90[feature_cols].values)
        current_seq_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0)
        
        steps = len(df_actual)
        predicted_values = {col: [] for col in target_cols}
        
        lstm_model.eval()
        with torch.no_grad():
            for i in range(steps):
                pred_scaled = lstm_model(current_seq_tensor)
                vals_scaled = pred_scaled.detach().cpu().numpy()[0]
                
                # Use real weather from df_actual
                real_weather_scaled = scaler_X.transform(df_actual.iloc[[i]][feature_cols].values)[0]
                vals_scaled[0:3] = real_weather_scaled[0:3]
                
                pred_inv = scaler_y.inverse_transform(vals_scaled.reshape(1, -1))[0]
                for j, col in enumerate(target_cols):
                    predicted_values[col].append(round(max(0, float(pred_inv[j])), 2))
                
                new_step_tensor = torch.tensor(vals_scaled, dtype=torch.float32).view(1, 1, -1)
                current_seq_tensor = torch.cat((current_seq_tensor[:, 1:, :], new_step_tensor), dim=1)

        return jsonify({
            "status": "success",
            "targets": target_cols,
            "historical": {col: [round(float(v), 2) for v in df_input.tail(3)[col]] for col in target_cols},
            "actual": {col: [round(float(v), 2) for v in df_actual[col]] for col in target_cols},
            "predicted": predicted_values,
            "historical_labels": [dt.strftime("%b %d, %H:%M") for dt in df_input.tail(3).index],
            "validation_labels": [dt.strftime("%b %d, %H:%M") for dt in df_actual.index]
        })

    except Exception as e:
        print(f"Validation Error: {e}")
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
        headers={"Content-Disposition": "attachment;filename=data.csv"}
    )

@app.route("/api/correlation")
def api_correlation():
    """Return correlation matrix of all numeric sensor parameters and weather events."""
    start = time.time()
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
    
    print(f"[DEBUG] Correlation compute took {time.time() - start:.3f}s")
    return jsonify({
        "columns": all_numeric_cols,
        "matrix": corr_matrix
    })

@app.route("/api/sensor-data")
def api_sensor_data():
    """Return processed sensor data from data.csv."""
    df, numeric_cols = get_processed_sensor_data()
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
    for col in numeric_cols:
        res_data["all_params"][col] = safe_get_col(col, 3)
    
    return jsonify({
        "metrics": metrics,
        "trends": res_data,
        "numeric_columns": numeric_cols.tolist(),
        "total_records": len(df)
    })


@app.route("/api/hydro-data")
def api_hydro_data():
    """Return hydrometric throughput (flow) data from export_hydro_series.csv."""
    start = time.time()
    file_path = "data/export_hydro_series.csv"
    if not os.path.exists(file_path):
        return jsonify({"error": "Hydro data file not found."}), 404
        
    try:
        # Load CSV with semi-colon separator. Skip first two rows (headers)
        df_raw = pd.read_csv(file_path, sep=';', skiprows=2, header=None)
        
        # Assign meaningful names based on view_file output
        df_raw.columns = ['CdSiteHydro', 'CdStationHydro', 'CdCapteur', 'GrdSerieObsHydro', 
                         'DtObsHydro', 'StObsHydro', 'ResObsHydro', 'QualifObsHydro', 
                         'MethObsHydro', 'ContObsHydro', 'FLG']
        
        # Clean quotes and parse dates
        df_raw['DtObsHydro'] = pd.to_datetime(df_raw['DtObsHydro'].astype(str).str.replace('"', ''), errors='coerce')
        
        # Ensure Numeric and Drop NaNs
        df_raw['ResObsHydro'] = pd.to_numeric(df_raw['ResObsHydro'], errors='coerce')
        df_clean = df_raw.dropna(subset=['DtObsHydro', 'ResObsHydro'])
        
        df_clean.set_index('DtObsHydro', inplace=True)
        
        # Filter from 2025-08-01 onwards for speed
        df_filtered = df_clean[df_clean.index >= '2025-08-01']
        
        # Resample daily (mean)
        df_daily = df_filtered['ResObsHydro'].resample('D').mean()
        
        print(f"[DEBUG] Hydro Data Processing took {time.time() - start:.3f}s")
        return jsonify({
            "labels": df_daily.index.strftime("%Y-%m-%d").tolist(),
            "throughput": [round(float(v), 2) if pd.notna(v) else None for v in df_daily.values]
        })
    except Exception as e:
        print(f"Hydro Data Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/farming-events")
def api_farming_events():
    """Return farming events list and active months derived from farming_event dictionary."""
    global _farming_cache
    
    if _farming_cache is None:
        start = time.time()
        from farming_event import get_event_metadata
        _farming_cache = get_event_metadata()
        print(f"[DEBUG] Farming Events cache generated in {time.time() - start:.3f}s")
        
    return jsonify(_farming_cache)

if __name__ == "__main__":
    # Use LITE_MODE=true for skipping heavy libraries
    print(f"Starting HydroLens in {'LITE' if LITE_MODE else 'AI'} mode...")
    
    # Check if we are running in a production or PM2 environment
    is_prod = os.environ.get("DEPLOYMENT") == "true" or os.environ.get("PM2_HOME") is not None

    if is_prod:
        # Waitress for production stability (even on phone)
        from waitress import serve
        print("Production Deployment: Running with Waitress on port 5000...")
        # Use Fewer threads for phone hosting to save memory/CPU
        num_threads = 4 if LITE_MODE else 6
        serve(app, host="0.0.0.0", port=5000, threads=num_threads)
    else:
        # Local development with debugger
        print("Development Mode: Running with Flask Debugger...")
        app.run(host="0.0.0.0", port=5000, debug=True)


