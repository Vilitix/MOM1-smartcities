import pandas as pd
import numpy as np
from datetime import datetime

def load_and_clean_data(file_path="data.csv"):
    """
    Loads and cleans the sensor data from data.csv.
    Standardizes dates and handles missing values.
    """
    try:
        # Load CSV - only load what we need and use a faster engine if possible
        # Use low_memory=False to avoid DtypeWarning and speed up inference
        df = pd.read_csv(file_path, low_memory=False)
        
        # Convert 'Date' column to datetime - Specify format for 10x speed up
        df['datetime'] = pd.to_datetime(df['Date'], format='%d/%m-%y %H:%M:%S', errors='coerce')
        
        # Drop rows where date couldn't be parsed
        df = df.dropna(subset=['datetime'])
        df.set_index('datetime', inplace=True)
        
        # Filter data from August 2025 onwards for performance optimization
        df = df[df.index >= '2025-08-01']
        
        # Drop the original string 'Date' and 'Timestamp' columns
        cols_to_drop = [c for c in ['Date', 'Timestamp'] if c in df.columns]
        df = df.drop(columns=cols_to_drop)

        # Convert entire dataframe to numeric in one go (vectorized)
        df = df.apply(pd.to_numeric, errors='coerce')
        
        numeric_cols = df.columns
        return df, numeric_cols
    except Exception as e:
        print(f"Error in data_handler: {e}")
        return pd.DataFrame(), []

def get_latest_sensor_metrics(df):
    """Returns the most recent reading for key metrics."""
    if df.empty:
        return {}

    def safe_val(val, ndigits=2):
        try:
            v = float(val)
            return round(v, ndigits) if pd.notna(v) else None
        except:
            return None

    # Forward fill to ensure we get the latest non-null readings if the very last row has missing metrics
    latest = df.ffill().iloc[-1]
    return {
        "ph": safe_val(latest.get("pH Test")),
        "turbidity": safe_val(latest.get("Turbidité"), 3),
        "conductivity": safe_val(latest.get("Conductivité"), 1),
        "oxygen_temp": safe_val(latest.get("O2 Temperature"), 1),
        "battery": safe_val(latest.get("Total battery charge #6f0d"), 0),
        "timestamp": df.index[-1].strftime("%Y-%m-%d %H:%M:%S")
    }

def get_resampled_sensor_data(df, interval='8h'):
    """Resamples the sensor data to match the weather data interval."""
    if df.empty:
        return pd.DataFrame()
    
    # Filter only numeric columns explicitly before resampling
    resampled = df.select_dtypes(include=[np.number]).resample(interval).mean()
    return resampled

if __name__ == "__main__":
    df, cols = load_and_clean_data()
    if not df.empty:
        print("Data loaded successfully.")
        print(f"Columns: {cols.tolist()}")
        print("\nLatest Metrics:")
        print(get_latest_sensor_metrics(df))
        
        resampled = get_resampled_sensor_data(df)
        print("\nResampled Data Head (8h):")
        print(resampled.head())
