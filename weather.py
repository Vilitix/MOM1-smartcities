import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from datetime import date, timedelta
import matplotlib.pyplot as plt

def get_weather_data(lat=48.693033, lon=6.204775, days=365):
    """
    Fetches weather data from Open-Meteo archive and returns a DataFrame 
    resampled to 3 times a day (8-hour intervals).
    """
    # Variables useful for water quality tracking
    useful_variables = [
        "temperature_2m", 
        "precipitation", 
        "wind_speed_10m",
        "snowfall"
    ]

    # Set up the cache and retry client
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # Set the parameters for the API request
    url = "https://archive-api.open-meteo.com/v1/archive"

    # Start since 2025-08-01 for speed as requested
    end_date = date.today() - timedelta(days=1)
    # Default to Aug 2025 if days isn't specifically restricting it
    start_date = max(date(2025, 8, 1), end_date - timedelta(days=days))

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": useful_variables,
        "timezone": "auto"
    }

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    # Process hourly data
    hourly = response.Hourly()
    hourly_data = {"date": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )}

    for i, var in enumerate(useful_variables):
        hourly_data[var] = hourly.Variables(i).ValuesAsNumpy()

    # Compile into Pandas DataFrame
    df_hourly = pd.DataFrame(data=hourly_data)
    df_hourly.set_index("date", inplace=True)

    # Resample to 3 times a day (8-hour intervals)
    resampling_rules = {
        "temperature_2m": "mean",
        "precipitation": "sum",
        "wind_speed_10m": "mean",
        "snowfall": "sum"
    }
    
    df_resampled = df_hourly.resample('8h').agg(resampling_rules)
    return df_resampled

def get_weather_forecast(lat=48.693033, lon=6.204775, forecast_days=15):
    """
    Fetches weather forecast from Open-Meteo and returns a DataFrame 
    resampled to 3 times a day (8-hour intervals).
    """
    useful_variables = [
        "temperature_2m", 
        "precipitation", 
        "wind_speed_10m",
        "snowfall"
    ]

    # Set up the cache and retry client
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # Set the parameters for the API request
    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": useful_variables,
        "forecast_days": forecast_days,
        "timezone": "auto"
    }

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    # Process hourly data
    hourly = response.Hourly()
    hourly_data = {"date": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )}

    for i, var in enumerate(useful_variables):
        hourly_data[var] = hourly.Variables(i).ValuesAsNumpy()

    # Compile into Pandas DataFrame
    df_hourly = pd.DataFrame(data=hourly_data)
    df_hourly.set_index("date", inplace=True)

    # Resample to 3 times a day (8-hour intervals)
    resampling_rules = {
        "temperature_2m": "mean",
        "precipitation": "sum",
        "wind_speed_10m": "mean",
        "snowfall": "sum"
    }
    
    df_resampled = df_hourly.resample('8h').agg(resampling_rules)
    return df_resampled

def plot_weather_data(df, title="Weather Variables (8h Intervals)", output_plot="weather_plot.png"):
    """Plots the weather variables from the given DataFrame."""
    variables = df.columns
    fig, axes = plt.subplots(nrows=len(variables), ncols=1, figsize=(12, 10), sharex=True)
    fig.suptitle(title, fontsize=16)

    if len(variables) == 1:
        axes = [axes]

    for i, var in enumerate(variables):
        axes[i].plot(df.index, df[var], label=var, color=f"C{i}")
        axes[i].set_ylabel(var)
        axes[i].legend(loc="upper right")
        axes[i].grid(True, linestyle="--", alpha=0.6)

    plt.xlabel("Date")
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    plt.savefig(output_plot)
    print(f"Plot saved to {output_plot}")
    plt.close(fig)

if __name__ == "__main__":
    # Nancy coordinates
    LAT = 48.693033 
    LON = 6.204775 

    print("Fetching and processing historical data...")
    df_hist = get_weather_data(lat=LAT, lon=LON)
    
    print("\n--- Usable Historical Data Prepared (3 times per day) ---")
    print(df_hist.head())
    
    plot_weather_data(df_hist, 
                      title="Weather Variables (Last 365 Days, 8h Intervals)",
                      output_plot="historical_weather.png")

    print("\nFetching and processing 15-day forecast...")
    df_forecast = get_weather_forecast(lat=LAT, lon=LON)
    
    print("\n--- Forecast Data Prepared (3 times per day) ---")
    print(df_forecast.head())
    
    plot_weather_data(df_forecast, 
                      title="Weather Forecast (Next 15 Days, 8h Intervals)",
                      output_plot="forecast_weather.png")
