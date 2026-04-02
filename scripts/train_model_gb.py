import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from weather import get_weather_data

def train_and_predict():
    # Fetch Weather Data 
    print("Fetching weather data using weather.py...")

    df_weather = get_weather_data(days=365)
    
    df_weather.index = df_weather.index.tz_localize(None)
    
    # Rename the index to 'Datetime' so it matches the water dataset later
    df_weather.index.name = 'Datetime'
    df_weather.reset_index(inplace=True)

    # Add dummy Event_Scale feature (quantified events)
    # Weekends have a higher chance of large events, weekdays have fewer
    # We need to add this from real event data
    df_weather['DayOfWeek'] = df_weather['Datetime'].dt.dayofweek
    def determine_event_scale(day_of_week):
        if day_of_week >= 5: # Saturday or Sunday
            return np.random.choice([0, 1, 2], p=[0.5, 0.4, 0.1])
        else: # Weekday
            return np.random.choice([0, 1], p=[0.9, 0.1])
            
    df_weather['Event_Scale'] = df_weather['DayOfWeek'].apply(determine_event_scale)
    df_weather.drop(columns=['DayOfWeek'], inplace=True)

    # Load and Preprocess Water Quality Data
    print("Loading water quality dataset...")
    df_water = pd.read_csv("Consibio Cloud Datalog.csv")
    
    try:
        df_water['Datetime'] = pd.to_datetime(df_water['Date'], format='%d/%m-%y %H:%M:%S', exact=False)
    except:
        df_water['Datetime'] = pd.to_datetime(df_water['Date'], format='mixed', dayfirst=True)
    
    df_water.set_index('Datetime', inplace=True)
    target_cols = ['Chlorophylle-a SCALED', 'Conductivité', 'NO3', 'O2 Saturation', 'pH Test', 'Turbidité']
    
    # CRITICAL STEP: Resample water data to 8-hour intervals to match weather data
    df_water_aligned = df_water[target_cols].resample('8h').mean()
    df_water_aligned.reset_index(inplace=True)
    
    # Merge Datasets
    df_merged = pd.merge(df_water_aligned, df_weather, on='Datetime', how='inner')
    
    if df_merged.empty:
        print("Error: Merged dataset is empty. Check if the date ranges overlap.")
        return

    imputer = SimpleImputer(strategy='mean')
    df_merged[target_cols] = imputer.fit_transform(df_merged[target_cols])
    
    X = df_merged[['temperature_2m', 'precipitation', 'wind_speed_10m', 'Event_Scale']]
    y = df_merged[target_cols]
    
    # Model Training
    # Split the dataset: 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize the Multi-Output Gradient Boosting Regressor
    model = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
    
    print("Training the multi-output model (Gradient Boosting)...")
    model.fit(X_train, y_train)
    
    # Evaluation & Future Prediction
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    print("\n--- Model Evaluation (MSE per Target) ---")
    for i, col in enumerate(target_cols):
        mse = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
        print(f"{col}: {mse:.4f}")

    # Example Scenario: Predicting water quality for an upcoming 8-hour window
    print("\n--- Future Prediction Scenario ---")
    future_X = pd.DataFrame({
        'temperature_2m': [22.0],        # 22°C
        'precipitation': [15.0],         # 15mm of rain
        'wind_speed_10m': [12.5],        # Wind speed 12.5 km/h
        'Event_Scale': [2]               # Large event
    })
    
    future_predictions = model.predict(future_X)
    for i, col in enumerate(target_cols):
        print(f"Predicted {col}: {future_predictions[0][i]:.4f}")

if __name__ == "__main__":
    train_and_predict()