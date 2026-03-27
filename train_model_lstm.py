import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from weather import get_weather_data

# Define LSTM Model (PyTorch)
class WaterQualityLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(WaterQualityLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Fully Connected Layer (Output Layer)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :]) 
        return out


# Sequence Generator Function
def create_sequences(X_data, y_data, seq_length):
    """
    Converts 2D tabular data into 3D sequential data for LSTM.
    Shape becomes: (Samples, Time_Steps, Features)
    """
    xs, ys = [], []
    for i in range(len(X_data) - seq_length):
        xs.append(X_data[i:(i + seq_length)])
        ys.append(y_data[i + seq_length])
    return np.array(xs), np.array(ys)

def train_and_predict():
    # Fetch and Preprocess Data
    print("Fetching weather data...")
    df_weather = get_weather_data(days=365)
    df_weather.index = df_weather.index.tz_localize(None)
    df_weather.index.name = 'Datetime'
    df_weather.reset_index(inplace=True)

    # # Add dummy Event_Scale
    # df_weather['DayOfWeek'] = df_weather['Datetime'].dt.dayofweek
    # def determine_event_scale(day_of_week):
    #     if day_of_week >= 5: return np.random.choice([0, 1, 2], p=[0.5, 0.4, 0.1])
    #     else: return np.random.choice([0, 1], p=[0.9, 0.1])
    # df_weather['Event_Scale'] = df_weather['DayOfWeek'].apply(determine_event_scale)
    # df_weather.drop(columns=['DayOfWeek'], inplace=True)

    print("Loading water quality dataset...")
    df_water = pd.read_csv("Consibio Cloud Datalog.csv")
    try:
        df_water['Datetime'] = pd.to_datetime(df_water['Date'], format='%d/%m-%y %H:%M:%S', exact=False)
    except:
        df_water['Datetime'] = pd.to_datetime(df_water['Date'], format='mixed', dayfirst=True)
    
    df_water.set_index('Datetime', inplace=True)
    target_cols = ['Conductivité', 'NO3', 'Chlorophylle-a SCALED','Turbidité', 'O2 Saturation', 'pH Test', 'MES']
    
    # Resample to 8-hour intervals
    df_water_aligned = df_water[target_cols].resample('8h').mean()
    df_water_aligned.reset_index(inplace=True)
    
    # Merge
    df_merged = pd.merge(df_water_aligned, df_weather, on='Datetime', how='inner')
    if df_merged.empty: return

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    df_merged[target_cols] = imputer.fit_transform(df_merged[target_cols])
    
    X = df_merged[['temperature_2m', 'precipitation', 'wind_speed_10m']+ target_cols].values
    y = df_merged[target_cols].values
    
    # Scale Data and Create Sequences
    # Neural Networks require standard scaling (mean=0, variance=1)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    # Define sequence length
    # 9 steps = 3 days of data (8h * 9 = 72h)
    SEQ_LENGTH = 9 
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQ_LENGTH)
    
    # Split into train and test sets
    split_idx = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
    
    # Convert to PyTorch Tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    # Create DataLoader
    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    
    # Train the LSTM Model
    input_size = X.shape[1]    # 4 features
    hidden_size = 64           # Number of features in hidden state
    num_layers = 2             # Number of stacked LSTM layers
    output_size = y.shape[1]   # 6 targets
    
    model = WaterQualityLSTM(input_size, hidden_size, num_layers, output_size)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 100
    print(f"Training LSTM for {epochs} epochs...")
    
    model.train()
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
        if (epoch+1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
            
    # Evaluation
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_tensor)
        
    # Inverse transform to get real-world values
    test_predictions_inv = scaler_y.inverse_transform(test_predictions.numpy())
    y_test_inv = scaler_y.inverse_transform(y_test_tensor.numpy())
    
    print("\n--- Model Evaluation (MSE per Target) ---")
    for i, col in enumerate(target_cols):
        mse = mean_squared_error(y_test_inv[:, i], test_predictions_inv[:, i])
        print(f"{col}: {mse:.4f}")

    # Future Prediction Scenario
    # We need the last 'SEQ_LENGTH' (9) steps from our data to predict the future
    last_sequence = X_scaled[-SEQ_LENGTH:]
    # Convert to shape (1, Sequence_Length, Features)
    last_sequence_tensor = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0)
    
    model.eval()
    with torch.no_grad():
        future_pred_scaled = model(last_sequence_tensor)
        
    future_predictions = scaler_y.inverse_transform(future_pred_scaled.numpy())
    
    print("\n--- Future Prediction Scenario (Next 8 Hours) ---")
    for i, col in enumerate(target_cols):
        print(f"Predicted {col}: {future_predictions[0][i]:.4f}")

if __name__ == "__main__":
    train_and_predict()