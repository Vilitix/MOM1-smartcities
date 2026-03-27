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

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import joblib # Add this for saving scalers
from weather import get_weather_data


def train_and_predict():
    print("Fetching weather data...")
    df_weather = get_weather_data(days=365)
    df_weather.index = df_weather.index.tz_localize(None)
    df_weather.index.name = 'Datetime'
    df_weather.reset_index(inplace=True)

    print("Loading water quality dataset...")
    df_water = pd.read_csv("Consibio Cloud Datalog.csv")
    try:
        df_water['Datetime'] = pd.to_datetime(df_water['Date'], format='%d/%m-%y %H:%M:%S', exact=False)
    except:
        df_water['Datetime'] = pd.to_datetime(df_water['Date'], format='mixed', dayfirst=True)
    
    df_water.set_index('Datetime', inplace=True)
    target_cols = ['Conductivité', 'NO3', 'Turbidité', 'O2 Saturation', 'pH Test', 'MES']
    
    df_water_aligned = df_water[target_cols].resample('8h').mean()
    df_water_aligned.reset_index(inplace=True)
    
    df_merged = pd.merge(df_water_aligned, df_weather, on='Datetime', how='inner')
    if df_merged.empty: return

    imputer = SimpleImputer(strategy='mean')
    df_merged[target_cols] = imputer.fit_transform(df_merged[target_cols])
    
    # --- Add event flags (fertilizer, snowmelt) with minimal changes ---
    df_merged['flag_fertilizer'] = 0
    df_merged['flag_snowmelt'] = 0
    
    # Include new flags in feature variables
    feature_cols = ['temperature_2m', 'precipitation', 'wind_speed_10m'] + target_cols
    X = df_merged[feature_cols].values
    y = df_merged[target_cols].values
    # -------------------------------------------------------------------
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    # --- Save scalers for backend inference ---
    joblib.dump(scaler_X, 'scaler_X.pkl')
    joblib.dump(scaler_y, 'scaler_y.pkl')
    # ----------------------------------------
    
    SEQ_LENGTH = 90 
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQ_LENGTH)
    
    split_idx = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    
    input_size = X.shape[1]
    hidden_size = 64
    num_layers = 2
    output_size = y.shape[1]
    
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

    # --- Save the trained model ---
    torch.save(model.state_dict(), 'lstm_model.pth')
    print("Model training complete and saved.")
    # ------------------------------

if __name__ == "__main__":
    train_and_predict()