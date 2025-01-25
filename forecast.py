import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from dataframe_nepse import stock_dataFrame

# Define a PyTorch model
class PriceForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PriceForecaster, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Function to prepare the dataset
def prepare_data(df, n_days):
    X, y = [], []
    for i in range(len(df) - n_days):
        # Use 'Close' and 'RSI' of n days as input
        X.append(np.hstack((df['Close'].iloc[i:i+n_days].values, df['RSI'].iloc[i:i+n_days].values)))
        # Forecast the 'Close' of the next day
        y.append(df['Close'].iloc[i + n_days])
    return np.array(X), np.array(y)

# Function to train the model
def forecast_closing_price(df, n_days, epochs=50, hidden_size=64):
    # Ensure RSI column exists
    if 'RSI' not in df.columns:
        raise ValueError("RSI column not found in the dataframe.")

    # Prepare data
    X, y = prepare_data(df, n_days)
    X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    # Define model, loss function, and optimizer
    input_size = X.shape[1]  # n_days * 2 (Close and RSI for each day)
    model = PriceForecaster(input_size=input_size, hidden_size=hidden_size, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs.squeeze(), y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    # Use the trained model to make predictions
    model.eval()
    with torch.no_grad():
        predictions = model(X).squeeze().numpy()

    return model, predictions

# Example usage
# Assuming df has 'Close' and 'RSI' columns
n_days = 5
# df = pd.DataFrame({
#     'Close': np.random.uniform(100, 200, 100),
#     'RSI': np.random.uniform(30, 70, 100),
# })

df = stock_dataFrame("ADBL")
print("****************")
print(df)
model, predictions = forecast_closing_price(df, n_days)

# Append predictions to the dataframe for visualization
df['Prediction'] = np.nan
df['Prediction'].iloc[n_days:] = predictions

print(df.tail())
