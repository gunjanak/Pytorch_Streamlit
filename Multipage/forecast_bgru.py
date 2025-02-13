import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd



class PriceForecasterGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=4, dropout=0.3):
        super(PriceForecasterGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            bidirectional=True, 
            dropout=dropout if num_layers > 1 else 0  # Dropout is ignored for a single layer
        )
        self.dropout = nn.Dropout(dropout)  # Dropout before the fully connected layer
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Multiply by 2 for bidirectional

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # Initialize hidden state
        out, _ = self.gru(x, h0)
        out = self.dropout(out[:, -1, :])  # Apply dropout before the fully connected layer
        out = self.fc(out)
        return out


# Function to prepare the dataset for sequence input
def prepare_data(df, n_days):
    X, y = [], []
    for i in range(len(df) - n_days):
        # Use 'Close' and 'RSI' of n days as input
        X.append(np.column_stack((df['Close'].iloc[i:i+n_days].values, df['RSI'].iloc[i:i+n_days].values)))
        # Forecast the 'Close' of the next day
        y.append(df['Close'].iloc[i + n_days])
    return np.array(X), np.array(y)

# Function to train the model
def train_gru_model(df, n_days,model_path,epochs=500, hidden_size=64, num_layers=1):
    """
    Train a GRU model to forecast closing prices.

    Args:
        df (DataFrame): Input dataframe containing 'Close' and 'RSI' columns.
        n_days (int): Number of days for input sequence.
        epochs (int): Number of training epochs.
        hidden_size (int): Number of GRU units.
        num_layers (int): Number of GRU layers.

    Returns:
        model (nn.Module): Trained GRU model.
        mape (float): Mean Absolute Percentage Error on training data.
    """
    # Ensure RSI column exists
    if 'RSI' not in df.columns:
        raise ValueError("RSI column not found in the dataframe.")

    # Prepare data
    X, y = prepare_data(df, n_days)
    X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    # Define model, loss function, and optimizer
    input_size = 2  # Two features: 'Close' and 'RSI'
    
    # model = PriceForecasterGRU(input_size=input_size, hidden_size=hidden_size, output_size=1, num_layers=num_layers)
    model = PriceForecasterGRU(input_size=input_size, hidden_size=64,
                               output_size=1, num_layers=2,
                               dropout=0.3)
    try:
        model.load_state_dict(torch.load(model_path))
        print("Model successfully loaded")
    except Exception as e:
        print(f"Exception occured: {e}")
        
    
    criterion = nn.MSELoss()
    
    
    optimizer = optim.Adam(model.parameters(),
                           lr=0.001, weight_decay=1e-4)  # L2 Regularization (weight decay)


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

    # Calculate MAPE on training data
    model.eval()
    with torch.no_grad():
        predictions = model(X).squeeze()
        mape = torch.mean(torch.abs((predictions - y) / y) * 100).item()

    return model, mape


def test_gru_model(model, df, n_days):
    """
    Use a trained GRU model to make predictions and calculate MAPE.

    Args:
        model (nn.Module): Trained GRU model.
        df (DataFrame): Input dataframe containing 'Close' and 'RSI' columns.
        n_days (int): Number of days for input sequence.

    Returns:
        predictions (numpy.ndarray): Model predictions for the input data.
        mape (float): Mean Absolute Percentage Error on testing data.
    """
    # Ensure RSI column exists
    if 'RSI' not in df.columns:
        raise ValueError("RSI column not found in the dataframe.")

    # Prepare data
    X, y = prepare_data(df, n_days)
    X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    # Predict with the trained model
    model.eval()
    with torch.no_grad():
        predictions = model(X).squeeze()
        mape = torch.mean(torch.abs((predictions - y) / y) * 100).item()

    return predictions.numpy(), mape

