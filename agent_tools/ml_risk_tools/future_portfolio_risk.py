"""
LSTM-Based Portfolio Direction Prediction Tool
==============================================
Retrieves a pre-trained LSTM model to predict whether a portfolio will move UP or DOWN.

Design:
  - Pre-trained LSTM model loaded from persistent storage
  - Binary classification output: {UP, DOWN}
  - Lightweight preprocessing aligned with training pipeline

Functionality:
  - Accepts portfolio 
  - Applies normalization/scaling consistent with training
  - Feeds processed sequence into LSTM model
  - Outputs directional prediction (UP or DOWN)

Assumptions:
  - Input portfolio data is clean and properly formatted
  - Feature schema is consistent with training data
  - No missing critical features at inference time

Model:
  - Architecture: Long Short-Term Memory (LSTM)
  - Task: Binary classification (directional movement)
  - Inference-only usage within agent pipeline

"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import datetime
from agent_tools import fetch_price_data
from agent_tools import calculate_returns

# Initialize the model
# LSTM
class LSTMModel(nn.Module):
    """
    LSTM for portfolio time series forecasting.

    This model processes sequential financial return data and produces:
    1. A regression output predicting future volatility.
    2. A binary classification output predicting market direction (up/down).

    Inputs:
        x (torch.Tensor): Shape (batch_size, seq_len, input_size)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - vol_output: Predicted volatility (batch_size, 1)
            - dir_output: Predicted probability of upward movement (batch_size, 1)

    """
    def __init__(self, input_size, hidden_size1=64, hidden_size2=32):
        super(LSTMModel, self).__init__()

        # LSTM layers
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)

        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)

        # Output heads
        self.volatility_head = nn.Linear(hidden_size2, 1)   # regression
        self.direction_head = nn.Linear(hidden_size2, 1)    # binary classification

        bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, x):
        # x: (batch, seq_len, input_size)

        out, _ = self.lstm1(x)
        out = self.dropout1(out)

        out, _ = self.lstm2(out)

        # Take last timestep
        out = out[:, -1, :]

        out = self.dropout2(out)

        vol_output = self.volatility_head(out)
        dir_output = torch.sigmoid(self.direction_head(out))

        return vol_output, dir_output
    

class PortfolioDataset(Dataset):
    
    def __init__(self, X, y_vol, y_dir):
        self.X = X
        self.y_vol = y_vol
        self.y_dir = y_dir

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_vol[idx], self.y_dir[idx]

model = LSTMModel(input_size=1)

threshold = 0.5

def portfolio_to_lstm_input(portfolio, window=60):
        """
        Convert a portfolio definition into LSTM-ready input tensor(s).

        This function:
            1. Normalizes portfolio weights.
            2. Fetches historical price data for given tickers.
            3. Computes log returns.
            4. Extracts the most recent time window.
            5. Normalizes the window for model input.

        Args:
            portfolio (dict):
                Dictionary containing:
                    - "tickers" (list[str]): List of asset tickers
                    - "weights" (list[float]): Corresponding portfolio weights
            window (int, optional): Number of timesteps to include. Default is 60.

        Returns:
            List[np.ndarray]:
                A list containing one LSTM input array of shape (1, window, 1)

        """
        X_inputs = []
        tickers = portfolio["tickers"]
        weights = np.array(portfolio["weights"], dtype=float)

        # Normalize weights (handles % like 50,30,20)
        weights = weights / weights.sum()

        # Convert to dict format if needed elsewhere; RE: no need for now bah!
        # tickers_weights = dict(zip(tickers, weights))

        # Fetch + compute
        prices = fetch_price_data(
            tickers,
            start="2020-01-01",
            end=str(datetime.date.today())
        )

        returns = calculate_returns(prices, method="log")

        # Weighted portfolio return (single series)
        returns = returns[tickers]
        portfolio_returns = returns.dot(weights)

        recent_window = portfolio_returns[-window:]

        # Normalize
        recent_window = (
            recent_window - np.mean(recent_window)
                ) / (np.std(recent_window) + 1e-8)
        
        X_input = recent_window.reshape(1, window, 1)

        X_inputs.append(X_input)
        return X_inputs

    

def future_portfolio_risk(portfolio, window=60):
    """
    Predict future portfolio volatility and direction using a trained LSTM model.

    This function:
        1. Ensures input is a torch tensor.
        2. Loads pretrained model weights.
        3. Runs inference in evaluation mode.
        4. Computes:
            - Volatility prediction (regression)
            - Direction prediction (binary classification)
            - Probability of upward movement
            - Confidence score

    Args:
        X_input (np.ndarray or torch.Tensor):
            Input tensor of shape (batch_size, seq_len, input_size)

    Returns:
        dict:
            {
                "predicted_volatility": float,
                "predicted_direction": str,    # "Up" or "Down"
                "confidence": float,           # range [0, 1]
                "prob_up": float               # probability of upward movement
            }


    """

    # i put portfolio_to_lstm_input here instead so i can just export future_portfolio_risk
    X_input = portfolio_to_lstm_input(portfolio, window=60)

    all_preds = []
    all_targets = []
    all_probs = []
    # Ensure input is tensor
    if not torch.is_tensor(X_input):
        X_input = torch.tensor(X_input[0], dtype=torch.float32)  # unwrap list → (1, window, 1)
        
    # Load the saved state_dict
    model.load_state_dict(torch.load('../LSTM/model_weights.pth'))

    # Set the model to 
    model.eval()
    with torch.no_grad():
        vol_pred, dir_pred = model(X_input)
        
        vol = vol_pred
        y_dir = dir_pred.view(-1, 1)


        # Apply sigmoid before threshold
        prob_up = torch.sigmoid(dir_pred)
        preds = (prob_up > threshold).float()

        print(preds)
        direction = "Up" if preds.item() == 1.0 else "Down"

    # Confidence = probability distance from 0.5
    confidence = float((abs(prob_up - 0.5) * 2).item())   # scaled to [0,1]

    return {
        "predicted_volatility": float(vol_pred.item()),
        "predicted_direction": direction,
        "confidence": confidence,
        "prob_up": float(prob_up.item())
    }

