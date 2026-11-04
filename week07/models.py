"""定义 LSTM、GRU 预测器和两种基线。"""

import torch
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn


class LSTMPredictor(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0,
            proj_size=0,
        )
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, X):
        sequence_output, (h_n, c_n) = self.lstm(X)
        prediction = self.output(sequence_output[:, -1, :])
        return prediction

    def inspect_forward(self, X):
        sequence_output, (h_n, c_n) = self.lstm(X)
        prediction = self.output(sequence_output[:, -1, :])
        return prediction, sequence_output, h_n, c_n


class GRUPredictor(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0,
        )
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, X):
        sequence_output, h_n = self.gru(X)
        prediction = self.output(sequence_output[:, -1, :])
        return prediction

    def inspect_forward(self, X):
        sequence_output, h_n = self.gru(X)
        prediction = self.output(sequence_output[:, -1, :])
        return prediction, sequence_output, h_n


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def fit_lag_linear(X_train, y_train):
    model = make_pipeline(StandardScaler(), LinearRegression())
    return model.fit(X_train, y_train)


def predict_baseline(name, X, linear_model=None):
    if name == "last_value":
        return X[:, -1]
    return linear_model.predict(X)
