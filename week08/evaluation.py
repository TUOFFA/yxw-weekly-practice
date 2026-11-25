"""评价自注意力模型并核对三种读法的预测。"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def evaluate_mse(model, X, y, batch_size):
    """计算标准化尺度的逐样本均方误差。"""
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    squared_error = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for batch_X, batch_y in loader:
            prediction = model(batch_X)
            squared_error += torch.sum((prediction - batch_y) ** 2).item()
            count += batch_y.numel()
    return squared_error / count


def predict_attention(model, X, batch_size, mean, std, reading="original"):
    """按指定读法预测并还原到原始油温尺度。"""
    loader = DataLoader(torch.from_numpy(X), batch_size=batch_size, shuffle=False)
    predictions = []
    model.eval()
    with torch.no_grad():
        for batch_X in loader:
            prediction, *_ = model.predict_with_reading(batch_X, reading)
            predictions.append(prediction.numpy())
    standardized = np.concatenate(predictions).reshape(-1)
    return standardized * std + mean


def calculate_metrics(y_true, y_pred):
    """计算原始尺度 MAE 与 RMSE。"""
    errors = np.asarray(y_pred) - np.asarray(y_true)
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
    }
