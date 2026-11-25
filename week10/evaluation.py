"""评价单层编码器并计算原始油温尺度指标。"""

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from torch.utils.data import DataLoader, TensorDataset


def evaluate_mse(
    model: torch.nn.Module, X: np.ndarray, y: np.ndarray, batch_size: int
) -> float:
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


def predict(
    model: torch.nn.Module,
    X: np.ndarray,
    batch_size: int,
    mean: float,
    std: float,
) -> np.ndarray:
    """批量预测并还原到原始油温尺度。"""
    loader = DataLoader(torch.from_numpy(X), batch_size=batch_size, shuffle=False)
    standardized = []
    model.eval()
    with torch.no_grad():
        for batch_X in loader:
            standardized.append(model(batch_X).numpy())
    return np.concatenate(standardized).reshape(-1) * std + mean


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """使用 sklearn 计算原始尺度 MAE 与 RMSE。"""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
    }
