"""评价循环模型，计算指标并核对逐行预测。"""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset


def evaluate_mse(model, X, y, batch_size):
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


def predict_recurrent(model, X, batch_size, mean, std):
    loader = DataLoader(
        torch.from_numpy(X),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    predictions = []
    model.eval()
    with torch.no_grad():
        for batch_X in loader:
            predictions.append(model(batch_X).numpy())
    standardized = np.concatenate(predictions).reshape(-1)
    return standardized * std + mean


def calculate_metrics(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
    }


def verify_results(prediction_rows, metric_rows):
    predictions = pd.DataFrame(prediction_rows)
    if predictions.duplicated(["model", "split", "target_index"]).any():
        raise ValueError("预测记录存在重复")

    for row in metric_rows:
        part = predictions[
            (predictions["model"] == row["model"])
            & (predictions["split"] == row["split"])
        ]
        scores = calculate_metrics(part["y_true"], part["y_pred"])
        if len(part) != row["n_samples"]:
            raise ValueError("预测数量与 metrics.csv 不一致")
        if not np.isclose(scores["mae"], row["mae"]):
            raise ValueError("MAE 核对失败")
        if not np.isclose(scores["rmse"], row["rmse"]):
            raise ValueError("RMSE 核对失败")
