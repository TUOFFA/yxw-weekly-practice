"""在无参数更新的情况下完成预测和指标计算。"""

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error


def scaled_mse(model, tensors, loss_fn):
    """评价完整数据在标准化目标上的 MSE。"""
    model.eval()
    x, y = tensors
    with torch.no_grad():
        return float(loss_fn(model(x), y).item())


def predict_mlp_raw(model, x, y_mean, y_std):
    """将 MLP 的标准化预测恢复到原始目标单位。"""
    model.eval()
    with torch.no_grad():
        pred_scaled = model(x).cpu().numpy().reshape(-1)
    return pred_scaled * y_std + y_mean


def regression_metrics(y_true, y_pred):
    """计算原始目标单位下的 MAE、RMSE 和 MSE。"""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
    }


def evaluate_mlp(model, data, split):
    """评价一个已恢复最佳参数的 MLP。"""
    x = data["tensors"][split][0]
    prediction = predict_mlp_raw(model, x, data["y_mean"], data["y_std"])
    return regression_metrics(data["y_raw"][split], prediction)


def evaluate_baseline(model, data, split):
    """评价均值基线或 sklearn 线性回归。"""
    x = data["x_scaled"][split]
    prediction = np.asarray(model.predict(x)).reshape(-1)
    return regression_metrics(data["y_raw"][split], prediction)
