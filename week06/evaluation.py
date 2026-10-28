"""计算预测指标并组装结果记录。"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_metrics(y_true, y_pred):
    """计算原始油温尺度的 MAE 和 RMSE。"""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
    }


def make_metric_row(model, split, y_true, y_pred, selected):
    """组装一行汇总指标。"""
    scores = calculate_metrics(y_true, y_pred)
    return {
        "model": model,
        "split": split,
        "n_samples": len(y_true),
        "mae": scores["mae"],
        "rmse": scores["rmse"],
        "selected": selected,
    }


def make_prediction_rows(model, split, split_data, y_pred):
    """组装逐目标时刻预测记录。"""
    rows = []
    for index, target_time, y_true, prediction in zip(
        split_data["target_index"], split_data["target_time"], split_data["y"], y_pred
    ):
        rows.append({
            "model": model,
            "split": split,
            "target_index": int(index),
            "target_time": target_time,
            "y_true": y_true,
            "y_pred": prediction,
        })
    return rows


def verify_prediction_rows(rows, metric_rows):
    """用预测长表重算指标并检查重复和汇总一致性。"""
    import pandas as pd

    predictions = pd.DataFrame(rows)
    if predictions.duplicated(["model", "split", "target_index"]).any():
        raise ValueError("预测记录存在重复")

    for metric_row in metric_rows:
        part = predictions[
            (predictions["model"] == metric_row["model"])
            & (predictions["split"] == metric_row["split"])
        ]
        scores = calculate_metrics(part["y_true"], part["y_pred"])
        if len(part) != metric_row["n_samples"]:
            raise ValueError("预测行数与汇总表不一致")
        if not np.isclose(scores["mae"], metric_row["mae"]):
            raise ValueError("MAE 核对失败")
        if not np.isclose(scores["rmse"], metric_row["rmse"]):
            raise ValueError("RMSE 核对失败")
