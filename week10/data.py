"""读取 ETTh1，构造 24 小时窗口并按目标时间划分。"""

from pathlib import Path

import numpy as np
import pandas as pd


def load_series(path: Path, length: int) -> tuple[pd.DataFrame, np.ndarray]:
    """读取并检查指定长度的小时油温序列。"""
    frame = pd.read_csv(path, parse_dates=["date"]).iloc[:length].copy()
    if len(frame) != length:
        raise ValueError("ETTh1 行数不足")
    if not frame["date"].is_monotonic_increasing or frame["date"].duplicated().any():
        raise ValueError("date 必须严格递增且不能重复")
    if not frame["date"].diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError("相邻记录必须相差 1 小时")
    values = pd.to_numeric(frame["OT"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("OT 必须都是有效数值")
    return frame, values


def make_windows(values: np.ndarray, times: np.ndarray, window_size: int) -> dict:
    """把连续序列变为过去 24 小时预测下一小时的样本。"""
    target_index = np.arange(window_size, len(values))
    return {
        "X": np.asarray([values[index - window_size:index] for index in target_index]),
        "y": values[target_index],
        "target_index": target_index,
        "target_time": times[target_index],
    }


def split_windows(windows: dict, boundaries: list[tuple[str, str, str]]) -> dict:
    """按预测目标时间划分窗口。"""
    splits = {}
    for name, start, end in boundaries:
        mask = (
            (windows["target_time"] >= np.datetime64(start))
            & (windows["target_time"] < np.datetime64(end))
        )
        splits[name] = {key: values[mask] for key, values in windows.items()}
    return splits


def training_statistics(frame: pd.DataFrame, start: str, end: str) -> tuple[float, float]:
    """只用训练时段的原始观测估计标准化统计量。"""
    mask = (frame["date"] >= pd.Timestamp(start)) & (frame["date"] < pd.Timestamp(end))
    train_values = frame.loc[mask, "OT"].to_numpy(dtype=float)
    mean = float(train_values.mean())
    std = float(train_values.std(ddof=0))
    if std <= 0:
        raise ValueError("训练集标准差必须大于 0")
    return mean, std


def standardize_splits(splits: dict, mean: float, std: float) -> dict:
    """用训练统计量变换全部划分。"""
    standardized = {}
    for name, part in splits.items():
        standardized[name] = {
            "X": ((part["X"] - mean) / std).astype(np.float32)[..., None],
            "y": ((part["y"] - mean) / std).astype(np.float32)[:, None],
            "target_index": part["target_index"],
            "target_time": part["target_time"],
        }
    return standardized
