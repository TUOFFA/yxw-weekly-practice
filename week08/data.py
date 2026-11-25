"""读取 ETTh1，构造 24 小时窗口并按目标时间划分。"""

import numpy as np
import pandas as pd


def load_series(path, length):
    """读取并检查指定长度的小时油温序列。"""
    df = pd.read_csv(path, parse_dates=["date"]).iloc[:length].copy()
    if len(df) != length:
        raise ValueError("ETTh1 行数不足")
    if not df["date"].is_monotonic_increasing or df["date"].duplicated().any():
        raise ValueError("date 必须严格递增且不能重复")
    if not df["date"].diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError("相邻记录必须相差 1 小时")
    values = pd.to_numeric(df["OT"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("OT 必须都是有效数值")
    return df, values


def make_windows(values, times, window_size):
    """把连续序列变为过去 24 小时预测下一小时的样本。"""
    target_index = np.arange(window_size, len(values))
    return {
        "X": np.asarray([
            values[index - window_size:index] for index in target_index
        ]),
        "y": values[target_index],
        "target_index": target_index,
        "target_time": times[target_index],
    }


def split_windows(windows, boundaries):
    """按预测目标时间划分窗口，避免边界含义含糊。"""
    splits = {}
    for name, start, end in boundaries:
        mask = (
            (windows["target_time"] >= np.datetime64(start))
            & (windows["target_time"] < np.datetime64(end))
        )
        splits[name] = {key: values[mask] for key, values in windows.items()}
    return splits


def training_statistics(df, start, end):
    """只用原始训练时段内每个观测一次来估计标准化统计量。"""
    mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] < pd.Timestamp(end))
    train_values = df.loc[mask, "OT"].to_numpy(dtype=float)
    mean = train_values.mean()
    std = train_values.std(ddof=0)
    if std <= 0:
        raise ValueError("训练集标准差必须大于 0")
    return mean, std


def standardize_splits(splits, mean, std):
    """用训练统计量变换全部划分。"""
    result = {}
    for name, part in splits.items():
        result[name] = {
            "X": ((part["X"] - mean) / std).astype(np.float32)[..., None],
            "y": ((part["y"] - mean) / std).astype(np.float32)[:, None],
            "target_index": part["target_index"],
            "target_time": part["target_time"],
        }
    return result
