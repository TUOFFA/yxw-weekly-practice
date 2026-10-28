"""读取 ETTh1、构造序列窗口并按目标时刻划分。"""

import numpy as np
import pandas as pd


def load_etth1(path):
    """读取数据并解析时间列。"""
    return pd.read_csv(path, parse_dates=["date"])


def check_data(df):
    """检查时间和数值列是否满足实验要求。"""
    if not df["date"].is_monotonic_increasing:
        raise ValueError("date 必须递增")
    if df["date"].duplicated().any():
        raise ValueError("date 不能重复")
    if not df["date"].diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError("相邻记录必须相差 1 小时")
    if df[["OT", "HUFL"]].isna().any().any():
        raise ValueError("OT 和 HUFL 不能有缺失值")


def make_windows(series, window_size):
    """用连续历史值构造下一时刻预测样本。"""
    if window_size <= 0 or window_size >= len(series):
        raise ValueError("window_size 必须大于 0 且小于序列长度")

    X = []
    y = []
    target_index = []
    for index in range(window_size, len(series)):
        X.append(series[index - window_size:index])
        y.append(series[index])
        target_index.append(index)
    return np.asarray(X), np.asarray(y), np.asarray(target_index)


def split_windows(X, y, target_index, times, boundaries):
    """根据目标时刻生成训练、验证和测试数据。"""
    target_times = times[target_index]
    splits = {}
    for name, start, end in boundaries:
        mask = (target_times >= np.datetime64(start)) & (target_times < np.datetime64(end))
        splits[name] = {
            "X": X[mask],
            "y": y[mask],
            "target_index": target_index[mask],
            "target_time": target_times[mask],
        }
    return splits
