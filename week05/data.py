"""准备一次更新示例和 diabetes 回归实验的数据。"""

import numpy as np
import torch
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


HOLDOUT_SIZE = 0.4
HOLDOUT_TEST_SHARE = 0.5


def prepare_step_data():
    """返回实践一的两个 float32 样本。"""
    x = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
    y = torch.tensor([[2.0], [4.0]], dtype=torch.float32)
    return x, y


def prepare_diabetes_data(random_state):
    """划分数据，只用训练集统计量标准化特征和目标。"""
    frame = load_diabetes(as_frame=True, scaled=False).frame
    x_all = frame.drop(columns="target")
    y_all = frame["target"].to_numpy(dtype=float)

    x_train, x_holdout, y_train, y_holdout = train_test_split(
        x_all, y_all, test_size=HOLDOUT_SIZE, random_state=random_state
    )
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_holdout,
        y_holdout,
        test_size=HOLDOUT_TEST_SHARE,
        random_state=random_state,
    )

    scaler = StandardScaler().fit(x_train)
    x_scaled = {
        "train": scaler.transform(x_train),
        "validation": scaler.transform(x_validation),
        "test": scaler.transform(x_test),
    }
    y_raw = {
        "train": y_train,
        "validation": y_validation,
        "test": y_test,
    }
    y_mean = float(y_train.mean())
    y_std = float(y_train.std(ddof=0))
    if y_std <= 0:
        raise ValueError("训练目标的标准差必须大于 0")

    tensors = {}
    for split in ("train", "validation", "test"):
        x_tensor = torch.tensor(x_scaled[split], dtype=torch.float32)
        y_scaled = (y_raw[split] - y_mean) / y_std
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32).reshape(-1, 1)
        tensors[split] = (x_tensor, y_tensor)

    return {
        "x_scaled": x_scaled,
        "y_raw": y_raw,
        "tensors": tensors,
        "y_mean": y_mean,
        "y_std": y_std,
    }


def make_train_loader(train_tensors, batch_size, seed):
    """为一组实验新建顺序可复现的训练 DataLoader。"""
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(*train_tensors)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        generator=generator,
    )
