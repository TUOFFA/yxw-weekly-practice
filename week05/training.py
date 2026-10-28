"""执行一次更新、MLP 训练循环和基线拟合。"""

import copy

import torch
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from torch import nn

from evaluation import scaled_mse


def parameter_row(stage, model, loss=None):
    """读取单层线性模型当前的参数和梯度。"""
    weight_grad = model.weight.grad
    bias_grad = model.bias.grad
    return {
        "stage": stage,
        "weight": model.weight.item(),
        "bias": model.bias.item(),
        "weight_grad": None if weight_grad is None else weight_grad.item(),
        "bias_grad": None if bias_grad is None else bias_grad.item(),
        "loss": loss,
    }


def run_single_update(model, optimizer, x, y):
    """执行并记录清梯度、反传和更新三个阶段。"""
    loss_fn = nn.MSELoss()
    optimizer.zero_grad()
    initial_pred = model(x)
    assert initial_pred.shape == y.shape
    initial_loss = loss_fn(initial_pred, y)
    rows = [parameter_row("after_zero_grad", model, initial_loss.item())]

    initial_loss.backward()
    rows.append(parameter_row("after_backward", model))

    old_weight = model.weight.item()
    old_bias = model.bias.item()
    weight_grad = model.weight.grad.item()
    bias_grad = model.bias.grad.item()
    learning_rate = optimizer.param_groups[0]["lr"]
    optimizer.step()

    assert abs(model.weight.item() - (old_weight - learning_rate * weight_grad)) < 1e-6
    assert abs(model.bias.item() - (old_bias - learning_rate * bias_grad)) < 1e-6
    with torch.no_grad():
        updated_loss = loss_fn(model(x), y).item()
    rows.append(parameter_row("after_step", model, updated_loss))
    return rows


def train_one_epoch(model, loader, optimizer, loss_fn):
    """遍历一轮小批量并更新参数。"""
    model.train()
    for xb, yb in loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()


def train_mlp(model, loader, optimizer, tensors, epochs, config_name, learning_rate):
    """训练满指定轮数，按验证 MSE 保存较早的最佳参数。"""
    loss_fn = nn.MSELoss()
    history = []
    best_validation_mse = float("inf")
    best_epoch = None
    best_state = None

    for epoch in range(1, epochs + 1):
        train_one_epoch(model, loader, optimizer, loss_fn)
        train_mse = scaled_mse(model, tensors["train"], loss_fn)
        validation_mse = scaled_mse(model, tensors["validation"], loss_fn)
        history.append(
            {
                "config": config_name,
                "learning_rate": learning_rate,
                "epoch": epoch,
                "train_MSE": train_mse,
                "validation_MSE": validation_mse,
                "target_scale": "standardized",
            }
        )
        if validation_mse < best_validation_mse:
            best_validation_mse = validation_mse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return history, best_epoch


def fit_baselines(data):
    """在同一训练划分上拟合均值基线和线性回归。"""
    x_train = data["x_scaled"]["train"]
    y_train = data["y_raw"]["train"]
    return {
        "Mean baseline": DummyRegressor(strategy="mean").fit(x_train, y_train),
        "LinearRegression": LinearRegression().fit(x_train, y_train),
    }
