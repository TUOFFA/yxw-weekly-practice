"""训练自注意力模型并保留验证 MSE 最小时的参数。"""

import copy

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from evaluation import evaluate_mse


def probe_first_update(model, batch_X, batch_y, optimizer, loss_function):
    """记录第一批 backward 前后和 step 前后的梯度与参数状态。"""
    before = model.value.weight.detach().clone()
    optimizer.zero_grad()
    loss = loss_function(model(batch_X), batch_y)
    loss.backward()
    before_step = model.value.weight.detach().clone()
    gradient_norms = {
        name: float(layer.weight.grad.norm().item())
        for name, layer in [
            ("query", model.query),
            ("key", model.key),
            ("value", model.value),
        ]
    }
    gradient = model.value.weight.grad
    nonzero = torch.nonzero(gradient != 0, as_tuple=False)
    if len(nonzero) == 0:
        raise ValueError("V 层首批梯度意外全为 0")
    row, column = nonzero[0].tolist()
    gradient_value = float(gradient[row, column].item())
    optimizer.step()
    after_step = model.value.weight.detach().clone()
    return {
        "loss": float(loss.item()),
        "gradient_norms": gradient_norms,
        "value_weight_changed_before_step": not torch.equal(before, before_step),
        "value_weight_changed_after_step": not torch.equal(before, after_step),
        "observed_index": [row, column],
        "observed_gradient": gradient_value,
        "observed_parameter_before": float(before[row, column].item()),
        "observed_parameter_after": float(after_step[row, column].item()),
    }


def train_model(model, train_data, validation_data, epochs, batch_size,
                learning_rate, seed):
    """训练 40 轮并返回历史、最佳状态、最佳轮次和首批观察。"""
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_data["X"]), torch.from_numpy(train_data["y"])),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        generator=generator,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
    loss_function = nn.MSELoss()
    best_state = None
    best_epoch = None
    best_validation_mse = float("inf")
    history = []
    first_update = None

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_number, (batch_X, batch_y) in enumerate(loader):
            if epoch == 1 and batch_number == 0:
                first_update = probe_first_update(
                    model, batch_X, batch_y, optimizer, loss_function
                )
                continue
            optimizer.zero_grad()
            loss = loss_function(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()

        train_mse = evaluate_mse(model, train_data["X"], train_data["y"], batch_size)
        validation_mse = evaluate_mse(
            model, validation_data["X"], validation_data["y"], batch_size
        )
        history.append({
            "epoch": epoch,
            "train_mse": train_mse,
            "val_mse": validation_mse,
        })
        if validation_mse < best_validation_mse:
            best_validation_mse = validation_mse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return history, best_state, best_epoch, first_update
