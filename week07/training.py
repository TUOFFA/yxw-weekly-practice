"""训练循环模型并保留验证 MSE 最小时的参数。"""

import copy

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from evaluation import evaluate_mse


def train_model(model, train_data, validation_data, epochs, batch_size, learning_rate, seed):
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_data["X"]), torch.from_numpy(train_data["y"])),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        generator=generator,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=0
    )
    loss_function = nn.MSELoss()
    best_state = None
    best_epoch = None
    best_validation_mse = float("inf")
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            prediction = model(batch_X)
            loss = loss_function(prediction, batch_y)
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
    return history, best_state, best_epoch
