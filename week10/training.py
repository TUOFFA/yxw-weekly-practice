"""训练单层 Transformer 编码器并保存最佳验证参数。"""

import copy

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from evaluation import evaluate_mse


def probe_first_update(
    model: nn.Module,
    batch_X: torch.Tensor,
    batch_y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
) -> dict:
    """记录首批 backward 与 step 之间的梯度、参数状态。"""
    query_before = model.attention.head1.query.weight.detach().clone()
    predictor_before = model.predictor.weight.detach().clone()
    optimizer.zero_grad()
    loss = loss_function(model(batch_X), batch_y)
    loss.backward()
    query_before_step = model.attention.head1.query.weight.detach().clone()
    predictor_before_step = model.predictor.weight.detach().clone()
    if model.attention.head1.query.weight.grad is None or model.predictor.weight.grad is None:
        raise ValueError("注意力头或预测层没有接入损失计算图")
    result = {
        "loss": float(loss.item()),
        "head1_query_grad_norm": float(
            model.attention.head1.query.weight.grad.norm().item()
        ),
        "predictor_grad_norm": float(model.predictor.weight.grad.norm().item()),
        "query_changed_before_step": not torch.equal(query_before, query_before_step),
        "predictor_changed_before_step": not torch.equal(
            predictor_before, predictor_before_step
        ),
    }
    optimizer.step()
    result["query_changed_after_step"] = not torch.equal(
        query_before, model.attention.head1.query.weight.detach()
    )
    result["predictor_changed_after_step"] = not torch.equal(
        predictor_before, model.predictor.weight.detach()
    )
    return result


def train_model(
    model: nn.Module,
    train_data: dict,
    validation_data: dict,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[list[dict], dict, int, dict]:
    """训练固定轮数并恢复验证 MSE 最低的参数。"""
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
        history.append(
            {"epoch": epoch, "train_mse": train_mse, "val_mse": validation_mse}
        )
        if validation_mse < best_validation_mse:
            best_validation_mse = validation_mse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None or best_epoch is None or first_update is None:
        raise RuntimeError("训练未产生可用结果")
    model.load_state_dict(best_state)
    return history, best_state, best_epoch, first_update
