"""实践二入口：组织基线、两组 MLP 实验、指标和图表。"""

import copy
from pathlib import Path

import torch

from data import make_train_loader, prepare_diabetes_data
from evaluation import evaluate_baseline, evaluate_mlp
from models import create_mlp, forward_shapes
from optimizers import create_optimizer
from reporting import (
    plot_loss_curves,
    plot_rmse_comparison,
    save_history,
    save_metrics,
)
from training import fit_baselines, train_mlp


RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS = 200
INPUT_DIM = 10
HIDDEN_DIM = 16
PRACTICE_HIDDEN_DIM = 8
OUTPUT_DIR = Path(__file__).parent / "results"
EXPERIMENTS = (
    {"name": "MLP + SGD", "optimizer": "SGD", "learning_rate": 0.01},
    {"name": "MLP + Adam", "optimizer": "Adam", "learning_rate": 0.001},
)


def metric_row(model_name, learning_rate=None, best_epoch=None):
    return {
        "model": model_name,
        "learning_rate": learning_rate,
        "best_epoch": best_epoch,
        "train_MAE": None,
        "train_RMSE": None,
        "validation_MAE": None,
        "validation_RMSE": None,
        "test_MAE": None,
        "test_RMSE": None,
        "selected_mlp": False,
        "metric_scale": "original target units",
    }


def add_split_metrics(row, split, metrics):
    row[f"{split}_MAE"] = metrics["MAE"]
    row[f"{split}_RMSE"] = metrics["RMSE"]


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    data = prepare_diabetes_data(RANDOM_STATE)

    torch.manual_seed(RANDOM_STATE)
    initial_model = create_mlp(INPUT_DIM, HIDDEN_DIM)
    sample_batch = data["tensors"]["train"][0][:BATCH_SIZE]
    assert forward_shapes(initial_model, sample_batch) == [
        (BATCH_SIZE, 10),
        (BATCH_SIZE, 16),
        (BATCH_SIZE, 16),
        (BATCH_SIZE, 1),
    ]
    practice_model = create_mlp(INPUT_DIM, PRACTICE_HIDDEN_DIM)
    assert forward_shapes(practice_model, sample_batch) == [
        (BATCH_SIZE, 10),
        (BATCH_SIZE, 8),
        (BATCH_SIZE, 8),
        (BATCH_SIZE, 1),
    ]

    trained_models = {}
    best_epochs = {}
    history_rows = []
    for config in EXPERIMENTS:
        model = copy.deepcopy(initial_model)
        loader = make_train_loader(data["tensors"]["train"], BATCH_SIZE, RANDOM_STATE)
        optimizer = create_optimizer(
            model.parameters(), config["optimizer"], config["learning_rate"]
        )
        history, best_epoch = train_mlp(
            model,
            loader,
            optimizer,
            data["tensors"],
            EPOCHS,
            config["name"],
            config["learning_rate"],
        )
        trained_models[config["name"]] = model
        best_epochs[config["name"]] = best_epoch
        history_rows.extend(history)

    metric_rows = []
    baseline_models = fit_baselines(data)
    for name, model in baseline_models.items():
        row = metric_row(name)
        for split in ("train", "validation"):
            add_split_metrics(row, split, evaluate_baseline(model, data, split))
        metric_rows.append(row)

    for config in EXPERIMENTS:
        name = config["name"]
        row = metric_row(name, config["learning_rate"], best_epochs[name])
        for split in ("train", "validation"):
            add_split_metrics(row, split, evaluate_mlp(trained_models[name], data, split))
        metric_rows.append(row)

    mlp_rows = metric_rows[2:]
    selected_row = min(mlp_rows, key=lambda row: row["validation_RMSE"])
    selected_row["selected_mlp"] = True

    for row in metric_rows[:2]:
        add_split_metrics(row, "test", evaluate_baseline(baseline_models[row["model"]], data, "test"))
    add_split_metrics(
        selected_row,
        "test",
        evaluate_mlp(trained_models[selected_row["model"]], data, "test"),
    )

    save_history(history_rows, OUTPUT_DIR / "history.csv")
    save_metrics(metric_rows, OUTPUT_DIR / "metrics.csv")
    plot_loss_curves(history_rows, best_epochs, OUTPUT_DIR / "loss_curves.png")
    plot_rmse_comparison(metric_rows, OUTPUT_DIR / "rmse_comparison.png")


if __name__ == "__main__":
    main()
