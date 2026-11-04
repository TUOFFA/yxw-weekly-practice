"""保存实验表格并绘制损失曲线和预测曲线。"""

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FIGURE_DPI = 150


def save_table(rows, path, float_format="%.6f"):
    pd.DataFrame(rows).to_csv(
        path, index=False, encoding="utf-8-sig", float_format=float_format
    )


def plot_loss_curves(history_rows, best_epochs, model_label, path):
    table = pd.DataFrame(history_rows)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for axis, hidden_size in zip(axes, [16, 32]):
        part = table[table["hidden_size"] == hidden_size]
        axis.plot(part["epoch"], part["train_mse"], label="Train MSE")
        axis.plot(part["epoch"], part["val_mse"], label="Validation MSE")
        axis.axvline(
            best_epochs[hidden_size], color="black", linestyle="--", linewidth=1,
            label=f"Best epoch: {best_epochs[hidden_size]}",
        )
        axis.set_title(f"{model_label} hidden size = {hidden_size}")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("MSE (standardized scale)")
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=FIGURE_DPI)
    plt.close(figure)


def plot_validation_predictions(rows, start, end, chosen_model, model_label, path):
    table = pd.DataFrame(rows)
    table["target_time"] = pd.to_datetime(table["target_time"])
    models = ["last_value", "lag_linear", chosen_model]
    week = table[
        (table["split"] == "validation")
        & (table["model"].isin(models))
        & (table["target_time"] >= pd.Timestamp(start))
        & (table["target_time"] < pd.Timestamp(end))
    ]
    truth = week.drop_duplicates("target_time").sort_values("target_time")

    figure, axis = plt.subplots(figsize=(13, 5.5))
    axis.plot(
        truth["target_time"], truth["y_true"], color="black", linewidth=2,
        label="Actual OT",
    )
    for model in models:
        part = week[week["model"] == model].sort_values("target_time")
        axis.plot(part["target_time"], part["y_pred"], linewidth=1.1, label=model)
    axis.set_xlabel("Target time")
    axis.set_ylabel("OT (original scale)")
    axis.set_title(f"Validation first-week predictions with {model_label}")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=FIGURE_DPI)
    plt.close(figure)
