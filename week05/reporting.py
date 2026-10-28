"""保存 CSV，并绘制训练曲线和 RMSE 对照图。"""

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FIGURE_DPI = 150
TRAIN_COLOR = "#4C78A8"
VALIDATION_COLOR = "#F58518"
BAR_WIDTH = 0.36


def save_step_trace(rows, output_path):
    """保存一次更新的三个阶段。"""
    pd.DataFrame(rows).to_csv(
        output_path, index=False, encoding="utf-8-sig", float_format="%.6f"
    )


def save_history(rows, output_path):
    """保存两组各 200 轮的完整记录。"""
    pd.DataFrame(rows).to_csv(
        output_path, index=False, encoding="utf-8-sig", float_format="%.6f"
    )


def save_metrics(rows, output_path):
    """保存四组模型的原始尺度指标。"""
    pd.DataFrame(rows).to_csv(
        output_path, index=False, encoding="utf-8-sig", float_format="%.4f"
    )


def plot_loss_curves(history_rows, best_epochs, output_path):
    """绘制 SGD 与 Adam 的训练、验证标准化 MSE。"""
    history = pd.DataFrame(history_rows)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for axis, config_name in zip(axes, best_epochs):
        rows = history[history["config"] == config_name]
        axis.plot(rows["epoch"], rows["train_MSE"], label="Train MSE", color=TRAIN_COLOR)
        axis.plot(
            rows["epoch"],
            rows["validation_MSE"],
            label="Validation MSE",
            color=VALIDATION_COLOR,
        )
        axis.axvline(
            best_epochs[config_name], color="gray", linestyle="--", label="Best epoch"
        )
        axis.set_title(config_name)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("MSE (standardized target)")
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(figure)


def plot_rmse_comparison(metric_rows, output_path):
    """绘制四组模型的训练与验证 RMSE。"""
    metrics = pd.DataFrame(metric_rows)
    positions = np.arange(len(metrics))
    figure, axis = plt.subplots(figsize=(10, 5.2))
    axis.bar(
        positions - BAR_WIDTH / 2,
        metrics["train_RMSE"],
        BAR_WIDTH,
        label="Train RMSE",
        color=TRAIN_COLOR,
    )
    axis.bar(
        positions + BAR_WIDTH / 2,
        metrics["validation_RMSE"],
        BAR_WIDTH,
        label="Validation RMSE",
        color=VALIDATION_COLOR,
    )
    axis.set_xticks(positions, metrics["model"], rotation=15, ha="right")
    axis.set_ylabel("RMSE (original target units)")
    axis.set_title("Train and validation RMSE")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(figure)
