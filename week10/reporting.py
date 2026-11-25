"""保存实验表格并绘制对照图。"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_table(rows: list[dict], path: Path) -> None:
    """把记录保存为 UTF-8 长表。"""
    pd.DataFrame(rows).to_csv(
        path, index=False, encoding="utf-8-sig", float_format="%.9f"
    )


def save_json(data: dict, path: Path) -> None:
    """把程序核对结果保存为 JSON。"""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def plot_head_comparison(
    hours,
    same_head1,
    same_head2,
    independent_head1,
    independent_head2,
    target_time: str,
    same_difference: float,
    independent_difference: float,
    path: Path,
) -> None:
    """并排绘制同参和独立参数双头的末行权重。"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True, sharey=True)
    panels = [
        (axes[0], same_head1, same_head2, f"Copied parameters | max diff={same_difference:.2e}"),
        (
            axes[1],
            independent_head1,
            independent_head2,
            f"Independent parameters | max diff={independent_difference:.2e}",
        ),
    ]
    for axis, head1, head2, title in panels:
        axis.plot(hours, head1, marker="o", markersize=3, label="Head 1")
        axis.plot(hours, head2, marker="s", markersize=3, label="Head 2")
        axis.set(
            title=title,
            xlabel="Hour in the 24-hour window",
            ylabel="Attention weight",
            xticks=[1, 4, 8, 12, 16, 20, 24],
        )
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle(f"Two-head comparison | target {target_time}")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_loss_curves(history: list[dict], best_epoch: int, path: Path) -> None:
    """绘制标准化训练和验证 MSE。"""
    frame = pd.DataFrame(history)
    fig, axis = plt.subplots(figsize=(8, 4.8))
    axis.plot(frame["epoch"], frame["train_mse"], label="Train MSE")
    axis.plot(frame["epoch"], frame["val_mse"], label="Validation MSE")
    axis.axvline(best_epoch, color="black", linestyle="--", label=f"Best epoch {best_epoch}")
    axis.set(
        xlabel="Epoch",
        ylabel="Standardized MSE",
        title="One-layer Transformer encoder training",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
