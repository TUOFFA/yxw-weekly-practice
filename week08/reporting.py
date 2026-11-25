"""保存表格并绘制训练、注意力和路径图。"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch


def save_table(rows, path):
    """把记录保存为便于核对的 UTF-8 长表。"""
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig", float_format="%.9f")


def save_json(data, path):
    """把程序核对结果保存为 JSON。"""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def plot_loss_curves(history, best_epoch, path):
    """绘制标准化训练和验证 MSE。"""
    frame = pd.DataFrame(history)
    fig, axis = plt.subplots(figsize=(8, 4.8))
    axis.plot(frame["epoch"], frame["train_mse"], label="Train MSE")
    axis.plot(frame["epoch"], frame["val_mse"], label="Validation MSE")
    axis.axvline(best_epoch, color="black", linestyle="--", label=f"Best epoch {best_epoch}")
    axis.set(xlabel="Epoch", ylabel="Standardized MSE", title="Single-head attention training")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_scores_and_weights(scores, weights, target_time, score_value,
                            weight_value, row_sum, path):
    """并排绘制同一次前向的 S 与 A，并圈出第 24 行第 3 列。"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    for axis, matrix, title, color_map in [
        (axes[0], scores, "Scores S (before softmax)", "coolwarm"),
        (axes[1], weights, "Attention A (after softmax)", "viridis"),
    ]:
        image = axis.imshow(matrix, aspect="auto", cmap=color_map)
        axis.scatter([2], [23], s=130, facecolors="none", edgecolors="red", linewidths=2)
        axis.set(
            xlabel="Key/value hour",
            ylabel="Query hour",
            xticks=np.arange(0, 24, 3),
            yticks=np.arange(0, 24, 3),
            xticklabels=np.arange(1, 25, 3),
            yticklabels=np.arange(1, 25, 3),
            title=title,
        )
        fig.colorbar(image, ax=axis, shrink=0.84)
    fig.suptitle(
        f"Target {target_time} | S[24,3]={score_value:.6f}, "
        f"A[24,3]={weight_value:.6f}, sum(A[24,:])={row_sum:.6f}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def draw_path_diagram(kind, path):
    """绘制 LSTM 或单头注意力中第 1 小时到预测层的路径。"""
    fig, axis = plt.subplots(figsize=(12, 4.8))
    axis.set_xlim(0, 13.5)
    axis.set_ylim(0, 5)
    axis.axis("off")

    if kind == "lstm":
        nodes = [
            (0.5, 2.1, 1.2, "x1"), (2.3, 2.1, 1.6, "(h1, c1)"),
            (4.6, 2.1, 2.2, "... 22 updates ..."), (7.5, 2.1, 1.8, "(h24, c24)"),
            (10.0, 2.1, 1.4, "Linear"), (12.0, 2.1, 1.0, "y25")
        ]
        subtitle = "Hour 1 must pass through every recurrent state update"
        arrows = [((1.7, 2.5), (2.3, 2.5)), ((3.9, 2.5), (4.6, 2.5)),
                  ((6.8, 2.5), (7.5, 2.5)), ((9.3, 2.5), (10.0, 2.5)),
                  ((11.4, 2.5), (12.0, 2.5))]
    else:
        nodes = [
            (0.5, 3.2, 1.2, "x1"), (2.1, 3.2, 1.2, "k1"),
            (3.8, 3.2, 2.0, "score(q24,k1)"), (6.3, 3.2, 1.6, "A[24,1]"),
            (0.5, 0.9, 1.2, "x1"), (2.1, 0.9, 1.2, "v1"),
            (8.4, 2.1, 2.0, "sum A[24,j] vj"), (10.9, 2.1, 1.3, "Linear"),
            (12.6, 2.1, 0.8, "y25")
        ]
        subtitle = "Hour 1 reaches the last representation through key and value paths"
        arrows = [
            ((1.7, 3.6), (2.1, 3.6)), ((3.3, 3.6), (3.8, 3.6)),
            ((5.8, 3.6), (6.3, 3.6)), ((7.9, 3.6), (8.4, 2.7)),
            ((1.7, 1.3), (2.1, 1.3)), ((3.3, 1.3), (8.4, 2.3)),
            ((10.4, 2.5), (10.9, 2.5)), ((12.2, 2.5), (12.6, 2.5))
        ]
        axis.text(4.8, 4.45, "q24 asks", ha="center", fontsize=11)
        axis.annotate("", xy=(4.8, 4.0), xytext=(4.8, 4.25), arrowprops={"arrowstyle": "->"})

    for x, y, width, label in nodes:
        box = FancyBboxPatch(
            (x, y), width, 0.8, boxstyle="round,pad=0.08",
            facecolor="#f4f4f4", edgecolor="#222222", linewidth=1.2
        )
        axis.add_patch(box)
        axis.text(x + width / 2, y + 0.4, label, ha="center", va="center", fontsize=10)

    for start, end in arrows:
        axis.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.4})
    axis.set_title(subtitle, fontsize=14, pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
