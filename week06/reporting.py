"""保存实验表格并绘制结果图。"""

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


FIGURE_DPI = 150
TRAIN_COLOR = "#4C78A8"
VALIDATION_COLOR = "#F58518"
BAR_WIDTH = 0.36


def save_table(rows, path, float_format="%.6f"):
    """将普通字典记录保存为 CSV。"""
    pd.DataFrame(rows).to_csv(
        path, index=False, encoding="utf-8-sig", float_format=float_format
    )


def plot_tuning(tuning_rows, path):
    """绘制普通线性回归和 Ridge 候选的训练、验证 RMSE。"""
    table = pd.DataFrame(tuning_rows)
    labels = [
        "lag_linear" if row.model == "lag_linear" else f"ridge α={row.param_value:g}"
        for row in table.itertuples()
    ]
    positions = np.arange(len(table))
    figure, axis = plt.subplots(figsize=(9, 5.2))
    axis.bar(positions - BAR_WIDTH / 2, table["train_rmse"], BAR_WIDTH,
             label="Train RMSE", color=TRAIN_COLOR)
    axis.bar(positions + BAR_WIDTH / 2, table["val_rmse"], BAR_WIDTH,
             label="Validation RMSE", color=VALIDATION_COLOR)
    axis.set_xticks(positions, labels)
    axis.set_ylabel("RMSE (original OT scale)")
    axis.set_title("Training and validation RMSE by configuration")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=FIGURE_DPI)
    plt.close(figure)


def plot_series_split(df, window_size, validation_start, test_start, path):
    """绘制实验序列、时间划分和首个样本的初始历史。"""
    figure, axis = plt.subplots(figsize=(13, 5.2))
    axis.plot(df["date"], df["OT"], linewidth=0.7, color="#4C78A8", label="OT")
    axis.axvspan(df["date"].iloc[0], df["date"].iloc[window_size - 1],
                 color="#72B7B2", alpha=0.35, label="Initial history")
    axis.axvline(pd.Timestamp(validation_start), color="#F58518", linestyle="--",
                 label="Validation start")
    axis.axvline(pd.Timestamp(test_start), color="#E45756", linestyle="--",
                 label="Test start")
    axis.set_xlabel("Time")
    axis.set_ylabel("油温 OT（原始数据尺度）")
    axis.set_title("ETTh1 oil temperature and time splits")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=FIGURE_DPI)
    plt.close(figure)


def plot_validation_predictions(prediction_rows, source_df, start, end, path):
    """绘制验证首周四种预测与 HUFL 负载。"""
    predictions = pd.DataFrame(prediction_rows)
    predictions["target_time"] = pd.to_datetime(predictions["target_time"])
    week = predictions[
        (predictions["split"] == "validation")
        & (predictions["target_time"] >= pd.Timestamp(start))
        & (predictions["target_time"] < pd.Timestamp(end))
    ]
    load = source_df[(source_df["date"] >= pd.Timestamp(start)) &
                     (source_df["date"] < pd.Timestamp(end))]

    figure, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    truth = week.drop_duplicates("target_time").sort_values("target_time")
    axes[0].plot(truth["target_time"], truth["y_true"], color="black",
                 linewidth=2, label="Actual OT")
    for model, rows in week.groupby("model", sort=False):
        rows = rows.sort_values("target_time")
        axes[0].plot(rows["target_time"], rows["y_pred"], linewidth=1.1, label=model)
    axes[0].set_ylabel("油温 OT（原始数据尺度）")
    axes[0].set_title("Validation first-week predictions")
    axes[0].legend(ncol=3)

    axes[1].plot(load["date"], load["HUFL"], color="#59A14F", linewidth=1.2)
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("HUFL（原始数据尺度）")
    axes[1].set_title("HUFL load channel")
    figure.tight_layout()
    figure.savefig(path, dpi=FIGURE_DPI)
    plt.close(figure)
