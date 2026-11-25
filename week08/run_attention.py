"""训练单头模型并完成权重替换与位置交换实验。"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data import load_series, make_windows, split_windows, standardize_splits, training_statistics
from evaluation import calculate_metrics, predict_attention
from models import SingleHeadAttentionPredictor, count_parameters
from reporting import (
    draw_path_diagram,
    plot_loss_curves,
    plot_scores_and_weights,
    save_json,
    save_table,
)
from training import train_model


DATA_PATH = Path("../week06/data/ETTh1.csv")
LSTM_METRICS_PATH = Path("../week07/results/metrics.csv")
OUTPUT_DIR = Path("results")
SERIES_LENGTH = 14400
WINDOW_SIZE = 24
MODEL_DIM = 16
EPOCHS = 40
BATCH_SIZE = 128
LEARNING_RATE = 0.001
RANDOM_SEED = 42
SWAP_HOURS = (3, 18)
BOUNDARIES = [
    ("train", "2016-07-01 00:00", "2017-06-26 00:00"),
    ("validation", "2017-06-26 00:00", "2017-10-24 00:00"),
    ("test", "2017-10-24 00:00", "2018-02-21 00:00"),
]


def original_scale(value, mean, std):
    """把标量标准化预测还原为原始油温。"""
    return float(value * std + mean)


def inspect_window(model, standardized_window, raw_window, target_time,
                   target_value, mean, std):
    """完成指定验证窗口的 S/A/V/Z 与三种读法核对。"""
    inputs = torch.from_numpy(standardized_window[None, ...])
    model.eval()
    with torch.no_grad():
        prediction, embedded, queries, keys, values, scores, weights, contexts = (
            model.inspect_forward(inputs)
        )
        original_prediction, _, _, _, original_context = model.predict_with_reading(inputs, "original")
        mean_prediction, _, _, _, mean_context = model.predict_with_reading(inputs, "mean")
        last_prediction, _, _, _, last_context = model.predict_with_reading(inputs, "last")

    restored_context = weights[:, -1, :] @ values
    checks = {
        "target_time": str(pd.Timestamp(target_time)),
        "window_start": str(pd.Timestamp(target_time) - pd.Timedelta(hours=24)),
        "window_end": str(pd.Timestamp(target_time) - pd.Timedelta(hours=1)),
        "hour_1_ot": float(raw_window[0]),
        "hour_24_ot": float(raw_window[-1]),
        "target_ot": float(target_value),
        "score_24_3": float(scores[0, 23, 2].item()),
        "weight_24_3": float(weights[0, 23, 2].item()),
        "last_weight_sum": float(weights[0, 23].sum().item()),
        "context_recovery_max_abs_error": float((restored_context - contexts[:, -1, :]).abs().max().item()),
        "original_context_recovery_max_abs_error": float((original_context - contexts[:, -1, :]).abs().max().item()),
        "mean_context_check_max_abs_error": float((mean_context - values.mean(dim=1)).abs().max().item()),
        "last_context_check_max_abs_error": float((last_context - values[:, -1, :]).abs().max().item()),
        "prediction_ot": original_scale(prediction.item(), mean, std),
        "original_prediction_ot": original_scale(original_prediction.item(), mean, std),
        "mean_prediction_ot": original_scale(mean_prediction.item(), mean, std),
        "last_prediction_ot": original_scale(last_prediction.item(), mean, std),
    }
    arrays = {
        "scores": scores[0].numpy(),
        "weights": weights[0].numpy(),
        "values": values[0].numpy(),
    }
    return checks, arrays


def inspect_swap(model, standardized_window, raw_window, window_times, mean, std):
    """交换第 3、18 小时，比较无位置与加入位置的末位结果。"""
    first, second = [hour - 1 for hour in SWAP_HOURS]
    if raw_window[first] == raw_window[second]:
        raise ValueError("交换实验需要两个不同油温")
    swapped = standardized_window.copy()
    swapped[[first, second]] = swapped[[second, first]]
    original = torch.from_numpy(standardized_window[None, ...])
    changed = torch.from_numpy(swapped[None, ...])
    rows = []
    comparisons = {}
    model.eval()

    for add_position in [False, True]:
        with torch.no_grad():
            original_result = model.inspect_forward(original, add_position=add_position)
            changed_result = model.inspect_forward(changed, add_position=add_position)
        original_prediction, *original_tensors = original_result
        changed_prediction, *changed_tensors = changed_result
        original_weights = original_tensors[-2][0, -1]
        changed_weights = changed_tensors[-2][0, -1]
        original_context = original_tensors[-1][0, -1]
        changed_context = changed_tensors[-1][0, -1]
        label = "with_position" if add_position else "without_position"
        comparisons[label] = {
            "query_max_abs_difference": float(
                (original_tensors[1][0, -1] - changed_tensors[1][0, -1]).abs().max().item()
            ),
            "swapped_weight_pair_max_abs_error": float(max(
                abs(original_weights[first].item() - changed_weights[second].item()),
                abs(original_weights[second].item() - changed_weights[first].item()),
            )),
            "context_max_abs_difference": float((original_context - changed_context).abs().max().item()),
            "prediction_abs_difference_ot": abs(
                original_scale(original_prediction.item(), mean, std)
                - original_scale(changed_prediction.item(), mean, std)
            ),
        }
        for arrangement, prediction in [
            ("original", original_prediction), ("swapped", changed_prediction)
        ]:
            rows.append({
                "position_mode": label,
                "arrangement": arrangement,
                "prediction_ot": original_scale(prediction.item(), mean, std),
            })

    metadata = {
        "first_hour": SWAP_HOURS[0],
        "first_time": str(pd.Timestamp(window_times[first])),
        "first_ot": float(raw_window[first]),
        "second_hour": SWAP_HOURS[1],
        "second_time": str(pd.Timestamp(window_times[second])),
        "second_ot": float(raw_window[second]),
        "comparisons": comparisons,
    }
    return rows, metadata


def validation_rows(model, raw_validation, standardized_validation, mean, std):
    """计算 LSTM 与三种注意力读法在同一验证目标上的指标。"""
    rows = []
    predictions_by_reading = {}
    for reading in ["original", "mean", "last"]:
        predictions = predict_attention(
            model, standardized_validation["X"], BATCH_SIZE, mean, std, reading
        )
        predictions_by_reading[reading] = predictions
        scores = calculate_metrics(raw_validation["y"], predictions)
        rows.append({
            "model_or_reading": {
                "original": "attention_original",
                "mean": "attention_mean_read",
                "last": "attention_last_hour_read",
            }[reading],
            "split": "validation",
            "n_samples": len(predictions),
            **scores,
        })

    lstm_metrics = pd.read_csv(LSTM_METRICS_PATH)
    selected = lstm_metrics[
        (lstm_metrics["split"] == "validation")
        & (lstm_metrics["chosen_lstm"].astype(str).str.lower() == "true")
    ]
    if len(selected) != 1:
        raise ValueError("无法唯一找到第七周选定 LSTM 的验证指标")
    row = selected.iloc[0]
    rows.insert(0, {
        "model_or_reading": row["model"],
        "split": "validation",
        "n_samples": int(row["n_samples"]),
        "mae": float(row["mae"]),
        "rmse": float(row["rmse"]),
    })
    return rows, predictions_by_reading


def main():
    """编排本周三项实践并保存所有提交产物。"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(RANDOM_SEED)
    df, values = load_series(DATA_PATH, SERIES_LENGTH)
    windows = make_windows(values, df["date"].to_numpy(), WINDOW_SIZE)
    raw_splits = split_windows(windows, BOUNDARIES)
    mean, std = training_statistics(df, BOUNDARIES[0][1], BOUNDARIES[0][2])
    standardized_splits = standardize_splits(raw_splits, mean, std)

    expected_counts = {"train": 8616, "validation": 2880, "test": 2880}
    actual_counts = {name: len(part["y"]) for name, part in raw_splits.items()}
    if actual_counts != expected_counts:
        raise ValueError(f"数据划分数量错误：{actual_counts}")

    model = SingleHeadAttentionPredictor(MODEL_DIM)
    history, best_state, best_epoch, first_update = train_model(
        model,
        standardized_splits["train"],
        standardized_splits["validation"],
        EPOCHS,
        BATCH_SIZE,
        LEARNING_RATE,
        RANDOM_SEED,
    )

    window_index = 0
    raw_window = raw_splits["validation"]["X"][window_index]
    standardized_window = standardized_splits["validation"]["X"][window_index]
    target_time = raw_splits["validation"]["target_time"][window_index]
    target_value = raw_splits["validation"]["y"][window_index]
    target_index = raw_splits["validation"]["target_index"][window_index]
    window_times = df["date"].to_numpy()[target_index - WINDOW_SIZE:target_index]

    window_checks, arrays = inspect_window(
        model, standardized_window, raw_window, target_time, target_value, mean, std
    )
    swap_rows, swap_metadata = inspect_swap(
        model, standardized_window, raw_window, window_times, mean, std
    )
    metric_rows, predictions_by_reading = validation_rows(
        model, raw_splits["validation"], standardized_splits["validation"], mean, std
    )

    prediction_rows = []
    for reading, predictions in predictions_by_reading.items():
        for index, time, truth, prediction in zip(
            raw_splits["validation"]["target_index"],
            raw_splits["validation"]["target_time"],
            raw_splits["validation"]["y"],
            predictions,
        ):
            prediction_rows.append({
                "reading": reading,
                "target_index": int(index),
                "target_time": time,
                "y_true": truth,
                "y_pred": prediction,
            })

    attention_rows = []
    for query_hour in range(WINDOW_SIZE):
        for key_hour in range(WINDOW_SIZE):
            attention_rows.append({
                "query_hour": query_hour + 1,
                "key_hour": key_hour + 1,
                "score": arrays["scores"][query_hour, key_hour],
                "weight": arrays["weights"][query_hour, key_hour],
            })

    summary = {
        "split_counts": actual_counts,
        "training_mean": float(mean),
        "training_std": float(std),
        "model_dim": MODEL_DIM,
        "parameter_count": count_parameters(model),
        "best_epoch": best_epoch,
        "first_update": first_update,
        "window_checks": window_checks,
        "swap_experiment": swap_metadata,
    }
    save_table(history, OUTPUT_DIR / "history.csv")
    save_table(metric_rows, OUTPUT_DIR / "metrics.csv")
    save_table(prediction_rows, OUTPUT_DIR / "predictions.csv")
    save_table(attention_rows, OUTPUT_DIR / "attention_window.csv")
    save_table([window_checks], OUTPUT_DIR / "single_window.csv")
    save_table(swap_rows, OUTPUT_DIR / "swap_predictions.csv")
    save_json(summary, OUTPUT_DIR / "checks.json")
    plot_loss_curves(history, best_epoch, OUTPUT_DIR / "loss_curves.png")
    plot_scores_and_weights(
        arrays["scores"], arrays["weights"], window_checks["target_time"],
        window_checks["score_24_3"], window_checks["weight_24_3"],
        window_checks["last_weight_sum"], OUTPUT_DIR / "scores_weights_heatmap.png"
    )
    draw_path_diagram("lstm", OUTPUT_DIR / "lstm_path.png")
    draw_path_diagram("attention", OUTPUT_DIR / "attention_path.png")
    torch.save({
        "model_state": best_state,
        "model_dim": MODEL_DIM,
        "best_epoch": best_epoch,
        "mean": float(mean),
        "std": float(std),
        "window_size": WINDOW_SIZE,
    }, OUTPUT_DIR / "selected_attention.pt")

    print(pd.DataFrame(metric_rows).to_string(index=False))
    print(pd.Series(window_checks).to_string())
    print(pd.DataFrame(swap_rows).to_string(index=False))


if __name__ == "__main__":
    main()
