"""实践二入口：训练两种 LSTM，比较基线并生成提交产物。"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data import (
    load_series,
    make_windows,
    split_windows,
    standardize_splits,
    training_statistics,
)
from evaluation import calculate_metrics, predict_recurrent, verify_results
from models import LSTMPredictor, count_parameters, fit_lag_linear, predict_baseline
from reporting import plot_loss_curves, plot_validation_predictions, save_table
from training import train_model


DATA_PATH = Path("../week06/data/ETTh1.csv")
OUTPUT_DIR = Path("results")
SERIES_LENGTH = 14400
WINDOW_SIZE = 24
HIDDEN_SIZES = [16, 32]
EPOCHS = 40
BATCH_SIZE = 128
LEARNING_RATE = 0.001
RANDOM_SEED = 42
BOUNDARIES = [
    ("train", "2016-07-01 00:00", "2017-06-26 00:00"),
    ("validation", "2017-06-26 00:00", "2017-10-24 00:00"),
    ("test", "2017-10-24 00:00", "2018-02-21 00:00"),
]
VALIDATION_WEEK_START = "2017-06-26 00:00"
VALIDATION_WEEK_END = "2017-07-03 00:00"


def add_result(model_name, split_name, split_data, predictions, hidden_size,
               best_epoch, chosen, metric_rows, prediction_rows):
    scores = calculate_metrics(split_data["y"], predictions)
    metric_rows.append({
        "model": model_name,
        "split": split_name,
        "n_samples": len(predictions),
        "hidden_size": hidden_size,
        "best_epoch": best_epoch,
        "mae": scores["mae"],
        "rmse": scores["rmse"],
        "chosen_lstm": chosen,
    })
    for index, time, truth, prediction in zip(
        split_data["target_index"], split_data["target_time"],
        split_data["y"], predictions
    ):
        prediction_rows.append({
            "model": model_name,
            "split": split_name,
            "target_index": int(index),
            "target_time": time,
            "y_true": truth,
            "y_pred": prediction,
        })


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    df, values = load_series(DATA_PATH, SERIES_LENGTH)
    windows = make_windows(values, df["date"].to_numpy(), WINDOW_SIZE)
    raw_splits = split_windows(windows, BOUNDARIES)
    mean, std = training_statistics(df, BOUNDARIES[0][1], BOUNDARIES[0][2])
    standardized_splits = standardize_splits(raw_splits, mean, std)

    expected_counts = {"train": 8616, "validation": 2880, "test": 2880}
    actual_counts = {name: len(part["y"]) for name, part in raw_splits.items()}
    if actual_counts != expected_counts:
        raise ValueError(f"数据划分数量错误：{actual_counts}")
    sample = standardized_splits["validation"]
    expected_first = (values[8616:8640] - mean) / std
    if not np.allclose(sample["X"][0, :, 0], expected_first.astype(np.float32)):
        raise ValueError("首条验证窗口核对失败")

    models = {}
    best_states = {}
    best_epochs = {}
    parameter_counts = {}
    shape_records = {}
    history_rows = []

    for hidden_size in HIDDEN_SIZES:
        torch.manual_seed(RANDOM_SEED)
        model = LSTMPredictor(hidden_size)
        parameter_counts[hidden_size] = count_parameters(model)
        batch = torch.from_numpy(standardized_splits["train"]["X"][:BATCH_SIZE])
        with torch.no_grad():
            prediction, output, h_n, c_n = model.inspect_forward(batch)
        shape_records[hidden_size] = {
            "input": tuple(batch.shape),
            "output": tuple(output.shape),
            "h_n": tuple(h_n.shape),
            "c_n": tuple(c_n.shape),
            "prediction": tuple(prediction.shape),
            "last_equals_h_n": bool(torch.allclose(output[:, -1, :], h_n[0])),
        }
        history, best_state, best_epoch = train_model(
            model,
            standardized_splits["train"],
            standardized_splits["validation"],
            EPOCHS,
            BATCH_SIZE,
            LEARNING_RATE,
            RANDOM_SEED,
        )
        for row in history:
            history_rows.append({
                "model": f"lstm_h{hidden_size}",
                "hidden_size": hidden_size,
                **row,
            })
        models[hidden_size] = model
        best_states[hidden_size] = best_state
        best_epochs[hidden_size] = best_epoch

    validation_rmse = {}
    lstm_predictions = {}
    for hidden_size in HIDDEN_SIZES:
        lstm_predictions[hidden_size] = {}
        for split_name in ["train", "validation"]:
            prediction = predict_recurrent(
                models[hidden_size], standardized_splits[split_name]["X"],
                BATCH_SIZE, mean, std
            )
            lstm_predictions[hidden_size][split_name] = prediction
        validation_rmse[hidden_size] = calculate_metrics(
            raw_splits["validation"]["y"],
            lstm_predictions[hidden_size]["validation"],
        )["rmse"]

    chosen_hidden_size = min(
        HIDDEN_SIZES, key=lambda size: (validation_rmse[size], size)
    )
    chosen_model_name = f"lstm_h{chosen_hidden_size}"
    lstm_predictions[chosen_hidden_size]["test"] = predict_recurrent(
        models[chosen_hidden_size], standardized_splits["test"]["X"],
        BATCH_SIZE, mean, std
    )

    linear_model = fit_lag_linear(
        raw_splits["train"]["X"], raw_splits["train"]["y"]
    )
    metric_rows = []
    prediction_rows = []
    for model_name in ["last_value", "lag_linear"]:
        for split_name, split_data in raw_splits.items():
            prediction = predict_baseline(
                model_name, split_data["X"], linear_model
            )
            add_result(
                model_name, split_name, split_data, prediction,
                np.nan, np.nan, False, metric_rows, prediction_rows
            )

    for hidden_size in HIDDEN_SIZES:
        model_name = f"lstm_h{hidden_size}"
        split_names = ["train", "validation"]
        if hidden_size == chosen_hidden_size:
            split_names.append("test")
        for split_name in split_names:
            add_result(
                model_name, split_name, raw_splits[split_name],
                lstm_predictions[hidden_size][split_name], hidden_size,
                best_epochs[hidden_size], hidden_size == chosen_hidden_size,
                metric_rows, prediction_rows
            )

    verify_results(prediction_rows, metric_rows)
    save_table(history_rows, OUTPUT_DIR / "history.csv")
    save_table(metric_rows, OUTPUT_DIR / "metrics.csv")
    save_table(prediction_rows, OUTPUT_DIR / "predictions.csv")
    plot_loss_curves(
        history_rows, best_epochs, "LSTM", OUTPUT_DIR / "loss_curves.png"
    )
    plot_validation_predictions(
        prediction_rows, VALIDATION_WEEK_START, VALIDATION_WEEK_END,
        chosen_model_name, "LSTM", OUTPUT_DIR / "validation_predictions.png"
    )
    torch.save({
        "model_state": best_states[chosen_hidden_size],
        "hidden_size": chosen_hidden_size,
        "best_epoch": best_epochs[chosen_hidden_size],
        "mean": float(mean),
        "std": float(std),
        "window_size": WINDOW_SIZE,
    }, OUTPUT_DIR / "selected_lstm.pt")

    print(f"split_counts={actual_counts}, mean={mean:.6f}, std={std:.6f}")
    print(f"parameter_counts={parameter_counts}")
    print(f"shape_records={shape_records}")
    print(f"best_epochs={best_epochs}, chosen={chosen_model_name}")
    print(pd.DataFrame(metric_rows).to_string(index=False))


if __name__ == "__main__":
    main()
