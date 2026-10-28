"""实践二入口：组织基线、Ridge 调参、评价和结果输出。"""

from pathlib import Path

import numpy as np
import pandas as pd

from data import check_data, load_etth1, make_windows, split_windows
from evaluation import (
    calculate_metrics,
    make_metric_row,
    make_prediction_rows,
    verify_prediction_rows,
)
from models import (
    fit_model,
    make_lag_linear,
    make_ridge,
    predict_last_value,
    predict_model,
    predict_window_mean,
)
from reporting import (
    plot_series_split,
    plot_tuning,
    plot_validation_predictions,
    save_table,
)


DATA_PATH = Path("data/ETTh1.csv")
OUTPUT_DIR = Path("results")
SERIES_LENGTH = 14400
WINDOW_SIZE = 24
RIDGE_ALPHAS = [0.1, 10.0, 1000.0]
BOUNDARIES = [
    ("train", "2016-07-01 00:00", "2017-06-26 00:00"),
    ("validation", "2017-06-26 00:00", "2017-10-24 00:00"),
    ("test", "2017-10-24 00:00", "2018-02-21 00:00"),
]
MODEL_ORDER = ["last_value", "window_mean", "lag_linear", "ridge_tuned"]
VALIDATION_WEEK_START = "2017-06-26 00:00"
VALIDATION_WEEK_END = "2017-07-03 00:00"


def fixed_predictions(name, X):
    if name == "last_value":
        return predict_last_value(X)
    return predict_window_mean(X)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = load_etth1(DATA_PATH).iloc[:SERIES_LENGTH].copy()
    check_data(df)
    series = df["OT"].to_numpy(dtype=float)
    times = df["date"].to_numpy()
    X, y, target_index = make_windows(series, WINDOW_SIZE)
    splits = split_windows(X, y, target_index, times, BOUNDARIES)

    split_total = sum(len(part["y"]) for part in splits.values())
    if split_total != len(y):
        raise ValueError("时间划分遗漏了可用目标")
    all_indices = np.concatenate([part["target_index"] for part in splits.values()])
    if len(np.unique(all_indices)) != len(all_indices):
        raise ValueError("时间划分存在重复目标")

    linear = fit_model(make_lag_linear(), splits["train"]["X"], splits["train"]["y"])
    tuning_rows = []
    linear_train = predict_model(linear, splits["train"]["X"])
    linear_val = predict_model(linear, splits["validation"]["X"])
    linear_train_scores = calculate_metrics(splits["train"]["y"], linear_train)
    linear_val_scores = calculate_metrics(splits["validation"]["y"], linear_val)
    tuning_rows.append({
        "model": "lag_linear", "param_name": "", "param_value": np.nan,
        "train_mae": linear_train_scores["mae"], "val_mae": linear_val_scores["mae"],
        "train_rmse": linear_train_scores["rmse"], "val_rmse": linear_val_scores["rmse"],
        "rmse_gap": linear_val_scores["rmse"] - linear_train_scores["rmse"],
        "chosen_candidate": False,
    })

    ridge_models = {}
    for alpha in RIDGE_ALPHAS:
        model = fit_model(make_ridge(alpha), splits["train"]["X"], splits["train"]["y"])
        ridge_models[alpha] = model
        train_pred = predict_model(model, splits["train"]["X"])
        val_pred = predict_model(model, splits["validation"]["X"])
        train_scores = calculate_metrics(splits["train"]["y"], train_pred)
        val_scores = calculate_metrics(splits["validation"]["y"], val_pred)
        tuning_rows.append({
            "model": "ridge", "param_name": "alpha", "param_value": alpha,
            "train_mae": train_scores["mae"], "val_mae": val_scores["mae"],
            "train_rmse": train_scores["rmse"], "val_rmse": val_scores["rmse"],
            "rmse_gap": val_scores["rmse"] - train_scores["rmse"],
            "chosen_candidate": False,
        })

    ridge_rows = tuning_rows[1:]
    chosen_row = min(ridge_rows, key=lambda row: (row["val_rmse"], row["param_value"]))
    chosen_row["chosen_candidate"] = True
    chosen_alpha = chosen_row["param_value"]
    final_models = {"lag_linear": linear, "ridge_tuned": ridge_models[chosen_alpha]}

    validation_scores = {}
    for name in MODEL_ORDER:
        if name in final_models:
            predictions = predict_model(final_models[name], splits["validation"]["X"])
        else:
            predictions = fixed_predictions(name, splits["validation"]["X"])
        validation_scores[name] = calculate_metrics(splits["validation"]["y"], predictions)["rmse"]
    selected_model = min(MODEL_ORDER, key=lambda name: (validation_scores[name], MODEL_ORDER.index(name)))

    metric_rows = []
    prediction_rows = []
    for name in MODEL_ORDER:
        for split_name, split_data in splits.items():
            if name in final_models:
                predictions = predict_model(final_models[name], split_data["X"])
            else:
                predictions = fixed_predictions(name, split_data["X"])
            selected = name == selected_model
            metric_rows.append(make_metric_row(name, split_name, split_data["y"], predictions, selected))
            prediction_rows.extend(make_prediction_rows(name, split_name, split_data, predictions))

    verify_prediction_rows(prediction_rows, metric_rows)
    save_table(tuning_rows, OUTPUT_DIR / "tuning.csv")
    save_table(metric_rows, OUTPUT_DIR / "metrics.csv")
    save_table(prediction_rows, OUTPUT_DIR / "predictions.csv")
    plot_tuning(tuning_rows, OUTPUT_DIR / "tuning_comparison.png")
    plot_series_split(df, WINDOW_SIZE, BOUNDARIES[1][1], BOUNDARIES[2][1],
                      OUTPUT_DIR / "series_split.png")
    plot_validation_predictions(prediction_rows, df, VALIDATION_WEEK_START,
                                VALIDATION_WEEK_END,
                                OUTPUT_DIR / "validation_predictions.png")

    split_counts = {name: len(part["y"]) for name, part in splits.items()}
    first_validation = splits["validation"]["target_index"][0]
    first_test = splits["test"]["target_index"][0]
    print(f"split_counts={split_counts}")
    print(f"first_validation={times[first_validation-WINDOW_SIZE]}..{times[first_validation-1]} -> {times[first_validation]}")
    print(f"first_test={times[first_test-WINDOW_SIZE]}..{times[first_test-1]} -> {times[first_test]}")
    print(f"chosen_alpha={chosen_alpha:g}, selected_model={selected_model}")
    print(pd.DataFrame(metric_rows).to_string(index=False))


if __name__ == "__main__":
    main()
