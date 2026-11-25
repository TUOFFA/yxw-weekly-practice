"""完成双头、因果遮罩和单层 Transformer 编码器三项实践。"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from data import (
    load_series,
    make_windows,
    split_windows,
    standardize_splits,
    training_statistics,
)
from evaluation import calculate_metrics, predict
from models import AttentionHead, TransformerEncoderPredictor, TwoHeadAttention, count_parameters
from reporting import plot_head_comparison, plot_loss_curves, save_json, save_table
from training import train_model


DATA_PATH = Path("../week06/data/ETTh1.csv")
LSTM_METRICS_PATH = Path("../week07/results/metrics.csv")
SINGLE_HEAD_METRICS_PATH = Path("../week08/results/metrics.csv")
SINGLE_HEAD_CHECKS_PATH = Path("../week08/results/checks.json")
OUTPUT_DIR = Path("results")
SERIES_LENGTH = 14_400
WINDOW_SIZE = 24
MODEL_DIM = 16
HEAD_DIM = 8
FEEDFORWARD_DIM = 32
EPOCHS = 40
BATCH_SIZE = 128
LEARNING_RATE = 0.001
RANDOM_SEED = 42
BOUNDARIES = [
    ("train", "2016-07-01 00:00", "2017-06-26 00:00"),
    ("validation", "2017-06-26 00:00", "2017-10-24 00:00"),
    ("test", "2017-10-24 00:00", "2018-02-21 00:00"),
]


def load_data() -> tuple[pd.DataFrame, dict, dict, float, float]:
    """读取数据、按目标时间划分并用训练统计量标准化。"""
    frame, values = load_series(DATA_PATH, SERIES_LENGTH)
    windows = make_windows(values, frame["date"].to_numpy(), WINDOW_SIZE)
    raw_splits = split_windows(windows, BOUNDARIES)
    mean, std = training_statistics(frame, BOUNDARIES[0][1], BOUNDARIES[0][2])
    standardized_splits = standardize_splits(raw_splits, mean, std)
    expected_counts = {"train": 8616, "validation": 2880, "test": 2880}
    actual_counts = {name: len(part["y"]) for name, part in raw_splits.items()}
    if actual_counts != expected_counts:
        raise ValueError(f"数据划分数量错误：{actual_counts}")
    return frame, raw_splits, standardized_splits, mean, std


def run_practice4(
    standardized_window: np.ndarray, target_time
) -> tuple[nn.Linear, TwoHeadAttention, list[dict], list[dict], dict]:
    """比较复制参数与独立参数的两个注意力头。"""
    inputs = torch.from_numpy(standardized_window[None, ...])
    input_projection = nn.Linear(1, MODEL_DIM)
    head1 = AttentionHead(MODEL_DIM, HEAD_DIM)
    copied_head2 = AttentionHead(MODEL_DIM, HEAD_DIM)
    copied_head2.load_state_dict(head1.state_dict())
    independent_head2 = AttentionHead(MODEL_DIM, HEAD_DIM)
    attention = TwoHeadAttention(MODEL_DIM, HEAD_DIM)
    attention.head1.load_state_dict(head1.state_dict())
    attention.head2.load_state_dict(independent_head2.state_dict())
    input_projection.eval()
    attention.eval()

    with torch.no_grad():
        embedded = input_projection(inputs)
        same_weights1, same_context1 = head1(embedded)
        same_weights2, same_context2 = copied_head2(embedded)
        independent_weights1, independent_context1 = attention.head1(embedded)
        independent_weights2, independent_context2 = attention.head2(embedded)
        independent_concat = torch.cat(
            [independent_context1, independent_context2], dim=-1
        )
        mixed = attention.output_projection(independent_concat)

    cases = [
        (
            "copied_parameters",
            same_weights1,
            same_weights2,
            same_context1,
            same_context2,
        ),
        (
            "independent_parameters",
            independent_weights1,
            independent_weights2,
            independent_context1,
            independent_context2,
        ),
    ]
    comparison_rows = []
    weight_rows = []
    for scenario, weights1, weights2, context1, context2 in cases:
        weight_difference = float(
            (weights1[0, 23, :] - weights2[0, 23, :]).abs().max().item()
        )
        context_difference = float(
            (context1[0, 23, :] - context2[0, 23, :]).abs().max().item()
        )
        comparison_rows.append(
            {
                "scenario": scenario,
                "target_time": str(pd.Timestamp(target_time)),
                "last_weight_max_abs_difference": weight_difference,
                "last_context_max_abs_difference": context_difference,
                "head_context_shape": str(tuple(context1.shape)),
                "concatenated_shape": str(tuple(independent_concat.shape)),
                "mixed_shape": str(tuple(mixed.shape)),
            }
        )
        for head_name, weights in [("head1", weights1), ("head2", weights2)]:
            for hour, weight in enumerate(weights[0, 23, :].tolist(), start=1):
                weight_rows.append(
                    {
                        "scenario": scenario,
                        "head": head_name,
                        "query_hour": 24,
                        "key_hour": hour,
                        "weight": weight,
                    }
                )

    same_row, independent_row = comparison_rows
    plot_head_comparison(
        np.arange(1, 25),
        same_weights1[0, 23, :].numpy(),
        same_weights2[0, 23, :].numpy(),
        independent_weights1[0, 23, :].numpy(),
        independent_weights2[0, 23, :].numpy(),
        str(pd.Timestamp(target_time)),
        same_row["last_weight_max_abs_difference"],
        independent_row["last_weight_max_abs_difference"],
        OUTPUT_DIR / "practice4_head_comparison.png",
    )
    checks = {
        "same_weight_difference": same_row["last_weight_max_abs_difference"],
        "same_context_difference": same_row["last_context_max_abs_difference"],
        "independent_weight_difference": independent_row[
            "last_weight_max_abs_difference"
        ],
        "independent_context_difference": independent_row[
            "last_context_max_abs_difference"
        ],
        "head_context_shape": list(same_context1.shape),
        "concatenated_shape": list(independent_concat.shape),
        "mixed_shape": list(mixed.shape),
    }
    return input_projection, attention, comparison_rows, weight_rows, checks


def run_practice5(
    standardized_window: np.ndarray,
    input_projection: nn.Linear,
    attention: TwoHeadAttention,
) -> tuple[list[dict], dict]:
    """只改第 18 小时，比较第 6 行在遮罩前后的变化。"""
    changed_window = standardized_window.copy()
    changed_window[17, 0] += 1.0
    original = torch.from_numpy(standardized_window[None, ...])
    changed = torch.from_numpy(changed_window[None, ...])
    rows = []
    results = {}

    for use_causal_mask in [False, True]:
        label = "on" if use_causal_mask else "off"
        with torch.no_grad():
            original_result = attention.inspect_forward(
                input_projection(original), use_causal_mask
            )
            changed_result = attention.inspect_forward(
                input_projection(changed), use_causal_mask
            )
        row_weights = original_result["weights1"][0, 5, :]
        output_difference = float(
            (
                original_result["output"][0, 5, :]
                - changed_result["output"][0, 5, :]
            )
            .abs()
            .max()
            .item()
        )
        row = {
            "causal_mask": label,
            "changed_hour": 18,
            "observed_query_hour": 6,
            "head1_weight_to_hour18": float(row_weights[17].item()),
            "head1_future_max_weight": float(row_weights[6:].max().item()),
            "head1_row_weight_sum": float(row_weights.sum().item()),
            "row6_output_max_abs_difference": output_difference,
        }
        rows.append(row)
        results[label] = row

    checks = {
        "changed_standardized_value": 1.0,
        "unmasked": results["off"],
        "masked": results["on"],
    }
    return rows, checks


def shape_trace(
    model: TransformerEncoderPredictor,
    standardized_window: np.ndarray,
    mean: float,
    std: float,
) -> tuple[list[dict], dict]:
    """记录一条窗口经过编码器时各张量形状。"""
    inputs = torch.from_numpy(standardized_window[None, ...])
    model.eval()
    with torch.no_grad():
        tensors = model.inspect_forward(inputs)
    descriptions = {
        "X": "输入映射加位置，进入注意力前",
        "M": "双头注意力合并并投影后",
        "H": "第一处残差加法与 LayerNorm 后",
        "F": "逐行前馈网络输出",
        "E": "第二处残差加法与 LayerNorm 后",
        "last_row": "编码器第 24 行，交给预测层",
        "prediction": "标准化第 25 小时预测",
    }
    rows = [
        {"tensor": name, "shape": str(tuple(tensor.shape)), "meaning": descriptions[name]}
        for name, tensor in tensors.items()
    ]
    prediction_standardized = float(tensors["prediction"].item())
    checks = {
        "shapes": {name: list(tensor.shape) for name, tensor in tensors.items()},
        "prediction_standardized": prediction_standardized,
        "prediction_ot": prediction_standardized * std + mean,
    }
    return rows, checks


def previous_model_rows() -> list[dict]:
    """读取 week07 与 week08 的同批验证指标。"""
    lstm_metrics = pd.read_csv(LSTM_METRICS_PATH)
    selected_lstm = lstm_metrics[
        (lstm_metrics["split"] == "validation")
        & (lstm_metrics["chosen_lstm"].astype(str).str.lower() == "true")
    ]
    if len(selected_lstm) != 1:
        raise ValueError("无法唯一找到第七周选定 LSTM 验证指标")
    lstm_row = selected_lstm.iloc[0]

    attention_metrics = pd.read_csv(SINGLE_HEAD_METRICS_PATH)
    selected_attention = attention_metrics[
        attention_metrics["model_or_reading"] == "attention_original"
    ]
    if len(selected_attention) != 1:
        raise ValueError("无法唯一找到第八周单头注意力验证指标")
    attention_row = selected_attention.iloc[0]
    attention_checks = json.loads(SINGLE_HEAD_CHECKS_PATH.read_text(encoding="utf-8"))

    return [
        {
            "model": "lstm_h32",
            "split": "validation",
            "n_samples": int(lstm_row["n_samples"]),
            "mae": float(lstm_row["mae"]),
            "rmse": float(lstm_row["rmse"]),
            "result_source_or_best_epoch": f"week07 selected model, epoch {int(lstm_row['best_epoch'])}",
        },
        {
            "model": "single_head_attention",
            "split": "validation",
            "n_samples": int(attention_row["n_samples"]),
            "mae": float(attention_row["mae"]),
            "rmse": float(attention_row["rmse"]),
            "result_source_or_best_epoch": (
                f"week08 selected model, epoch {attention_checks['best_epoch']}"
            ),
        },
    ]


def validation_rows(
    model: TransformerEncoderPredictor,
    raw_validation: dict,
    standardized_validation: dict,
    mean: float,
    std: float,
    best_epoch: int,
) -> tuple[list[dict], np.ndarray]:
    """汇总最近值、旧模型和本周编码器的验证指标。"""
    rows = []
    last_value_predictions = raw_validation["X"][:, -1]
    rows.append(
        {
            "model": "last_value",
            "split": "validation",
            "n_samples": len(last_value_predictions),
            **calculate_metrics(raw_validation["y"], last_value_predictions),
            "result_source_or_best_epoch": "same validation targets, no training",
        }
    )
    rows.extend(previous_model_rows())
    transformer_predictions = predict(
        model, standardized_validation["X"], BATCH_SIZE, mean, std
    )
    rows.append(
        {
            "model": "one_layer_transformer_encoder",
            "split": "validation",
            "n_samples": len(transformer_predictions),
            **calculate_metrics(raw_validation["y"], transformer_predictions),
            "result_source_or_best_epoch": f"week10 best epoch {best_epoch}",
        }
    )
    return rows, transformer_predictions


def validate_practice4(checks: dict) -> None:
    """检查复制参数的两个头是否得到相同结果。"""
    if checks["same_weight_difference"] != 0.0:
        raise AssertionError("复制参数的两个头应产生完全相同的权重")
    if checks["same_context_difference"] != 0.0:
        raise AssertionError("复制参数的两个头应产生完全相同的读出")


def validate_practice5(checks: dict) -> None:
    """检查第 18 小时的改动是否被因果遮罩阻断。"""
    masked = checks["masked"]
    if masked["head1_weight_to_hour18"] != 0.0:
        raise AssertionError("因果遮罩后第 6 行不能读取第 18 小时")
    if masked["head1_future_max_weight"] != 0.0:
        raise AssertionError("因果遮罩后的未来权重必须为 0")
    if masked["row6_output_max_abs_difference"] != 0.0:
        raise AssertionError("因果遮罩后第 18 小时不能影响第 6 行")


def validate_encoder_shapes(shapes: dict[str, list[int]]) -> None:
    """检查编码器保留所有时间位置，直到末端预测层。"""
    expected_shapes = {
        "X": [1, 24, 16],
        "M": [1, 24, 16],
        "H": [1, 24, 16],
        "F": [1, 24, 16],
        "E": [1, 24, 16],
        "last_row": [1, 16],
        "prediction": [1, 1],
    }
    if shapes != expected_shapes:
        raise AssertionError(f"编码器形状错误：{shapes}")


def validate_first_update(first_update: dict) -> None:
    """检查首批反向传播和优化器更新之间的参数状态。"""
    if first_update["query_changed_before_step"] or first_update[
        "predictor_changed_before_step"
    ]:
        raise AssertionError("optimizer.step() 前参数不应变化")
    if not first_update["query_changed_after_step"] or not first_update[
        "predictor_changed_after_step"
    ]:
        raise AssertionError("optimizer.step() 后两处分支参数都应变化")


def main() -> None:
    """按实践四、五、六的顺序运行并保存结果。"""
    # 三项实践共用第一条验证窗口。
    OUTPUT_DIR.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(RANDOM_SEED)
    _, raw_splits, standardized_splits, mean, std = load_data()
    raw_validation = raw_splits["validation"]
    standardized_validation = standardized_splits["validation"]
    window_index = 0
    standardized_window = standardized_validation["X"][window_index]
    target_time = raw_validation["target_time"][window_index]

    # 实践四：对照同参和独立参数的双头读法，记录末行权重。
    print("实践四｜双头注意力对照")
    (
        input_projection,
        practice_attention,
        practice4_rows,
        weight_rows,
        practice4_checks,
    ) = run_practice4(standardized_window, target_time)
    validate_practice4(practice4_checks)
    save_table(practice4_rows, OUTPUT_DIR / "practice4_comparison.csv")
    save_table(weight_rows, OUTPUT_DIR / "practice4_weights.csv")
    print(pd.DataFrame(practice4_rows).to_string(index=False))

    # 实践五：复用实践四的窗口、输入映射和双头参数，验证因果遮罩。
    print("\n实践五｜因果遮罩对照")
    practice5_rows, practice5_checks = run_practice5(
        standardized_window, input_projection, practice_attention
    )
    validate_practice5(practice5_checks)
    save_table(practice5_rows, OUTPUT_DIR / "practice5_mask_comparison.csv")
    print(pd.DataFrame(practice5_rows).to_string(index=False))

    # 实践六：新建编码器，先用单窗口核对接线，再训练整层。
    print("\n实践六｜组装编码器并检查训练前形状")
    torch.manual_seed(RANDOM_SEED)
    model = TransformerEncoderPredictor(MODEL_DIM, HEAD_DIM, FEEDFORWARD_DIM)
    shape_rows, initial_checks = shape_trace(model, standardized_window, mean, std)
    validate_encoder_shapes(initial_checks["shapes"])
    print(pd.DataFrame(shape_rows).to_string(index=False))
    print(f"训练前单窗口标准化预测值为 {initial_checks['prediction_standardized']:.6f}")

    print(f"\n开始训练，共 {EPOCHS} 轮")
    history, best_state, best_epoch, first_update = train_model(
        model,
        standardized_splits["train"],
        standardized_validation,
        EPOCHS,
        BATCH_SIZE,
        LEARNING_RATE,
        RANDOM_SEED,
    )
    validate_first_update(first_update)

    # train_model 已恢复最佳参数，用该模型记录预测并比较验证误差。
    _, shape_checks = shape_trace(model, standardized_window, mean, std)
    metric_rows, predictions = validation_rows(
        model,
        raw_validation,
        standardized_validation,
        mean,
        std,
        best_epoch,
    )

    prediction_rows = [
        {
            "target_index": int(index),
            "target_time": time,
            "y_true": truth,
            "y_pred": prediction,
        }
        for index, time, truth, prediction in zip(
            raw_validation["target_index"],
            raw_validation["target_time"],
            raw_validation["y"],
            predictions,
        )
    ]
    save_table(shape_rows, OUTPUT_DIR / "shape_trace.csv")
    save_table(history, OUTPUT_DIR / "history.csv")
    save_table(metric_rows, OUTPUT_DIR / "metrics.csv")
    save_table(prediction_rows, OUTPUT_DIR / "predictions.csv")
    plot_loss_curves(history, best_epoch, OUTPUT_DIR / "loss_curves.png")
    torch.save(
        {
            "model_state": best_state,
            "model_dim": MODEL_DIM,
            "head_dim": HEAD_DIM,
            "feedforward_dim": FEEDFORWARD_DIM,
            "best_epoch": best_epoch,
            "mean": mean,
            "std": std,
            "window_size": WINDOW_SIZE,
        },
        OUTPUT_DIR / "selected_transformer_encoder.pt",
    )

    print(f"\n第 {best_epoch} 轮最佳模型的验证指标")
    print(pd.DataFrame(metric_rows).to_string(index=False))

    # 汇总三项实践的检查结果。
    summary = {
        "data_path": str(DATA_PATH),
        "split_counts": {name: len(part["y"]) for name, part in raw_splits.items()},
        "training_mean": mean,
        "training_std": std,
        "target_time": str(pd.Timestamp(target_time)),
        "model_dim": MODEL_DIM,
        "head_dim": HEAD_DIM,
        "feedforward_dim": FEEDFORWARD_DIM,
        "parameter_count": count_parameters(model),
        "best_epoch": best_epoch,
        "first_update": first_update,
        "practice4": practice4_checks,
        "practice5": practice5_checks,
        "practice6_shape_trace": shape_checks,
    }
    save_json(summary, OUTPUT_DIR / "checks.json")


if __name__ == "__main__":
    main()
