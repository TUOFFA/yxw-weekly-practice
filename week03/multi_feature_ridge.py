"""diabetes 十特征线性回归与 Ridge 正则化对照。

main() 按任务顺序组织 6 个步骤，每个步骤一个函数：1 固定划分、2 按列标准化、
3 训练五个模型、4 比较训练与验证并选定 alpha、5 观察权重、6 输出固定摘要。
产物写入脚本同级的 results/。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.datasets import load_diabetes
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
RIDGE_ALPHAS = (1, 100)
FEATURES = ("age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6")
TARGET = "target"
BMI_INDEX = FEATURES.index("bmi")
OUTPUT_DIR = Path(__file__).parent / "results"


def metric_row(model: str, split: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | str]:
    """一个 (model, split) 的 MAE、RMSE、MSE。"""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "model": model,
        "split": split,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "MSE": mse,
    }


def split_rows() -> dict[str, tuple[pd.DataFrame, np.ndarray]]:
    """1. 对样本行一次性划分，单特征与多特征取同一批行里的不同列。"""
    frame = load_diabetes(as_frame=True, scaled=False).frame
    x_all = frame[list(FEATURES)]
    y_all = frame[TARGET].to_numpy(dtype=float)

    x_train, x_holdout, y_train, y_holdout = train_test_split(
        x_all, y_all, test_size=0.4, random_state=RANDOM_STATE
    )
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_holdout, y_holdout, test_size=0.5, random_state=RANDOM_STATE
    )
    return {
        "train": (x_train, y_train),
        "validation": (x_validation, y_validation),
        "test": (x_test, y_test),
    }


def scale_splits(
    raw_splits: dict[str, tuple[pd.DataFrame, np.ndarray]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """2. 只在训练特征上 fit StandardScaler，再用同一组统计量变换验证与测试。"""
    scaler = StandardScaler().fit(raw_splits["train"][0])
    return {name: (scaler.transform(x), y) for name, (x, y) in raw_splits.items()}


def train_models(
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, tuple[list[int], RegressorMixin]]:
    """3. 五个模型共用同一组训练样本，每组设置都从训练数据独立拟合。"""
    x_train, y_train = splits["train"]
    x_bmi = x_train[:, [BMI_INDEX]]

    baseline = DummyRegressor(strategy="mean").fit(x_train, y_train)
    single = LinearRegression().fit(x_bmi, y_train)
    multi = LinearRegression().fit(x_train, y_train)
    ridge_1 = Ridge(alpha=RIDGE_ALPHAS[0]).fit(x_train, y_train)
    ridge_100 = Ridge(alpha=RIDGE_ALPHAS[1]).fit(x_train, y_train)

    # 模型名 -> (取哪些输入列, 拟合好的估计器)；单特征模型只取 bmi 一列。
    all_columns = list(range(len(FEATURES)))
    return {
        "Mean baseline": (all_columns, baseline),
        "LinearRegression (bmi)": ([BMI_INDEX], single),
        "LinearRegression (10 features)": (all_columns, multi),
        "Ridge alpha=1": (all_columns, ridge_1),
        "Ridge alpha=100": (all_columns, ridge_100),
    }


def evaluate_and_select(
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    models: dict[str, tuple[list[int], RegressorMixin]],
) -> tuple[pd.DataFrame, str]:
    """4. 训练/验证指标与 RMSE 图，按验证 RMSE 选定 alpha，再评价测试集一次。"""
    rows = []
    for split_name in ("train", "validation"):
        x, y = splits[split_name]
        for name, (columns, model) in models.items():
            rows.append(metric_row(name, split_name, y, model.predict(x[:, columns])))
    train_validation = pd.DataFrame(rows, columns=("model", "split", "MAE", "RMSE", "MSE"))

    names = list(models)
    rmse = train_validation.pivot(index="model", columns="split", values="RMSE").loc[names]
    positions = np.arange(len(names))
    bar_width = 0.36
    plt.figure(figsize=(11, 5.5))
    plt.bar(positions - bar_width / 2, rmse["train"], bar_width, label="Train RMSE", color="#4C78A8")
    plt.bar(
        positions + bar_width / 2,
        rmse["validation"],
        bar_width,
        label="Validation RMSE",
        color="#F58518",
    )
    plt.xticks(positions, names, rotation=20, ha="right")
    plt.ylabel("RMSE (target units)")
    plt.title(f"Train vs validation RMSE, 60/20/20 split (random_state={RANDOM_STATE})")
    plt.legend()
    # 模型名较长且斜排，留出更大的下边距，避免旋转后的刻度标签被画布裁掉。
    plt.subplots_adjust(bottom=0.3, left=0.09, right=0.98, top=0.9)
    plt.savefig(OUTPUT_DIR / "train_validation_rmse_comparison.png", dpi=150)
    plt.close()

    candidates = train_validation[
        (train_validation["split"] == "validation")
        & train_validation["model"].str.startswith("Ridge")
    ]
    # 并列时 idxmin 取先出现的一行；Ridge 在 models 里按 alpha 升序，等价于取较小 alpha。
    selected = candidates.loc[candidates["RMSE"].idxmin(), "model"]

    x_test, y_test = splits["test"]
    test_rows = [
        metric_row(name, "test", y_test, model.predict(x_test[:, columns]))
        for name, (columns, model) in models.items()
        if not name.startswith("Ridge") or name == selected
    ]
    metrics = pd.concat([train_validation, pd.DataFrame(test_rows)], ignore_index=True)
    return metrics, selected


def compare_weights(models: dict[str, tuple[list[int], RegressorMixin]]) -> pd.DataFrame:
    """5. 三个多特征模型的权重宽表（截距单独一行）与同图分组柱状图。"""
    names = ("LinearRegression (10 features)", "Ridge alpha=1", "Ridge alpha=100")

    weights = pd.DataFrame({"feature": list(FEATURES)})
    for name in names:
        weights[name] = models[name][1].coef_
    intercept = pd.DataFrame(
        {"feature": ["intercept"], **{name: [float(models[name][1].intercept_)] for name in names}}
    )
    weights = pd.concat([weights, intercept], ignore_index=True)

    # 权重图不含截距，因为 Ridge 的正则化约束的是特征权重。
    feature_rows = weights.iloc[:-1]
    positions = np.arange(len(FEATURES))
    bar_width = 0.25
    offset_unit = (len(names) - 1) / 2
    colors = ("#4C78A8", "#F58518", "#54A24B")
    plt.figure(figsize=(12, 5.5))
    for offset, (name, color) in enumerate(zip(names, colors)):
        plt.bar(
            positions + (offset - offset_unit) * bar_width,
            feature_rows[name],
            bar_width,
            label=name,
            color=color,
        )
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xticks(positions, FEATURES)
    plt.xlabel("Feature (standardized using training statistics)")
    plt.ylabel("Coefficient")
    plt.title("Multi-feature model coefficient comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "multifeature_weight_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    return weights


def build_summary(
    splits: dict[str, tuple[np.ndarray, np.ndarray]], selected: str
) -> str:
    """6. 固定摘要：数据、划分、输入、预处理、基线、候选项、选中配置、测试纪律、权重。"""
    sizes = {name: len(values) for name, (_, values) in splits.items()}
    train_mean = float(splits["train"][1].mean())
    return "\n".join(
        [
            "Week 03: multi-feature regression and Ridge comparison",
            "=" * 55,
            "Data: sklearn load_diabetes(as_frame=True, scaled=False); 10 features + target.",
            f"Fixed random seed: {RANDOM_STATE}; samples: train={sizes['train']}, "
            f"validation={sizes['validation']}, test={sizes['test']}.",
            "Inputs: bmi only for the single-feature model; all 10 features for the multi-feature models.",
            "Preprocessing: StandardScaler fitted on the training features only; target left unscaled.",
            f"Baseline: training-target mean = {train_mean:.6f}.",
            f"Ridge candidates: alpha={RIDGE_ALPHAS[0]}, alpha={RIDGE_ALPHAS[1]}; "
            f"SELECTED by validation RMSE only: {selected}.",
            "The test set was evaluated once after alpha selection and was not used for tuning.",
            "Weights in multifeature_weights.csv use standardized inputs; they are not causal effects.",
        ]
    )


def main() -> None:
    """按任务 6 个步骤编排主流程，计算细节都在上面的函数里。"""
    OUTPUT_DIR.mkdir(exist_ok=True)

    raw_splits = split_rows()
    split_info = pd.DataFrame(
        {"split": list(raw_splits), "samples": [len(values) for _, values in raw_splits.values()]}
    )
    split_info.to_csv(OUTPUT_DIR / "split_info.csv", index=False, encoding="utf-8-sig")

    splits = scale_splits(raw_splits)
    models = train_models(splits)
    metrics, selected = evaluate_and_select(splits, models)
    metrics.to_csv(
        OUTPUT_DIR / "metrics_comparison.csv", index=False, encoding="utf-8-sig", float_format="%.4f"
    )

    weights = compare_weights(models)
    weights.to_csv(
        OUTPUT_DIR / "multifeature_weights.csv", index=False, encoding="utf-8-sig", float_format="%.4f"
    )

    summary = build_summary(splits, selected)
    print(summary)
    (OUTPUT_DIR / "selection_summary.txt").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
