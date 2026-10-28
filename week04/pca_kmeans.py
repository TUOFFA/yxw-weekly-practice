"""diabetes 十特征的 PCA 降维回归对照与 K-means 聚类。

main() 只做编排，两个实践共用同一份 60/20/20 划分与同一套标准化结果：
1 划分样本、2 只在训练集标准化、3 拟合 2/5/10 维 PCA、4 训练四组回归模型、
5 比较训练与验证并选定维数、6 累计解释方差、7 重构误差、8 拟合两组 K-means、
9 轮廓系数、10 组内统计、11 聚类双面板。
产物写入脚本同级的 results/。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.cluster import KMeans
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
HOLDOUT_SIZE = 0.4
HOLDOUT_TEST_SHARE = 0.5
FEATURES = ("age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6")
TARGET = "target"
DESCRIBED_FEATURES = ("bmi", "bp")

SCALED = "scaled"
BASELINE = "Mean baseline"
FULL_MODEL = "LinearRegression (10 features)"
PCA_COMPONENTS = (2, 5)
FULL_COMPONENTS = len(FEATURES)
PCA_MODEL_NAMES = {n: f"PCA {n}D + LinearRegression" for n in PCA_COMPONENTS}

# 模型名 -> 输入表示：SCALED 表示 10 维标准化特征，整数表示该维数的主成分坐标。
# 这里的顺序就是指标表与柱状图的顺序；并列时取先出现的行，因此 2 维排在 5 维之前。
MODEL_INPUTS: dict[str, str | int] = {
    BASELINE: SCALED,
    FULL_MODEL: SCALED,
    **{PCA_MODEL_NAMES[n]: n for n in PCA_COMPONENTS},
}
METRIC_COLUMNS = (
    "model",
    "pca_components",
    "train_MAE",
    "train_RMSE",
    "validation_MAE",
    "validation_RMSE",
    "test_MAE",
    "test_RMSE",
    "selected",
)

SVD_SOLVER = "full"
WHITEN = False
CLUSTER_COUNTS = (2, 3)
KMEANS_N_INIT = 10

TRAIN_COLOR = "#4C78A8"
VALIDATION_COLOR = "#F58518"
CLUSTER_COLORS = ("#4C78A8", "#F58518", "#54A24B")
BAR_WIDTH = 0.36
POINT_SIZE = 14
CENTER_SIZE = 170
AXIS_MARGIN = 0.08
FIGURE_DPI = 150
OUTPUT_DIR = Path(__file__).parent / "results"


def split_rows() -> dict[str, tuple[pd.DataFrame, np.ndarray]]:
    """1. 对样本行划分一次约 60/20/20，回归与聚类共用这批训练行。"""
    frame = load_diabetes(as_frame=True, scaled=False).frame
    x_all = frame[list(FEATURES)]
    y_all = frame[TARGET].to_numpy(dtype=float)

    x_train, x_holdout, y_train, y_holdout = train_test_split(
        x_all, y_all, test_size=HOLDOUT_SIZE, random_state=RANDOM_STATE
    )
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_holdout, y_holdout, test_size=HOLDOUT_TEST_SHARE, random_state=RANDOM_STATE
    )
    return {
        "train": (x_train, y_train),
        "validation": (x_validation, y_validation),
        "test": (x_test, y_test),
    }


def scale_splits(
    raw_splits: dict[str, tuple[pd.DataFrame, np.ndarray]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """2. 只在训练特征上 fit StandardScaler，同一组统计量变换三部分；target 保持原样。"""
    scaler = StandardScaler().fit(raw_splits["train"][0])
    return {name: (scaler.transform(x), y) for name, (x, y) in raw_splits.items()}


def fit_pcas(splits: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[int, PCA]:
    """3. 在标准化训练特征上分别拟合 2 维、5 维与保留全部 10 维的 PCA。"""
    x_train = splits["train"][0]
    return {
        n: PCA(n_components=n, svd_solver=SVD_SOLVER, whiten=WHITEN).fit(x_train)
        for n in (*PCA_COMPONENTS, FULL_COMPONENTS)
    }


def project(
    pca: PCA, splits: dict[str, tuple[np.ndarray, np.ndarray]]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """3. 三部分特征都用训练集学到的同一组投影方向变换。"""
    return {name: (pca.transform(x), y) for name, (x, y) in splits.items()}


def input_matrix(
    representation: str | int,
    split: str,
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    pca_splits: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]],
) -> np.ndarray:
    """按模型的输入表示取该 split 的矩阵：SCALED 取 10 维特征，整数取主成分坐标。"""
    if representation == SCALED:
        return splits[split][0]
    return pca_splits[representation][split][0]


def split_metrics(
    model: RegressorMixin,
    representation: str | int,
    split: str,
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    pca_splits: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]],
) -> tuple[float, float]:
    """一个模型在一个 split 上的 MAE 与 RMSE。"""
    y_true = splits[split][1]
    y_pred = model.predict(input_matrix(representation, split, splits, pca_splits))
    return mean_absolute_error(y_true, y_pred), root_mean_squared_error(y_true, y_pred)


def train_models(
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    pca_splits: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]],
) -> dict[str, RegressorMixin]:
    """4. 四组模型共用同一组训练样本与同一个 LinearRegression，只换输入表示。"""
    y_train = splits["train"][1]
    models: dict[str, RegressorMixin] = {}
    for name, representation in MODEL_INPUTS.items():
        estimator = DummyRegressor(strategy="mean") if name == BASELINE else LinearRegression()
        models[name] = estimator.fit(
            input_matrix(representation, "train", splits, pca_splits), y_train
        )
    return models


def evaluate_and_select(
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    pca_splits: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]],
    models: dict[str, RegressorMixin],
) -> pd.DataFrame:
    """5. 训练/验证指标、RMSE 分组柱状图，按验证 RMSE 选维数，再评价选定的配置一次。"""
    rows = []
    for name, representation in MODEL_INPUTS.items():
        row: dict[str, float | str | int | None] = {
            "model": name,
            "pca_components": None if representation == SCALED else representation,
        }
        for split_name in ("train", "validation"):
            mae, rmse = split_metrics(
                models[name], representation, split_name, splits, pca_splits
            )
            row[f"{split_name}_MAE"] = mae
            row[f"{split_name}_RMSE"] = rmse
        rows.append(row)
    comparison = pd.DataFrame(rows)
    # 不适用主成分维数的两行保持空单元格，所以用可空的整数列。
    comparison["pca_components"] = comparison["pca_components"].astype("Int64")

    positions = np.arange(len(MODEL_INPUTS))
    plt.figure(figsize=(11, 5.5))
    plt.bar(
        positions - BAR_WIDTH / 2,
        comparison["train_RMSE"],
        BAR_WIDTH,
        label="Train RMSE",
        color=TRAIN_COLOR,
    )
    plt.bar(
        positions + BAR_WIDTH / 2,
        comparison["validation_RMSE"],
        BAR_WIDTH,
        label="Validation RMSE",
        color=VALIDATION_COLOR,
    )
    plt.xticks(positions, list(MODEL_INPUTS), rotation=20, ha="right")
    plt.ylabel("RMSE (target units)")
    plt.title(f"Train vs validation RMSE, 60/20/20 split (random_state={RANDOM_STATE})")
    plt.legend()
    # 模型名较长且斜排，留出更大的下边距，避免旋转后的刻度标签被画布裁掉。
    plt.subplots_adjust(bottom=0.3, left=0.09, right=0.98, top=0.9)
    plt.savefig(OUTPUT_DIR / "rmse_comparison.png", dpi=FIGURE_DPI)
    plt.close()

    candidates = comparison[comparison["model"].isin(PCA_MODEL_NAMES.values())]
    # 并列时 idxmin 取先出现的一行；PCA_MODEL_NAMES 按 2 维、5 维排序，等价于并列选 2 维。
    selected = str(candidates.loc[candidates["validation_RMSE"].idxmin(), "model"])

    comparison["test_MAE"] = np.nan
    comparison["test_RMSE"] = np.nan
    for name in (BASELINE, FULL_MODEL, selected):
        row_index = comparison.index[comparison["model"] == name][0]
        mae, rmse = split_metrics(models[name], MODEL_INPUTS[name], "test", splits, pca_splits)
        comparison.at[row_index, "test_MAE"] = mae
        comparison.at[row_index, "test_RMSE"] = rmse
    comparison["selected"] = comparison["model"] == selected
    return comparison[list(METRIC_COLUMNS)]


def reconstruction_mse(pca: PCA, x_scaled: np.ndarray) -> float:
    """7. 重构 MSE = mean((X_scaled - inverse_transform(transform(X_scaled)))²)，对全部样本与 10 列取平均。"""
    reconstructed = pca.inverse_transform(pca.transform(x_scaled))
    return float(np.mean((x_scaled - reconstructed) ** 2))


def pca_information(
    splits: dict[str, tuple[np.ndarray, np.ndarray]], pcas: dict[int, PCA]
) -> pd.DataFrame:
    """6-7. 两个降维 PCA 的累计解释方差与训练/验证重构 MSE，并画 1–10 维的累计曲线。"""
    summary = pd.DataFrame(
        [
            {
                "components": n,
                "cumulative_explained_variance": float(
                    pcas[n].explained_variance_ratio_.sum()
                ),
                "train_reconstruction_mse": reconstruction_mse(pcas[n], splits["train"][0]),
                "validation_reconstruction_mse": reconstruction_mse(
                    pcas[n], splits["validation"][0]
                ),
            }
            for n in PCA_COMPONENTS
        ]
    )

    cumulative = np.cumsum(pcas[FULL_COMPONENTS].explained_variance_ratio_)
    dimensions = np.arange(1, FULL_COMPONENTS + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(
        dimensions, cumulative, marker="o", color=TRAIN_COLOR, label="Cumulative explained variance"
    )
    for n in PCA_COMPONENTS:
        plt.axvline(n, color="gray", linestyle="--", linewidth=0.8)
        plt.annotate(
            f"{n}D: {cumulative[n - 1]:.3f}",
            (n, cumulative[n - 1]),
            xytext=(8, -14),
            textcoords="offset points",
        )
    plt.xticks(dimensions)
    plt.ylim(0.0, 1.02)
    plt.xlabel("Number of principal components kept")
    plt.ylabel("Cumulative explained variance ratio (training set)")
    plt.title("PCA cumulative explained variance, 1-10 components")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pca_variance.png", dpi=FIGURE_DPI)
    plt.close()
    return summary


def fit_clusters(splits: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[int, KMeans]:
    """8. 两组 K 在同一份 10 维标准化训练特征上独立拟合，n_init=10 取较好的初始化。"""
    x_train = splits["train"][0]
    return {
        k: KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=KMEANS_N_INIT).fit(x_train)
        for k in CLUSTER_COUNTS
    }


def cluster_metrics(
    splits: dict[str, tuple[np.ndarray, np.ndarray]], clusters: dict[int, KMeans]
) -> pd.DataFrame:
    """9. 每个 K 一行：轮廓系数、随机种子、n_init、拟合使用的特征维数。"""
    x_train = splits["train"][0]
    return pd.DataFrame(
        [
            {
                "K": k,
                "silhouette": silhouette_score(x_train, model.labels_, metric="euclidean"),
                "random_state": RANDOM_STATE,
                "n_init": KMEANS_N_INIT,
                "n_features": model.n_features_in_,
            }
            for k, model in clusters.items()
        ]
    )


def cluster_summary(
    raw_splits: dict[str, tuple[pd.DataFrame, np.ndarray]], clusters: dict[int, KMeans]
) -> pd.DataFrame:
    """10. 每组一行：样本数、原始 bmi 与 bp 均值；labels_ 与未标准化训练行顺序一致。"""
    described = raw_splits["train"][0].reset_index(drop=True)[list(DESCRIBED_FEATURES)]
    rows = []
    for k, model in clusters.items():
        labels = pd.Series(model.labels_, name="cluster")
        sizes = labels.value_counts().sort_index()
        if int(sizes.sum()) != len(described):
            raise ValueError(
                f"K={k}: 各组样本数之和 {int(sizes.sum())} 不等于训练样本数 {len(described)}"
            )
        means = described.groupby(labels).mean()
        rows.extend(
            {
                "K": k,
                "cluster": int(cluster_id),
                "samples": int(sizes[cluster_id]),
                "bmi_mean": values["bmi"],
                "bp_mean": values["bp"],
            }
            for cluster_id, values in means.iterrows()
        )
    return pd.DataFrame(rows)


def plot_clusters(
    pca: PCA, projection: np.ndarray, clusters: dict[int, KMeans]
) -> None:
    """11. 双面板：同一批 PC1/PC2 坐标按组号着色，中心从 10 维投影到同一平面。"""
    centers = {k: pca.transform(model.cluster_centers_) for k, model in clusters.items()}
    xs = np.concatenate([projection[:, 0], *(center[:, 0] for center in centers.values())])
    ys = np.concatenate([projection[:, 1], *(center[:, 1] for center in centers.values())])
    x_margin = AXIS_MARGIN * (xs.max() - xs.min())
    y_margin = AXIS_MARGIN * (ys.max() - ys.min())
    limits = {
        "x": (xs.min() - x_margin, xs.max() + x_margin),
        "y": (ys.min() - y_margin, ys.max() + y_margin),
    }

    figure, axes = plt.subplots(1, len(CLUSTER_COUNTS), figsize=(12, 5.2), sharex=True, sharey=True)
    for axis, (k, model) in zip(np.atleast_1d(axes), clusters.items()):
        for cluster_id, color in enumerate(CLUSTER_COLORS[:k]):
            mask = model.labels_ == cluster_id
            axis.scatter(
                projection[mask, 0],
                projection[mask, 1],
                s=POINT_SIZE,
                color=color,
                alpha=0.65,
                label=f"cluster {cluster_id} (n={int(mask.sum())})",
            )
        axis.scatter(
            centers[k][:, 0],
            centers[k][:, 1],
            marker="X",
            s=CENTER_SIZE,
            color="black",
            edgecolor="white",
            linewidth=1.0,
            label="projected center",
            zorder=3,
        )
        axis.set_xlim(*limits["x"])
        axis.set_ylim(*limits["y"])
        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")
        axis.set_title(f"K = {k}")
        axis.legend(fontsize=8, loc="best")
    figure.suptitle(
        "K-means on 10 standardized features (10D clustering, 2D display via the fitted PCA)"
    )
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "cluster_comparison.png", dpi=FIGURE_DPI)
    plt.close(figure)


def main() -> None:
    """按任务顺序编排两个实践，两者共用同一份划分与标准化结果。"""
    OUTPUT_DIR.mkdir(exist_ok=True)

    raw_splits = split_rows()
    splits = scale_splits(raw_splits)
    pcas = fit_pcas(splits)
    pca_splits = {n: project(pcas[n], splits) for n in PCA_COMPONENTS}

    models = train_models(splits, pca_splits)
    metrics = evaluate_and_select(splits, pca_splits, models)
    metrics.to_csv(
        OUTPUT_DIR / "metrics.csv", index=False, encoding="utf-8-sig", float_format="%.4f"
    )

    pca_summary = pca_information(splits, pcas)
    pca_summary.to_csv(
        OUTPUT_DIR / "pca_summary.csv", index=False, encoding="utf-8-sig", float_format="%.4f"
    )

    clusters = fit_clusters(splits)
    cluster_metrics(splits, clusters).to_csv(
        OUTPUT_DIR / "cluster_metrics.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.4f",
    )
    cluster_summary(raw_splits, clusters).to_csv(
        OUTPUT_DIR / "cluster_summary.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.4f",
    )
    plot_clusters(pcas[PCA_COMPONENTS[0]], pca_splits[PCA_COMPONENTS[0]]["train"][0], clusters)


if __name__ == "__main__":
    main()
