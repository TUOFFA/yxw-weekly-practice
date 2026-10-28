# 第四周实践：PCA 降维回归对照与 K-means 聚类

> 运行入口为 `python pca_kmeans.py`。第四周直接使用一个脚本完成 PCA 回归与 K-means 实验。

在 diabetes 的 10 个特征上，比较**未降维 / PCA 2 维 / PCA 5 维**三组线性回归与均值基线；
再用同一份标准化训练特征跑 K-means（K=2、K=3）。选型只看验证集，测试集只评价一次。

## 代码做什么

`pca_kmeans.py` 一个脚本跑完两项实践，共用同一份划分与标准化；`main()` 只做编排，
11 个函数按任务顺序排列，产物全部写进 `results/`。

实践一（PCA 回归对照）：

1. **划分**：442 条样本行划分一次 60/20/20（`random_state=42` → 265 / 88 / 89）；
2. **只在训练集标准化**：`StandardScaler` 在训练特征上 `fit`，同一组统计量变换三部分，`target` 保持原样；
3. **拟合 PCA**：2 维、5 维、以及保留全部 10 维的三个 `PCA(svd_solver="full", whiten=False)` 都在标准化训练特征上拟合，三部分都用训练集学到的方向 `transform`；
4. **训练四组模型**：均值基线、未降维线性回归、PCA 2 维回归、PCA 5 维回归；
5. **比较并选配置**：输出训练/验证 MAE、RMSE + 训练/验证 RMSE 分组柱状图，按验证 RMSE 在两组 PCA 中选维数，再给基线、未降维、选定的 PCA 补上测试指标；
6. **累计解释方差**：2 维、5 维的累计比例，以及 10 维 PCA 画的 1–10 维曲线；
7. **重构 MSE**：两个降维 PCA 在训练、验证上的 `inverse_transform` 误差。

实践二（K-means）：

8. **拟合两组配置**：`KMeans(n_clusters=2/3, random_state=42, n_init=10)`，输入是 10 维标准化训练特征；
9. **轮廓系数**：`silhouette_score(..., metric="euclidean")`；
10. **组内统计**：每组样本数与**原始** `bmi`、`bp` 均值，并核对各组样本数之和等于 265；
11. **双面板聚类图**：左 K=2、右 K=3，同一批 PC1/PC2 坐标与坐标范围，中心由 10 维投影到该平面。

## 数据从哪里来

`sklearn.datasets.load_diabetes(as_frame=True, scaled=False)`，随 scikit-learn 附带，无需联网。

- 442 行 × 11 列（10 个特征 + `target`）；
- 输入列：`age, sex, bmi, bp, s1..s6`；`scaled=False` 保留原始量纲，所以标准化参数必须自己算；
- 聚类只用特征，`target` 不参与分组。

## 主要文件

| 文件 | 作用 |
| --- | --- |
| `pca_kmeans.py` | 主脚本，运行即复现全部四表三图 |
| `代码讲解.md` | 逐函数讲解代码结构、公式，以及组会问题的实测答案 |
| `README.md` | 本文件：数据、运行、产物、结果 |
| `requirements.txt` | 依赖清单及版本 |
| `results/` | 运行时生成的输出目录 |

## 安装环境与运行

Python 3.14.6、numpy 2.5.0、pandas 3.0.3、matplotlib 3.11.0、scikit-learn 1.9.0。

```bash
pip install -r requirements.txt
python pca_kmeans.py
```

脚本没有命令行参数，产物固定写到脚本同级的 `results/`，从任意目录执行结果一样。
所有随机环节都固定了 `random_state=42`（划分、`KMeans`；PCA 的 `svd_solver="full"` 本身确定），重复运行数字不变。

## 结果保存在哪里

| 文件 | 内容 |
| --- | --- |
| `results/metrics.csv` | 指标比较表：`model, pca_components, train_MAE, train_RMSE, validation_MAE, validation_RMSE, test_MAE, test_RMSE, selected`；未降维两行 `pca_components` 留空，未入选的 PCA 行测试列留空 |
| `results/pca_summary.csv` | PCA 信息表：`components, cumulative_explained_variance, train_reconstruction_mse, validation_reconstruction_mse` |
| `results/rmse_comparison.png` | 图 1：四组模型的训练与验证 RMSE 分组柱状图 |
| `results/pca_variance.png` | 图 2：1–10 维累计解释方差曲线，标出 2 维、5 维 |
| `results/cluster_metrics.csv` | 聚类指标表：`K, silhouette, random_state, n_init, n_features` |
| `results/cluster_summary.csv` | 组内统计表：`K, cluster, samples, bmi_mean, bp_mean` |
| `results/cluster_comparison.png` | 图 3：K=2 / K=3 双面板，PC1–PC2 平面，标注投影后的中心 |

## 关键设置

- 选定 **PCA 5 维**：验证 RMSE 48.9917 < 2 维的 61.5189；`metrics.csv` 中该行 `selected=True`；
- 划分、标准化、PCA 都只用训练集 `fit`，验证/测试只 `transform`；测试集在选定维数之后才第一次使用；
- 重构 MSE 在**标准化特征空间**比较：`X_scaled` 对 `inverse_transform(transform(X_scaled))`，平方差对 265×10（或 88×10）个格子取平均；
- 聚类输入是 10 维标准化特征的 265 条训练样本；组内统计用**未标准化**的 `bmi`、`bp`，行顺序与 `labels_` 一一对应；图上的横纵轴是 PC1/PC2（复用 2 维 PCA），不是聚类空间。

## 实测结果

| 模型 | 训练 MAE | 训练 RMSE | 验证 MAE | 验证 RMSE | 测试 MAE | 测试 RMSE |
| --- | --- | --- | --- | --- | --- | --- |
| Mean baseline | 64.6565 | 77.3308 | 67.5162 | 76.1614 | 66.7289 | 76.9057 |
| LinearRegression (10 features) | 44.2874 | 54.2849 | 38.2213 | 49.1497 | 46.7240 | 56.9703 |
| PCA 2D + LinearRegression | 52.1983 | 62.8068 | 50.6151 | 61.5189 | — | — |
| PCA 5D + LinearRegression | 45.7253 | 55.8442 | 38.9891 | 48.9917 | 46.9023 | 58.2081 |

| 维数 | 累计解释方差 | 训练重构 MSE | 验证重构 MSE |
| --- | --- | --- | --- |
| 2 | 0.5488 | 0.4512 | 0.4807 |
| 5 | 0.8351 | 0.1649 | 0.1735 |

| K | 轮廓系数 | random_state | n_init | 拟合特征维数 |
| --- | --- | --- | --- | --- |
| 2 | 0.2334 | 42 | 10 | 10 |
| 3 | 0.1862 | 42 | 10 | 10 |

| K | 组号 | 样本数 | 原始 bmi 均值 | 原始 bp 均值 |
| --- | --- | --- | --- | --- |
| 2 | 0 | 130 | 24.1115 | 88.3845 |
| 2 | 1 | 135 | 29.0185 | 100.9630 |
| 3 | 0 | 124 | 24.1065 | 88.1424 |
| 3 | 1 | 81 | 30.0481 | 104.6874 |
| 3 | 2 | 60 | 27.1483 | 95.1775 |

与第三周对照：同一次划分、同一组标准化下，未降维线性回归的测试 MAE / RMSE 仍是 46.7240 / 56.9703，基线仍是 66.7289 / 76.9057，说明本周复用了第三周的样本划分。

## 结论

- **2 维压得太狠**：只保留 54.88% 的输入方差，验证 RMSE 61.52，比未降维的 49.15 高 12.37（MAE 高 12.39）；
- **5 维接近未降维**：保留 83.51% 方差，验证 RMSE 48.99 与 10 维的 49.15 相差 0.16（约 0.3%），测试 RMSE 58.21 反而比 10 维的 56.97 高 1.24；
- **选 5 维是按规则的结果**：验证 RMSE 48.9917 < 61.5189，验证集赢 12.53；但测试上没有随降维变好，说明保留更多输入方差并不保证预测更准；
- **重构误差与解释方差同向**：2 维 → 5 维，重构 MSE 0.4512 → 0.1649（约 −63.5%），累计解释方差 0.5488 → 0.8351；在标准化输入上重构 MSE 恰好等于 `1 - 累计解释方差比例`；
- **K=3 没有更好划分**：轮廓系数 0.2334 → 0.1862 下降，多出的第 3 组是从 K=2 的高 bmi/bp 组（135 条）里拆出来的 79 + 56 条，但拆开后的两组在 PC1–PC2 平面上与其他组大量重叠，图上看不出干净的边界。
