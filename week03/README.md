# 第三周实践：多特征线性回归与 Ridge

> 运行入口为 `python multi_feature_ridge.py`。第三周直接使用一个脚本完成数据准备、训练、评价和绘图。

用 diabetes 数据的 **10 个特征** 拟合线性回归，并与 `alpha = 1`、`alpha = 100` 两组 Ridge 对照。
选型只看验证集，测试集只评价一次。

## 代码做什么

按任务顺序走 6 步，产物全部写进 `results/`：

1. **固定划分**：对 442 条样本行划分一次（`random_state=42`，60/20/20 → 265 / 88 / 89），单特征与多特征取同一批行；与第二周是同一划分（单特征验证/测试 RMSE 60.8331 / 67.0440 与 week02 一致）。
2. **按列标准化**：`StandardScaler` 只在训练特征上 `fit`，同一组统计量变换验证与测试；`target` 保持原样。
3. **训练五个模型**：均值基线、单特征 `bmi`、多特征（10 列）、`Ridge alpha=1`、`Ridge alpha=100`，共用同一组训练样本，各自独立拟合。
4. **比较训练与验证 → 选型 → 测试**：5 个模型 ×（训练、验证）指标长表 → 训练/验证 RMSE 分组柱状图 → 按验证 RMSE 在两组 Ridge 中选定 alpha → 基线、单特征、多特征、选定 Ridge 在测试集上各评价一次。
5. **观察权重**：三个多特征模型的权重宽表（列名即特征名）+ 同图分组柱状图。
6. **事实清单**：`run_summary.txt`，只有数值与结论，没有推导段。

## 数据从哪里来

`sklearn.datasets.load_diabetes(as_frame=True, scaled=False)`，随 scikit-learn 附带，无需联网。

- 442 行 × 11 列（10 个特征 + `target`）；
- 输入列：`age, sex, bmi, bp, s1..s6`；`target` 不进输入，表格行号不当时间；
- `scaled=False` 保留原始量纲，所以标准化参数必须自己算。

## 主要文件

| 文件 | 作用 |
| --- | --- |
| `multi_feature_ridge.py` | 主脚本，运行即复现全部数字与两张图 |
| `代码讲解.md` | 逐段讲解代码结构、公式，以及组会三问的实测答案 |
| `README.md` | 本文件：数据、运行、产物、结果 |
| `requirements.txt` | 依赖清单及版本 |
| `results/` | 运行时生成的输出目录 |

## 安装环境与运行

Python 3.14.6、numpy 2.5.0、pandas 3.0.3、matplotlib 3.11.0、scikit-learn 1.9.0。

```bash
pip install -r requirements.txt
python multi_feature_ridge.py
```

脚本没有命令行参数，产物固定写到脚本同级的 `results/`，从任意目录执行结果一样。

## 结果保存在哪里

本周必做四项，外加一份事实清单：

| 文件 | 内容 |
| --- | --- |
| `results/metrics_comparison.csv` | 一份指标比较表：`model, split, MAE, RMSE, MSE, selected`，`selected` 列标出选定配置 |
| `results/weights.csv` | 一份权重表：3 行 × 10 个特征列 + `l2_norm` |
| `results/rmse_train_validation.png` | 图 1：训练与验证 RMSE 的分组柱状图 |
| `results/weights_comparison.png` | 图 2：多特征线性回归与两组 Ridge 的权重同图 |
| `results/run_summary.txt` | 事实清单（与控制台输出同一份） |

## 关键设置

- 选定 `alpha = 1`：验证 RMSE 49.2033 < 50.8149（`alpha = 100`）；`metrics_comparison.csv` 中该配置 `selected=True`；
- 测试集只在选定配置之后评价一次，未合并训练集与验证集重训；
- 权重表与权重图对应**标准化后**的输入；`alpha` 是正则化强度，与梯度下降的学习率无关。

## 实测结果

| 模型 | 训练 MAE | 训练 RMSE | 验证 MAE | 验证 RMSE | 测试 MAE | 测试 RMSE |
| --- | --- | --- | --- | --- | --- | --- |
| Mean baseline | 64.6565 | 77.3308 | 67.5162 | 76.1614 | 66.7289 | 76.9057 |
| LinearRegression (bmi) | 50.9522 | 61.4472 | 49.2051 | 60.8331 | 55.2107 | 67.0440 |
| LinearRegression (10 features) | 44.2874 | 54.2849 | 38.2213 | 49.1497 | 46.7240 | 56.9703 |
| Ridge alpha=1 | 44.3028 | 54.2966 | 38.2939 | 49.2033 | 46.7941 | 57.0834 |
| Ridge alpha=100 | 45.7447 | 55.4401 | 40.5034 | 50.8149 | — | — |

`alpha = 100` 没有测试指标：验证集已经决定用 `alpha = 1`，按规则不再拿另一组去碰测试集。

## 结论

- **加特征**：验证 RMSE 从 60.83 降到 49.15（−11.68，约 −19%），测试 RMSE 从 67.04 降到 56.97；
- **选 alpha = 1**：两组 Ridge 的验证 RMSE 差 1.61（49.20 vs 50.81），弱正则化这一组更准；
- **正则化没赢**：两组 Ridge 的训练/验证/测试误差都略高于不加正则的线性回归，`alpha = 100` 差得更明显——约束权重只保证权重变小（L2 范数 57.96 → 50.57 → 31.93），不保证预测更准。
