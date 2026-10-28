# 第五周实践：MLP 与 PyTorch 训练基础

本周包含两个入口：实践一用两个样本观察一次 SGD 更新；实践二在 diabetes 十特征数据上比较均值基线、线性回归、`MLP + SGD` 和 `MLP + Adam`。选定 MLP 只依据验证 RMSE，测试集不参与选型。

## 环境与运行

实测环境：Python 3.12.14、PyTorch 2.5.1 CPU、NumPy 2.5.4、pandas 3.1.0、Matplotlib 3.11.2、scikit-learn 1.9.1。

```bash
pip install -r requirements.txt
python run_step.py
python run_mlp.py
```

两个脚本没有命令行参数，从任意目录执行都将产物写到本目录的 `results/`。运行过程不打印教学旁白。

## 文件职责

| 文件 | 作用 |
| --- | --- |
| `data.py` | 两样本数据；diabetes 划分、标准化、张量与 DataLoader |
| `models.py` | 零初始化线性模型、可设置隐藏宽度的单隐藏层 MLP、前向形状检查 |
| `optimizers.py` | 根据入口配置创建 SGD 或 Adam |
| `training.py` | 一次更新、逐批逐轮训练、最佳参数记录、基线拟合 |
| `evaluation.py` | 无梯度评价、恢复目标尺度、MAE/RMSE/MSE |
| `reporting.py` | 三份 CSV 与两张 PNG 的保存和绘制 |
| `run_step.py` | 实践一配置与调用编排 |
| `run_mlp.py` | 实践二配置、实验遍历与指标组装 |
| `代码讲解.md` | 按组件解释输入、计算、输出及组会重点 |

## 实验设置

- diabetes 数据按 `random_state=42` 划分为 265/88/89 条训练、验证、测试样本。
- 特征的 `StandardScaler` 只在训练集拟合；目标使用训练集均值和总体标准差标准化。
- MLP 为 `10 → 16 → 16 → 1` 的层输出形状，其中中间两次 16 分别是 Linear 输出和 ReLU 输出。
- 两组 MLP 复制同一份随机初始参数，并使用相同种子的独立 DataLoader。
- SGD：学习率 0.01；Adam：学习率 0.001；均为 batch size 32、200 epochs、MSELoss、CPU float32。
- 每轮更新后重新评价完整训练集和验证集，保存验证 MSE 最小时的参数；相同值保留较早轮次。
- 测试集只评价均值基线、线性回归和验证 RMSE 更低的 MLP。

## 隐藏层宽度练习

`models.py` 的 `create_mlp()` 中，`hidden_dim` 同时控制第一层 `Linear(10, hidden_dim)` 的输出维数和第二层 `Linear(hidden_dim, 1)` 的输入维数。`run_mlp.py` 另建宽度 8 的临时网络并用断言检查：

```text
(32, 10) → (32, 8) → (32, 8) → (32, 1)
```

该网络只做前向检查，不训练、不加入正式结果。

## 输出与实测结果

| 文件 | 内容 |
| --- | --- |
| `results/step_trace.csv` | 清梯度后、反传后、更新后三阶段的参数、梯度和损失 |
| `results/history.csv` | SGD、Adam 各 200 轮的训练/验证标准化目标 MSE |
| `results/metrics.csv` | 四组模型的训练/验证指标及规定模型的测试指标，均为原始目标单位 |
| `results/loss_curves.png` | 两组 MLP 的训练/验证 MSE 双面板曲线和最佳轮次 |
| `results/rmse_comparison.png` | 四组模型的训练/验证 RMSE 对照 |

实践一初始 MSE 为 10，反传得到权重梯度 -10、偏置梯度 -6；学习率 0.1 更新后权重为 1、偏置为 0.6，重新计算的 MSE 为 1.06。

实践二实测中，SGD 最佳轮次为 167，验证 RMSE 为 50.4665；Adam 最佳轮次为 88，验证 RMSE 为 49.8631，因此选定 Adam。选定模型测试 MAE/RMSE 为 46.1523/55.9007；线性回归测试 MAE/RMSE 为 46.7240/56.9703。结果不要求 MLP 一定超过线性回归，本次运行中选定 MLP 略低于线性回归。
