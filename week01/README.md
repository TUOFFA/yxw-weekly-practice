# diabetes_eda.py

> 运行入口为 `python diabetes_eda.py`。第一周任务较小，读取、统计和绘图直接保留在一个文件中，结果统一写入 `results/`。

本说明对应 `diabetes_eda.py`（目录中没有 `diabete_eda.py`）。

## 代码做什么

对 sklearn 自带的 diabetes 数据集做基础检查，**不训练模型**：

- 区分 10 个特征列与 1 个目标列（`target`），打印形状与列名；
- 统计每列缺失值（`df.isna().sum()`）和数值列描述统计（`df.describe().T`）；
- 画两张图：`target` 的分布直方图（20 个分箱）、`bmi` 对 `target` 的散点图。

## 数据从哪里来

由 `sklearn.datasets.load_diabetes(as_frame=True, scaled=False)` 载入，随 scikit-learn 附带，**无需联网或手动下载**。

- 442 行 × 11 列（10 特征 + 1 目标）；
- 特征为 `age, sex, bmi, bp, s1, s2, s3, s4, s5, s6`，目标为 `target`（一年后疾病进展的连续值）；
- `scaled=False` 保留原始量纲，因此图上的 `bmi` 是真实取值（18.0–42.2）。默认的 `True` 会先对特征做中心化/缩放。

## 主要文件

| 文件 | 作用 |
| --- | --- |
| `diabetes_eda.py` | 主脚本，全部逻辑都在这里 |
| `requirements.txt` | 依赖清单及版本 |
| `result/` | 运行时生成的输出目录（存放上一次运行的结果） |

## 安装环境与运行

依赖版本：Python 3.14.6、pandas 3.0.3、matplotlib 3.11.0、scikit-learn 1.9.0。

```bash
pip install -r requirements.txt
python diabetes_eda.py
```

脚本无命令行参数。产物目录固定为脚本同级的 `result/`，因此从任意目录执行，结果都写到同一处。

## 结果保存在哪里

全部写入 `result/`：

| 文件 | 内容 |
| --- | --- |
| `result/summary.txt` | 形状、列名、逐列缺失值、`describe().T` 统计（UTF-8） |
| `result/target_distribution.png` | `target` 分布直方图，20 分箱，dpi 150 |
| `result/bmi_vs_target.png` | `bmi` 对 `target` 散点图，dpi 150 |
