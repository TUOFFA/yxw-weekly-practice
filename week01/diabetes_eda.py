"""Diabetes 数据检查与两张基础图。"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

OUTPUT_DIR = Path("results")
FIGURE_DPI = 150
HISTOGRAM_BINS = 20

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    dataset = load_diabetes(as_frame=True, scaled=False)
    frame = dataset.frame
    features = list(dataset.feature_names)
    target = dataset.target.name
    missing = frame.isna().sum()
    correlation = frame["bmi"].corr(frame[target])

    summary = [
        f"shape: {frame.shape[0]} rows x {frame.shape[1]} columns",
        f"features ({len(features)}): {', '.join(features)}",
        f"target: {target}", "", "missing values:", missing.to_string(),
        f"missing total: {int(missing.sum())}", "", "descriptive statistics:",
        frame.describe().T.to_string(), "", f"bmi-target correlation: {correlation:.6f}"]
    (OUTPUT_DIR / "summary.txt").write_text("\n".join(summary), encoding="utf-8")

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(frame[target], bins=HISTOGRAM_BINS, color="#4C78A8", edgecolor="white")
    axis.set(xlabel="Disease progression (target)", ylabel="Number of samples",
             title="Distribution of diabetes target")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "target_distribution.png", dpi=FIGURE_DPI)
    plt.close(figure)

    x, y = frame["bmi"], frame[target]
    x_pad, y_pad = (x.max()-x.min())*0.01, (y.max()-y.min())*0.01
    figure, axis = plt.subplots(figsize=(10, 8))
    axis.scatter(x, y, alpha=0.7, color="#F58518")
    axis.set_xlim(x.min()-x_pad, x.max()+x_pad)
    axis.set_ylim(y.min()-y_pad, y.max()+y_pad)
    axis.set(xlabel="BMI", ylabel="Disease progression (target)", title="BMI vs target")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "bmi_vs_target.png", dpi=FIGURE_DPI)
    plt.close(figure)
    print(f"shape={frame.shape}, missing={int(missing.sum())}, bmi_target_correlation={correlation:.6f}")

if __name__ == "__main__":
    main()
