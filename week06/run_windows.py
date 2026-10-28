"""实践一入口：比较两种窗口长度并保存实际样本。"""

from pathlib import Path

from data import load_etth1, make_windows
from reporting import save_table


DATA_PATH = Path("data/ETTh1.csv")
OUTPUT_DIR = Path("results")
WINDOW_SIZES = [3, 5]
SHORT_LENGTH = 12


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = load_etth1(DATA_PATH).iloc[:SHORT_LENGTH]
    series = df["OT"].to_numpy(dtype=float)
    times = df["date"].to_numpy()
    rows = []

    for window_size in WINDOW_SIZES:
        X, y, target_index = make_windows(series, window_size)
        print(f"L={window_size}: X{X.shape}, y{y.shape}, target_index{target_index.shape}")
        for position in [0, len(y) - 1]:
            index = target_index[position]
            print(
                f"L={window_size}, sample={position}, "
                f"input={times[index-window_size]}..{times[index-1]}, "
                f"values={X[position].tolist()}, target={times[index]}, y={y[position]:.3f}"
            )
        for values, target, index in zip(X, y, target_index):
            rows.append({
                "window_size": window_size,
                "input_start": int(index - window_size),
                "input_end": int(index - 1),
                "target_index": int(index),
                "target_time": times[index],
                "input_values": str(values.tolist()),
                "target_value": target,
            })

    save_table(rows, OUTPUT_DIR / "window_examples.csv")


if __name__ == "__main__":
    main()
