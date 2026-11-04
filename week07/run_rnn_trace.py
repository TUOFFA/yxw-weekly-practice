"""实践一入口：展开五步 RNN，记录状态并计算一次梯度。"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from reporting import save_table


DATA_PATH = Path("../week06/data/ETTh1.csv")
OUTPUT_DIR = Path("results")
WX_INITIAL = 0.6
WH_INITIAL = 0.5
BIAS_INITIAL = 0.0
OUTPUT_WEIGHT = 0.8
OUTPUT_BIAS = 0.1
REORDERED_INDICES = [0, 3, 2, 1, 4]


def run_sequence(inputs, source_indices, times, wx, wh, bias, case):
    state = torch.tensor(0.0)
    rows = []
    for step, source_index in enumerate(source_indices, start=1):
        value = inputs[source_index]
        previous = state
        input_term = wx * value
        memory_term = wh * previous
        state = torch.tanh(input_term + memory_term + bias)
        rows.append({
            "case": case,
            "step": step,
            "source_index": source_index,
            "source_time": times[source_index],
            "z": value.item(),
            "h_prev": previous.item(),
            "input_term": input_term.item(),
            "memory_term": memory_term.item(),
            "h": state.item(),
            "input_grad": np.nan,
        })
    prediction = OUTPUT_WEIGHT * state + OUTPUT_BIAS
    return rows, prediction


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_PATH, parse_dates=["date"]).iloc[:6]
    raw = df["OT"].to_numpy(dtype=np.float32)
    transformed = (raw - raw[0]) / 10
    inputs = torch.tensor(transformed[:5], requires_grad=True)
    target = torch.tensor(transformed[5])
    wx = torch.tensor(WX_INITIAL, requires_grad=True)
    wh = torch.tensor(WH_INITIAL, requires_grad=True)
    bias = torch.tensor(BIAS_INITIAL, requires_grad=True)
    times = df["date"].to_numpy()

    original_rows, prediction = run_sequence(
        inputs, list(range(5)), times, wx, wh, bias, "original"
    )
    loss = (prediction - target) ** 2
    loss.backward()
    for row, gradient in zip(original_rows, inputs.grad):
        row["input_grad"] = gradient.item()

    with torch.no_grad():
        step_two = torch.tanh(
            wx * inputs[1] + wh * torch.tanh(wx * inputs[0] + bias) + bias
        )
    if not np.isclose(step_two.item(), original_rows[1]["h"]):
        raise ValueError("第 2 步独立计算核对失败")

    reordered_rows, reordered_prediction = run_sequence(
        inputs.detach(), REORDERED_INDICES, times, wx.detach(), wh.detach(),
        bias.detach(), "reordered_1_4_3_2_5"
    )
    save_table(original_rows + reordered_rows, OUTPUT_DIR / "rnn_trace.csv")

    summary = {
        "prediction": prediction.item(),
        "target": target.item(),
        "loss": loss.item(),
        "wx_grad": wx.grad.item(),
        "wh_grad": wh.grad.item(),
        "b_grad": bias.grad.item(),
        "reordered_prediction": reordered_prediction.item(),
    }
    print(pd.Series(summary).to_string())


if __name__ == "__main__":
    main()
