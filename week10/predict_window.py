"""载入最佳模型并预测一条验证窗口。"""

import argparse
from pathlib import Path

import torch

from models import TransformerEncoderPredictor
from run_transformer import load_data


CHECKPOINT_PATH = Path("results/selected_transformer_encoder.pt")


def parse_args() -> argparse.Namespace:
    """读取要预测的验证窗口下标。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """恢复最佳参数，预测并打印形状与原始尺度油温。"""
    args = parse_args()
    _, raw_splits, standardized_splits, _, _ = load_data()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model = TransformerEncoderPredictor(
        checkpoint["model_dim"],
        checkpoint["head_dim"],
        checkpoint["feedforward_dim"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    index = args.validation_index
    inputs = torch.from_numpy(standardized_splits["validation"]["X"][index:index + 1])
    with torch.no_grad():
        result = model.inspect_forward(inputs)
    prediction = float(result["prediction"].item()) * checkpoint["std"] + checkpoint["mean"]
    target_time = raw_splits["validation"]["target_time"][index]
    truth = float(raw_splits["validation"]["y"][index])
    print(f"target_time={target_time}")
    print(f"encoder_output_shape={tuple(result['E'].shape)}")
    print(f"last_row_shape={tuple(result['last_row'].shape)}")
    print(f"prediction_ot={prediction:.6f}")
    print(f"truth_ot={truth:.6f}")


if __name__ == "__main__":
    main()
