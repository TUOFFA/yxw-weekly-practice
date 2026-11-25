"""双头注意力和单层 Transformer 编码器。"""

import math

import torch
from torch import nn


class AttentionHead(nn.Module):
    """把 16 维输入投影成一个 8 维缩放点积注意力头。"""

    def __init__(self, model_dim: int = 16, head_dim: int = 8):
        super().__init__()
        self.query = nn.Linear(model_dim, head_dim)
        self.key = nn.Linear(model_dim, head_dim)
        self.value = nn.Linear(model_dim, head_dim)

    def forward(
        self, inputs: torch.Tensor, use_causal_mask: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回注意力权重 A 和读出 Z。"""
        queries = self.query(inputs)
        keys = self.key(inputs)
        values = self.value(inputs)
        scores = queries @ keys.transpose(-2, -1) / math.sqrt(queries.size(-1))
        if use_causal_mask:
            length = scores.size(-1)
            future = torch.ones(
                length, length, dtype=torch.bool, device=scores.device
            ).triu(1)
            scores = scores.masked_fill(future, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        contexts = weights @ values
        return weights, contexts


class TwoHeadAttention(nn.Module):
    """用两个独立头读取同一输入，再拼接并投影回 16 维。"""

    def __init__(self, model_dim: int = 16, head_dim: int = 8):
        super().__init__()
        self.head1 = AttentionHead(model_dim, head_dim)
        self.head2 = AttentionHead(model_dim, head_dim)
        self.output_projection = nn.Linear(2 * head_dim, model_dim)

    def forward(
        self, inputs: torch.Tensor, use_causal_mask: bool = False
    ) -> torch.Tensor:
        """返回两个头合并并混合后的结果。"""
        _, context1 = self.head1(inputs, use_causal_mask)
        _, context2 = self.head2(inputs, use_causal_mask)
        concatenated = torch.cat([context1, context2], dim=-1)
        return self.output_projection(concatenated)

    def inspect_forward(
        self, inputs: torch.Tensor, use_causal_mask: bool = False
    ) -> dict[str, torch.Tensor]:
        """返回双头实验需要核对的中间张量。"""
        weights1, context1 = self.head1(inputs, use_causal_mask)
        weights2, context2 = self.head2(inputs, use_causal_mask)
        concatenated = torch.cat([context1, context2], dim=-1)
        output = self.output_projection(concatenated)
        return {
            "weights1": weights1,
            "weights2": weights2,
            "context1": context1,
            "context2": context2,
            "concatenated": concatenated,
            "output": output,
        }


class TransformerEncoderPredictor(nn.Module):
    """用一层后归一化 Transformer 编码器预测第 25 小时油温。"""

    def __init__(
        self, model_dim: int = 16, head_dim: int = 8, feedforward_dim: int = 32
    ):
        super().__init__()
        self.model_dim = model_dim
        self.input_projection = nn.Linear(1, model_dim)
        self.attention = TwoHeadAttention(model_dim, head_dim)
        self.norm1 = nn.LayerNorm(model_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(model_dim, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, model_dim),
        )
        self.norm2 = nn.LayerNorm(model_dim)
        self.predictor = nn.Linear(model_dim, 1)

    def encoder_tensors(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        """返回 X→M→H→F→E 的完整编码器路径。"""
        embedded = self.input_projection(inputs)
        embedded = embedded + simple_position_encoding(
            embedded.size(1), self.model_dim, embedded.device, embedded.dtype
        )
        attention_output = self.attention(embedded, use_causal_mask=False)
        first_norm = self.norm1(embedded + attention_output)
        feedforward_output = self.feedforward(first_norm)
        encoded = self.norm2(first_norm + feedforward_output)
        return {
            "X": embedded,
            "M": attention_output,
            "H": first_norm,
            "F": feedforward_output,
            "E": encoded,
        }

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """取编码器第 24 行预测标准化油温。"""
        encoded = self.encoder_tensors(inputs)["E"]
        return self.predictor(encoded[:, 23, :])

    def inspect_forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        """返回编码器路径、末行和最终预测。"""
        tensors = self.encoder_tensors(inputs)
        last_row = tensors["E"][:, 23, :]
        return {**tensors, "last_row": last_row, "prediction": self.predictor(last_row)}


def simple_position_encoding(
    length: int, model_dim: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """仅在首个分量加入从 0 到 1 的简化位置数。"""
    positions = torch.zeros((1, length, model_dim), device=device, dtype=dtype)
    positions[0, :, 0] = torch.linspace(0, 1, length, device=device, dtype=dtype)
    return positions


def count_parameters(model: nn.Module) -> int:
    """统计全部可训练参数。"""
    return sum(parameter.numel() for parameter in model.parameters())
