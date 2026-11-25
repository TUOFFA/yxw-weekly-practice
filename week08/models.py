"""定义单头缩放点积自注意力预测器。"""

import math

import torch
from torch import nn


class SingleHeadAttentionPredictor(nn.Module):
    """用第 24 个位置的自注意力读出预测下一小时油温。"""

    def __init__(self, model_dim=16):
        super().__init__()
        self.model_dim = model_dim
        self.input_projection = nn.Linear(1, model_dim)
        self.query = nn.Linear(model_dim, model_dim)
        self.key = nn.Linear(model_dim, model_dim)
        self.value = nn.Linear(model_dim, model_dim)
        self.output = nn.Linear(model_dim, 1)

    def attention_tensors(self, inputs, add_position=False):
        """返回输入映射、Q/K/V、分数、权重和加权结果。"""
        embedded = self.input_projection(inputs)
        if add_position:
            embedded = embedded + simple_position_encoding(
                embedded.shape[1], self.model_dim, embedded.device, embedded.dtype
            )
        queries = self.query(embedded)
        keys = self.key(embedded)
        values = self.value(embedded)
        scores = queries @ keys.transpose(-2, -1) / math.sqrt(self.model_dim)
        weights = scores.softmax(dim=-1)
        contexts = weights @ values
        return embedded, queries, keys, values, scores, weights, contexts

    def forward(self, inputs):
        """使用模型自己计算的权重完成标准化油温预测。"""
        *_, contexts = self.attention_tensors(inputs)
        return self.output(contexts[:, -1, :])

    def inspect_forward(self, inputs, add_position=False):
        """在一次前向中返回所有需要核对的中间量。"""
        tensors = self.attention_tensors(inputs, add_position=add_position)
        prediction = self.output(tensors[-1][:, -1, :])
        return (prediction, *tensors)

    def predict_with_reading(self, inputs, reading="original"):
        """只替换末位汇总权重，保持同一组 V 和预测层不变。"""
        *_, values, scores, weights, contexts = self.attention_tensors(inputs)
        if reading == "original":
            last_context = contexts[:, -1, :]
        elif reading == "mean":
            last_context = values.mean(dim=1)
        elif reading == "last":
            last_context = values[:, -1, :]
        else:
            raise ValueError(f"未知读法：{reading}")
        return self.output(last_context), values, scores, weights, last_context


def simple_position_encoding(length, model_dim, device, dtype):
    """只在首个分量加入从 0 到 1 的位置数。"""
    positions = torch.zeros((1, length, model_dim), device=device, dtype=dtype)
    positions[0, :, 0] = torch.linspace(0, 1, length, device=device, dtype=dtype)
    return positions


def count_parameters(model):
    """统计全部可训练参数。"""
    return sum(parameter.numel() for parameter in model.parameters())
