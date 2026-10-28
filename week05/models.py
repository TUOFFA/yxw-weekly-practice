"""定义实践一的线性模型和实践二的单隐藏层 MLP。"""

import torch
from torch import nn


DEFAULT_HIDDEN_DIM = 16


def create_linear_model():
    """创建权重和偏置均为零的单层线性模型。"""
    model = nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.fill_(0.0)
        model.bias.fill_(0.0)
    return model


def create_mlp(input_dim, hidden_dim=DEFAULT_HIDDEN_DIM):
    """hidden_dim 同时连接第一层输出和第二层输入。"""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )


def forward_shapes(model, x):
    """逐层前向，返回输入和各层输出形状。"""
    shapes = [tuple(x.shape)]
    output = x
    with torch.no_grad():
        for layer in model:
            output = layer(output)
            shapes.append(tuple(output.shape))
    return shapes
