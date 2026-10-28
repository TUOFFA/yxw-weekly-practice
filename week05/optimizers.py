"""按入口配置创建 PyTorch 优化器。"""

import torch


def create_optimizer(parameters, name, learning_rate):
    """创建不带动量和权重衰减的 SGD，或默认参数的 Adam。"""
    if name == "SGD":
        return torch.optim.SGD(
            parameters, lr=learning_rate, momentum=0, weight_decay=0
        )
    if name == "Adam":
        return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=0)
    raise ValueError(f"不支持的优化器: {name}")
