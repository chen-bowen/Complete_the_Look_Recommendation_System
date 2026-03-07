import torch.nn as nn


def init_weights(m):
    """Apply Xavier initialization on linear layer weights"""
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
