import torch
import torch.nn as nn
from typing import List

class MLPClassifier(nn.Module):
    """
    Multi-Layer Perceptron for final classification.
    """
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: List[int] = [256],
        dropout: float = 0.1
    ):
        super().__init__()
        layers = []
        curr_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            curr_dim = h_dim

        layers.append(nn.Linear(curr_dim, num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
