import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNonMantis(nn.Module):
    """
    Train a CNN on top of frozen Mantis embeddings.
    Input: (B, D) where D is embedding dimension (e.g., 256 or 512)
    """

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.net(x)
