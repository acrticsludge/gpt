import torch
import torch.nn as nn
import math


class RMSNorm(nn.Module):
    """
    WHAT: Root Mean Square Normalization.
    WHY: Stable layer normalization without bias.
         Used in LLaMA, Mistral, GPT-3+ to avoid the instability
         issues of LayerNorm with bias at large scales.
    """

    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  [batch, seq, d_model]
        Output: [batch, seq, d_model] — normalized
        """
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight