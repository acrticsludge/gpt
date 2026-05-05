import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    WHAT: SwiGLU activation — Sigmoid Linear Unit + Gated Linear Unit.
    WHY:  Best performing activation for transformer FFNs.
          Outperforms ReLU, GELU, SiLU on language tasks.
          Used in LLaMA, PaLM, Mistral, Qwen.
    """

    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.d_model = d_model

        # WHAT: Three projections: W (gate), V (value), W (output)
        # WHY: SwiGLU = silu(W_gate * x) * W_val * x
        #      This is equivalent to gating with a sigmoid-weighted value.
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  [batch, seq, d_model]
        Output: [batch, seq, d_model]
        """
        # WHAT: Gate and value projections
        # WHY: w1 provides the gating signal, w3 provides the value
        gate = F.silu(self.w1(x))
        value = self.w3(x)
        return self.dropout(self.w2(gate * value))