import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    """
    WHAT: One complete Transformer layer (attention + FFN with residuals).
    WHY: Stack N of these to build a deep language model.

         Architecture (Pre-Norm):
         ┌─────────────────────────────────────┐
         │ x = x + Attention(RMSNorm(x), mask) │  ← Mix information BETWEEN tokens
         │ x = x + SwiGLU(RMSNorm(x))          │  ← Process information WITHIN tokens
         └─────────────────────────────────────┘

         Each sublayer: normalize FIRST (pre-norm), then compute,
         then ADD back the original (residual connection).

         Without residuals: deep networks can't train (vanishing gradients)
         Without pre-norm: training is unstable at large depths
         Without FFN: no non-linear processing per token
         Without attention: no information mixing between tokens
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        # WHAT: First normalization — before attention
        # WHY: Pre-norm: clean, well-scaled input → stable attention computation
        self.norm1 = RMSNorm(d_model)

        # WHAT: Multi-head self-attention with RoPE and causal masking
        # WHY: The core mechanism that lets tokens "talk to" each other
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)

        # WHAT: Second normalization — before FFN
        # WHY: FFN expects normalized input for consistent behavior across layers
        self.norm2 = RMSNorm(d_model)

        # WHAT: SwiGLU feed-forward network
        # WHY: Non-linear processing per token. Without this, stacking more
        #      attention layers would be no more powerful than one layer.
        self.ffn = SwiGLU(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass: norm → sublayer → add residual.
        Executed twice: once for attention, once for FFN.
        """

        # ===== SUB-LAYER 1: Self-Attention with residual =====
        # WHAT: x = x + Attention(Norm(x))
        # WHY: The model learns what CHANGES (the delta) to make to x,
        #      not what to replace x with entirely. This is easier to learn.
        #      If attention can't improve things, it can output near-zero.
        x = x + self.attention(self.norm1(x), mask)

        # ===== SUB-LAYER 2: Feed-Forward with residual =====
        # WHAT: x = x + FFN(Norm(x))
        # WHY: Same residual pattern. After mixing information via attention,
        #      each token "thinks" independently via the FFN.
        #      Attention = group discussion. FFN = private reflection.
        x = x + self.ffn(self.norm2(x))

        return x