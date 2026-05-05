import jax
import jax.numpy as jnp
from flax import linen as nn
import math
from positional_encoding import RotaryPositionalEmbedding

class MultiHeadAttention(nn.Module):
    d_model: int
    num_heads: int
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray = None, training: bool = False) -> jnp.ndarray:
        batch_size, seq_len, _ = x.shape
        head_dim = self.d_model // self.num_heads

        # QKV projection
        qkv = nn.Dense(3 * self.d_model, use_bias=False, name='qkv_proj')(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, head_dim)
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))

        q, k, v = qkv[0], qkv[1], qkv[2]

        # RoPE
        rope = RotaryPositionalEmbedding(head_dim, max_seq_len=seq_len)
        q = rope(q, seq_len)
        k = rope(k, seq_len)

        # Attention scores
        scores = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) / math.sqrt(head_dim)

        # Causal mask
        if mask is not None:
            scores = jnp.where(mask == 0, float('-inf'), scores)

        # Softmax
        weights = nn.softmax(scores, axis=-1)
        weights = nn.Dropout(rate=self.dropout, deterministic=not training)(weights)

        # Attention output
        attn_output = jnp.matmul(weights, v)
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)

        # Output projection
        output = nn.Dense(self.d_model, use_bias=False, name='out_proj')(attn_output)
        output = nn.Dropout(rate=self.dropout, deterministic=not training)(output)

        return output

def create_causal_mask(seq_len: int) -> jnp.ndarray:
    mask = jnp.tril(jnp.ones((seq_len, seq_len)))
    return mask.reshape(1, 1, seq_len, seq_len)
