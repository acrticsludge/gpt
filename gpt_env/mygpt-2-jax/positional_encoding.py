import jax
import jax.numpy as jnp
import numpy as np

class RotaryPositionalEmbedding:
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute cos/sin cache
        inv_freq = 1.0 / (self.base ** (jnp.arange(0, dim, 2).astype(jnp.float32) / dim))
        t = jnp.arange(max_seq_len, dtype=jnp.float32)
        freqs = jnp.einsum('i,j->ij', t, inv_freq)
        emb = jnp.concatenate([freqs, freqs], axis=-1)

        self.cos_cached = jnp.cos(emb)
        self.sin_cached = jnp.sin(emb)

    def __call__(self, x: jnp.ndarray, seq_len: int) -> jnp.ndarray:
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]

        # x shape: [batch, heads, seq, head_dim]
        x_rot = jnp.concatenate([-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]], axis=-1)
        return x * cos + x_rot * sin
