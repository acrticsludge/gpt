import jax
import jax.numpy as jnp
from flax import linen as nn

class SwiGLU(nn.Module):
    d_model: int
    expansion: int = 4
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        hidden_dim = self.d_model * self.expansion

        w1 = nn.Dense(hidden_dim, use_bias=False, name='w1')(x)
        w3 = nn.Dense(hidden_dim, use_bias=False, name='w3')(x)

        x = w1 * nn.swish(w3)
        x = nn.Dropout(rate=self.dropout, deterministic=not training)(x)

        w2 = nn.Dense(self.d_model, use_bias=False, name='w2')(x)
        x = nn.Dropout(rate=self.dropout, deterministic=not training)(x)

        return x
