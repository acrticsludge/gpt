import jax
import jax.numpy as jnp
from flax import linen as nn

class RMSNorm(nn.Module):
    d_model: int
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        scale = self.param('scale', nn.initializers.ones, (self.d_model,))

        rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return (x / rms) * scale
