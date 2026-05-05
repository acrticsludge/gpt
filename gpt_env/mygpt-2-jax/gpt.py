import jax
import jax.numpy as jnp
from flax import linen as nn
from config import GPTConfig
from normalization import RMSNorm
from mlp import SwiGLU
from attention import MultiHeadAttention, create_causal_mask

class TransformerBlock(nn.Module):
    d_model: int
    num_heads: int
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray = None, training: bool = False) -> jnp.ndarray:
        norm1 = RMSNorm(self.d_model, name='norm1')
        attn = MultiHeadAttention(self.d_model, self.num_heads, self.dropout, name='attention')
        norm2 = RMSNorm(self.d_model, name='norm2')
        ffn = SwiGLU(self.d_model, self.dropout, name='ffn')

        x = x + attn(norm1(x), mask, training=training)
        x = x + ffn(norm2(x), training=training)

        return x

class GPT(nn.Module):
    config: GPTConfig

    @nn.compact
    def __call__(self, input_ids: jnp.ndarray, targets: jnp.ndarray = None, training: bool = False) -> tuple:
        batch_size, seq_len = input_ids.shape

        # Token embedding
        x = nn.Embed(num_embeddings=self.config.vocab_size, features=self.config.d_model, name='token_embedding')(input_ids)
        x = nn.Dropout(rate=self.config.embd_dropout, deterministic=not training)(x)

        # Causal mask
        mask = create_causal_mask(seq_len)

        # Transformer layers
        for _ in range(self.config.num_layers):
            x = TransformerBlock(
                d_model=self.config.d_model,
                num_heads=self.config.num_heads,
                dropout=self.config.dropout,
            )(x, mask, training=training)

        # Final norm
        x = RMSNorm(self.config.d_model, name='final_norm')(x)

        # LM head
        logits = nn.Dense(self.config.vocab_size, use_bias=False, name='lm_head')(x)

        loss = None
        if targets is not None:
            logits_flat = logits[:, :-1, :].reshape(-1, self.config.vocab_size)
            targets_flat = targets[:, 1:].reshape(-1)

            loss = jnp.mean(
                jax.nn.one_hot(targets_flat, self.config.vocab_size) *
                jax.nn.log_softmax(logits_flat)
            )
            loss = -loss

        return logits, loss
