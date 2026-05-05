from dataclasses import dataclass
import torch


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    d_model: int = 768
    num_heads: int = 12
    num_layers: int = 12
    max_seq_len: int = 1024
    dropout: float = 0.1
    embd_dropout: float = 0.1
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 500
    max_steps: int = 10000
    batch_size: int = 8
    grad_accum_steps: int = 4
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8

    def __post_init__(self):
        assert self.d_model % self.num_heads == 0