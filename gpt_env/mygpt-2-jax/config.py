from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 50257
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 8
    max_seq_len: int = 512

    batch_size: int = 4
    grad_accum_steps: int = 4
    max_steps: int = 20000
    warmup_steps: int = 500

    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    dropout: float = 0.1
    embd_dropout: float = 0.1

    @property
    def head_dim(self):
        return self.d_model // self.num_heads
