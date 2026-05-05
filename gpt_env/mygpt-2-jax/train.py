import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
from typing import List, Tuple
from config import GPTConfig
from gpt import GPT
from tokenizer import SimpleTokenizer
import numpy as np

class TextDataset:
    def __init__(self, texts: List[str], tokenizer: SimpleTokenizer, max_seq_len: int, batch_size: int = 4):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.tokens = self._tokenize_all()

    def _tokenize_all(self) -> jnp.ndarray:
        all_tokens = []
        for text in self.texts:
            tokens = self.tokenizer.encode(text)
            all_tokens.extend(tokens)
        return jnp.array(all_tokens[:len(all_tokens) - len(all_tokens) % (self.max_seq_len + 1)])

    def __len__(self) -> int:
        return len(self.tokens) // (self.max_seq_len + 1)

    def __iter__(self):
        tokens = np.array(self.tokens)
        num_batches = len(self) // self.batch_size

        for i in range(num_batches):
            batch_start = i * self.batch_size * (self.max_seq_len + 1)
            batch_end = batch_start + self.batch_size * (self.max_seq_len + 1)

            batch_tokens = tokens[batch_start:batch_end]
            batch_tokens = batch_tokens.reshape(self.batch_size, -1)

            input_ids = batch_tokens[:, :self.max_seq_len]
            targets = batch_tokens[:, 1:self.max_seq_len + 1]

            yield jnp.array(input_ids), jnp.array(targets)

def create_dataset(texts: List[str], tokenizer: SimpleTokenizer, max_seq_len: int, batch_size: int = 4) -> TextDataset:
    return TextDataset(texts, tokenizer, max_seq_len, batch_size)

def create_optimizer(learning_rate: float, weight_decay: float, num_steps: int, warmup_steps: int):
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=num_steps - warmup_steps,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay)
    )

    return optimizer

def create_train_state(config: GPTConfig, rng: jnp.ndarray) -> train_state.TrainState:
    model = GPT(config)
    params = model.init(rng, jnp.ones((1, config.max_seq_len), dtype=jnp.int32))['params']

    optimizer = create_optimizer(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        num_steps=config.max_steps,
        warmup_steps=config.warmup_steps
    )

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )

@jax.jit
def train_step(state: train_state.TrainState, batch: Tuple[jnp.ndarray, jnp.ndarray]) -> Tuple[train_state.TrainState, float]:
    input_ids, targets = batch

    def loss_fn(params):
        model = GPT(state.params)
        _, loss = model.apply({'params': params}, input_ids, targets, training=True)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    new_state = state.apply_gradients(grads=grads)

    return new_state, loss

def train(config: GPTConfig, dataset: TextDataset, model_dir: str = '/content/checkpoints'):
    print(f"Starting training: {config.d_model}d, {config.num_heads} heads, {config.num_layers} layers")
    print(f"Dataset size: {len(dataset)} batches")

    rng = jax.random.PRNGKey(0)
    state = create_train_state(config, rng)

    step = 0
    for epoch in range(10):
        for batch in dataset:
            state, loss = train_step(state, batch)
            step += 1

            if step % 100 == 0:
                print(f"Step {step:,} | Loss: {loss:.4f}")

            if step >= config.max_steps:
                break

        if step >= config.max_steps:
            break

    print(f"Training done! Final loss: {loss:.4f}")
    return state.params
