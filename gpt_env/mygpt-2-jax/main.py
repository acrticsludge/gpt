"""
GPT-2 Training Script for Google Colab (JAX/Flax version)
========================================================
Optimized for TPU v5e-1 acceleration.

USAGE:
  python main.py

Note: Use TPU v5e-1 runtime in Colab for best performance.
"""

import os
os.environ['JAX_PLATFORMS'] = 'tpu'

from config import GPTConfig
from tokenizer import SimpleTokenizer
from train import create_dataset, train
import jax

def load_training_data(max_samples: int = None):
    """Load WikiText-103 dataset - clean Wikipedia text."""
    from datasets import load_dataset
    print("Loading dataset: wikitext-103-raw-v1...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    texts = [item["text"] for item in dataset if item["text"].strip()]
    if max_samples:
        texts = texts[:max_samples]
    print(f"Loaded {len(texts):,} documents")
    return texts

def main():
    config = GPTConfig(
        d_model=512,
        num_heads=16,
        num_layers=12,
        max_seq_len=512,
        batch_size=16,
        grad_accum_steps=2,
        max_steps=20000,
        warmup_steps=500,
        learning_rate=1e-3,
        weight_decay=0.01,
    )

    tokenizer = SimpleTokenizer()

    print("\nLoading training data...")
    texts = load_training_data()
    dataset = create_dataset(texts, tokenizer, config.max_seq_len, config.batch_size)
    print(f"Dataset: {len(dataset):,} batches")

    print("\nTraining on JAX/TPU...")
    params = train(config, dataset, save_dir='/content/checkpoints')

    print("\nSaving final model...")
    import pickle
    with open('/content/gpt_model.pkl', 'wb') as f:
        pickle.dump(params, f)
    print("Done!")

if __name__ == "__main__":
    main()
