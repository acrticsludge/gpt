"""
GPT-2 Training Script for Google Colab
===================================
Run this in Colab with a GPU runtime for best performance.

USAGE:
  GPU:     python main.py --device cuda
  CPU:     python main.py --device cpu
  Auto:    python main.py (auto-detects)
"""

import os
import sys
import argparse
import torch

sys.argv = [a for a in sys.argv if not a.startswith("-f")]

os.environ["TOKENIZERS_parallelism"] = "false"


def setup_environment(device_name="cuda"):
    print(f"PyTorch version: {torch.__version__}")
    
    if device_name == "cuda":
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return torch.device("cuda")
        else:
            print("WARNING: CUDA requested but not available. Using CPU.")
            return torch.device("cpu")
    else:
        print(f"Using CPU (no GPU detected or requested)")
        return torch.device("cpu")


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
    parser = argparse.ArgumentParser(description="Train GPT-2 on Colab")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "auto"],
                     help="Device: cuda (GPU), cpu, or auto")
    parser.add_argument("--steps", type=int, default=5000, help="Max training steps")
    args = parser.parse_args()

    device = setup_environment(args.device)

    from config import GPTConfig
    from gpt import GPT
    from tokenizer import SimpleTokenizer
    from train import create_dataset, train

    is_gpu = device.type == "cuda"

    config = GPTConfig(
        d_model=256 if is_gpu else 128,
        num_heads=8 if is_gpu else 4,
        num_layers=8 if is_gpu else 4,
        max_seq_len=512 if is_gpu else 256,
        batch_size=4 if is_gpu else 2,
        grad_accum_steps=4 if is_gpu else 2,
        max_steps=args.steps,
        warmup_steps=200,
    )

    tokenizer = SimpleTokenizer()

    print("\nLoading training data...")
    texts = load_training_data(max_samples=500)
    dataset = create_dataset(texts, tokenizer, config.max_seq_len)
    print(f"Dataset: {len(dataset):,} batches")

    print("\nInitializing model...")
    model = GPT(config)
    print(f"Model parameters: {model.get_num_params():,}")

    print(f"\nTraining on {device}...")

    model = train(model, dataset, config, device, save_dir="/content/checkpoints")

    print("\nSaving final model...")
    torch.save(model.state_dict(), "/content/gpt_model.pt")
    print("Done!")


if __name__ == "__main__":
    main()