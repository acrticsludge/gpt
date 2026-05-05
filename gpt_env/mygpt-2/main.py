"""
GPT-2 Training Script for Google Colab
===================================
Run this in Colab with a GPU runtime for best performance.
"""

import os
import torch

os.environ["TOKENIZERS_parallelism"] = "false"


def setup_environment():
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: No GPU detected. Training will be slow.")


def load_data():
    from datasets import load_dataset
    print("Loading WikiText-103 dataset...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    texts = [item["text"] for item in ds if item["text"].strip()]
    print(f"Loaded {len(texts):,} documents")
    return texts


def main():
    setup_environment()
    
    from mygpt_2 import GPT, GPTConfig, SimpleTokenizer
    from mygpt_2.train import create_dataset, train

    config = GPTConfig(
        d_model=256,
        num_heads=8,
        num_layers=8,
        max_seq_len=512,
        batch_size=4,
        grad_accum_steps=4,
        max_steps=5000,
        warmup_steps=200,
    )

    tokenizer = SimpleTokenizer()
    
    print("\nLoading training data...")
    texts = load_data()
    dataset = create_dataset(texts[:500], tokenizer, config.max_seq_len)
    print(f"Dataset: {len(dataset):,} batches")

    print("\nInitializing model...")
    model = GPT(config)
    print(f"Model parameters: {model.get_num_params():,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining on {device}...")
    
    model = train(model, dataset, config, device, save_dir="/content/checkpoints")
    
    print("\nSaving final model...")
    torch.save(model.state_dict(), "/content/gpt_model.pt")
    print("Done!")


if __name__ == "__main__":
    main()