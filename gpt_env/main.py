# ===== WHAT: Standard Python libraries =====
import math              # WHY: sqrt(), sin(), cos() for positional encoding math
import time              # WHY: measure training speed (tokens per second)
import os                # WHY: create directories, save/load model checkpoint files
from dataclasses import dataclass  # WHY: clean config class — no messy dictionaries

# ===== WHAT: NumPy — the CPU array library =====
import numpy as np       # WHY: fast numerical operations on CPU arrays
                         #      (mostly used for quick data checks, not heavy lifting)

# ===== WHAT: PyTorch — the neural network framework =====
import torch             # WHY: core library — tensors, GPU support, autograd
import torch.nn as nn               # WHY: neural network building blocks:
                                     #      Linear (dense layers), Embedding (lookup tables),
                                     #      Dropout (regularization), ModuleList (stacking layers)
import torch.nn.functional as F     # WHY: stateless functions used inside forward():
                                     #      softmax (convert to probabilities),
                                     #      cross_entropy (measure prediction error),
                                     #      silu (SwiGLU activation function)
from torch.utils.data import Dataset, DataLoader  # WHY: efficient data pipeline
#                                  Dataset = define how to load one sample
#                                  DataLoader = batch them, shuffle, prefetch

# ===== WHAT: tiktoken — OpenAI's fast BPE tokenizer =====
import tiktoken          # WHY: same Byte Pair Encoding tokenizer as GPT-3.5/GPT-4
                         #      Written in Rust, ~100x faster than pure Python tokenizers
                         #      Handles 50K+ vocabulary efficiently

# ===== WHAT: HuggingFace datasets — download training text =====
from datasets import load_dataset    # WHY: one line to download WikiText-103
                                     #      Handles caching (only downloads once),
                                     #      streaming (for datasets too big for disk),
                                     #      and format conversion automatically

# ===== WHAT: matplotlib — plot loss curves =====
import matplotlib.pyplot as plt      # WHY: visualize training progress
                                     #      Is the loss going down? Is it plateauing?
                                     #      A picture is worth 1,000 log lines

# ===== WHAT: Quick verification =====
# WHY: Always test your environment before writing 500 lines of code.
#      A missing import now saves hours of debugging later.
print("All imports ready!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:             {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory:      {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")