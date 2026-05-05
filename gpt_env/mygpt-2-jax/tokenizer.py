import tiktoken
from dataclasses import dataclass
from typing import List

@dataclass
class TokenizerConfig:
    encoding_name: str = "gpt2"

class SimpleTokenizer:
    def __init__(self, encoding_name: str = "gpt2"):
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self.encoding.n_vocab

    def encode(self, text: str) -> List[int]:
        return self.encoding.encode(text)

    def decode(self, tokens: List[int]) -> str:
        return self.encoding.decode(tokens)

    def __call__(self, text: str) -> List[int]:
        return self.encode(text)
