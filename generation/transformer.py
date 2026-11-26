"""A causal Transformer for predicting piano tokens."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .tokenizer import VOCABULARY


@dataclass(frozen=True)
class TransformerConfig:
    context: int = 512
    width: int = 256
    heads: int = 8
    layers: int = 6
    dropout: float = 0.1

    def __post_init__(self):
        if min(self.context, self.width, self.heads, self.layers) < 1 or self.width % self.heads:
            raise ValueError("Use positive dimensions and a width divisible by the head count")
        if not 0 <= self.dropout < 1:
            raise ValueError("Dropout must be at least zero and less than one")


class CausalBlock(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.attention_norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width * 3)
        self.output = nn.Linear(width, width)
        self.feedforward_norm = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width))
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, steps, width = inputs.shape
        qkv = self.qkv(self.attention_norm(inputs))
        qkv = qkv.reshape(batch, steps, 3, self.heads, width // self.heads)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind()
        attended = F.scaled_dot_product_attention(
            query, key, value, is_causal=True, dropout_p=self.dropout.p if self.training else 0.0
        )
        projected = self.output(attended.transpose(1, 2).reshape(batch, steps, width))
        residual = inputs + self.dropout(projected)
        return residual + self.dropout(self.feedforward(self.feedforward_norm(residual)))


class PianoTransformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.tokens = nn.Embedding(len(VOCABULARY), config.width)
        self.positions = nn.Embedding(config.context, config.width)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.Sequential(
            *(CausalBlock(config.width, config.heads, config.dropout) for _ in range(config.layers))
        )
        self.norm = nn.LayerNorm(config.width)
        # BOS is the initial input and is never a prediction target.
        self.output = nn.Linear(config.width, len(VOCABULARY) - 1)
        self.apply(self.initialize_weights)

    @staticmethod
    def initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        hidden = self.dropout(self.tokens(tokens) + self.positions(positions))
        return self.output(self.norm(self.layers(hidden)))
