from torch.nn import Module, Parameter, Sequential, Linear, Dropout
from torch import ones, zeros, tanh
import math

from models.attention import MultiHeadAttentionEfficient


class GELU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        return (
            0.5
            * batch
            * (1 + tanh(math.sqrt(2.0 / math.pi) * (batch + 0.044715 * batch**3)))
        )


class LayerNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = Parameter(ones(dim))
        self.shift = Parameter(zeros(dim))
        self.eps = 1e-5

    def forward(self, batch):
        mean = batch.mean(dim=-1, keepdim=True)
        # turn off bessel's correction
        var = batch.var(dim=-1, keepdim=True, unbiased=False)
        layer_norm = (batch - mean) / (var + self.eps) ** 0.5
        return self.scale * layer_norm + self.shift


class FeedForwardNetwork(Module):
    def __init__(self, dim):
        super().__init__()
        self.ffn = Sequential(Linear(dim, 4 * dim), GELU(), Linear(4 * dim, dim))

    def forward(self, batch):
        return self.ffn(batch)


class Transformer(Module):
    def __init__(self, context_length, dim, n_heads, dropout):
        super().__init__()
        self.norm1 = LayerNorm(dim=dim)
        self.mha = MultiHeadAttentionEfficient(
            context_length=context_length,
            dim_in=dim,
            dim_out=dim,
            num_heads=n_heads,
            dropout=dropout,
        )
        self.dropout = Dropout(dropout)
        self.norm2 = LayerNorm(dim=dim)
        self.ffn = FeedForwardNetwork(dim=dim)

    def forward(self, batch):
        # mha block
        residual_path_1 = batch
        # layer norm for mha block
        x = self.norm1(batch)
        x = self.mha(x)
        x = self.dropout(x)
        x = x + residual_path_1
        # ffn block
        residual_path_2 = x
        # layer norm for ffn block
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        return x + residual_path_2
