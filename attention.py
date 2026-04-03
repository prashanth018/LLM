from torch.nn import Module, Linear, ModuleList
import torch.nn.functional as F
from torch import triu, ones, cat, bool


class SelfAttention(Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.W_Q = Linear(dim_in, dim_out, bias=False)
        self.W_K = Linear(dim_in, dim_out, bias=False)
        self.W_V = Linear(dim_in, dim_out, bias=False)

    def forward(self, batch):
        Q = self.W_Q(batch)
        K = self.W_K(batch)
        V = self.W_V(batch)

        dim_key = K.shape[-1]
        attention_scores = Q @ K.transpose(-2, -1)
        scaled_attention_scores = attention_scores / dim_key**0.5
        attention_weights = F.softmax(scaled_attention_scores, dim=-1)
        return attention_weights @ V


class CausalAttention(Module):
    def __init__(self, context_length, dim_in, dim_out):
        super().__init__()
        self.W_Q = Linear(dim_in, dim_out, bias=False)
        self.W_K = Linear(dim_in, dim_out, bias=False)
        self.W_V = Linear(dim_in, dim_out, bias=False)
        # boolean mask
        self.causal_mask = triu(
            ones(context_length, context_length, dtype=bool), diagonal=1
        )

    def forward(self, batch):
        Q = self.W_Q(batch)
        K = self.W_K(batch)
        V = self.W_V(batch)

        d_keys = K.shape[-1]

        # attention scores
        attention_scores = Q @ K.transpose(-1, -2)
        # causal mask to populate upper right triangle to -inf
        # self.causal_mask acts like a filter, float("-inf") populated in all the elements with 1 in self.causal_mask
        attention_weights = attention_scores.masked_fill(
            self.causal_mask, float("-inf")
        )
        # scale it to normalize variance
        scaled_attention_weights = attention_weights / d_keys**0.5
        # softmax across projected dim (dim_out)
        # elements corresponding to -inf turn to 0
        normed_attention_weights = F.softmax(scaled_attention_weights, dim=-1)
        # multiply with value to get context rich vector
        return normed_attention_weights @ V


class MultiHeadAttention(Module):
    def __init__(self, num_heads, context_length, dim_in, dim_out):
        super().__init__()
        self.num_heads = num_heads
        self.attention_heads = ModuleList([])
        for i in range(num_heads):
            self.attention_heads.append(
                CausalAttention(
                    context_length=context_length, dim_in=dim_in, dim_out=dim_out
                )
            )

    def forward(self, batch):
        self.mha_out = []
        for i in range(self.num_heads):
            self.mha_out.append(self.attention_heads[i](batch))
        return cat(self.mha_out, dim=-1)
