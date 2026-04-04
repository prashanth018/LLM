from torch.nn import Module, Linear, ModuleList, Dropout
import torch.nn.functional as F
from torch import triu, ones, cat


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


class MultiHeadAttentionEfficient(Module):
    def __init__(self, context_length, dim_in, dim_out, num_heads, dropout):
        super().__init__()
        qkv_bias = False
        self.num_heads = num_heads
        self.dim_out = dim_out
        self.W_Q = Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_K = Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_V = Linear(dim_in, dim_out, bias=qkv_bias)
        self.bool_mask = triu(
            ones(context_length, context_length, dtype=bool), diagonal=1
        )
        self.dropout = Dropout(dropout)

    def forward(self, batch):
        # batch = (b, context_length, dim_in)
        # W_Q = (dim_in, dim_out)
        # Q = (b, context_length, dim_out)
        Q = self.W_Q(batch)
        # W_K = (dim_in, dim_out)
        # K = (b, context_length, dim_out)
        K = self.W_K(batch)
        # W_V = (dim_in, dim_out)
        # V = (b, context_length, dim_out)
        V = self.W_V(batch)
        batch_size = batch.shape[0]
        context_length = batch.shape[1]
        # calculate output dim of each attention head
        head_dim = self.dim_out // self.num_heads
        Q_rearranged = Q.view(batch_size, context_length, self.num_heads, head_dim)
        # Q_rearranged = (b, context_length, num_heads, dim_out/num_heads)
        Q_rearranged = Q_rearranged.transpose(-2, -3)
        # Q_rearranged = (b, num_heads, context_length, dim_out/num_heads)

        # K_rearraged = (b, context_length, num_heads, head_dim)
        K_rearraged = K.view(batch_size, context_length, self.num_heads, head_dim)
        # K_rearraged = (b, num_heads, context_length, head_dim)
        K_rearraged = K_rearraged.transpose(-2, -3)
        # Taking transpose of Key Matrices. K_rearraged = (b, num_heads, head_dim, context_length)
        K_rearraged = K_rearraged.transpose(-2, -1)

        # attention_scores = (b, num_heads, context_length, context_length)
        attention_scores = Q_rearranged @ K_rearraged
        attention_scores = attention_scores.masked_fill(self.bool_mask, -float("inf"))
        attention_scores = attention_scores / head_dim**0.5
        # attention_weights = (b, num_heads, context_length, context_length)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # dropout regularization
        attention_weights = self.dropout(attention_weights)

        # V_rearraged = (b, context_length, num_heads, head_dim)
        V_rearranged = V.view(batch_size, context_length, self.num_heads, head_dim)
        # V_rearraged = (b, num_heads, context_length, head_dim)
        V_rearranged = V_rearranged.transpose(-2, -3)

        # context_vector = (b, num_heads, context_length, head_dim)
        context_vector = attention_weights @ V_rearranged
        # context_vector = (b, context_length, num_heads, head_dim)
        context_vector = context_vector.transpose(-2, -3)
        # context_vector = (b, context_length, num_heads * head_dim)
        return context_vector.reshape(batch_size, context_length, self.dim_out)


# Note:
# View Method: Takes an input, ensures that it is contiguous, generates an output
# that is contiguous (manipulates strides to generate output, doesn't change memory).
# Transpose Method: Takes an input, changes strides, thereby making the output non-contiguous (doesn't change memory).
# Problem: View method is picky about the input, wants it to be contiguous while transpose doesn't care. Therefore we
# should use reshape instead of view if we are using view after transpose operation
