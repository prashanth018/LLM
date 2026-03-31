from torch.nn import Module, Linear
import torch.nn.functional as F


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
