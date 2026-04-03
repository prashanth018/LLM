import torch.nn as nn
from torch.nn import Embedding, Module
from torch import arange


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, vector_dim):
        super().__init__()
        self.embedding = Embedding(vocab_size, vector_dim)

    def forward(self, batch):
        return self.embedding(batch)


class PositionalEmbedding(nn.Module):
    def __init__(self, context_length, vector_dim):
        super().__init__()
        self.embedding = Embedding(context_length, vector_dim)

    def forward(self):
        positions = arange(self.embedding.num_embeddings)
        return self.embedding(positions)
