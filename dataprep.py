import tiktoken
import torch.nn as nn
from torch.nn import Embedding, Module
from torch.utils.data import DataLoader


from dataset import GutenbergDataset

CONTEXT_LENGTH = 4
STRIDE = 4
VECTOR_DIM = 256
BATCH_SIZE = 32


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, vector_dim):
        super().__init__()
        self.embedding = Embedding(vocab_size, vector_dim)

    def forward(self, batch):
        return self.embedding(batch)


class PositionEmbedding(nn.Module):
    def __init__(self, context_length, vector_dim):
        super().__init__()
        self.embedding = Embedding(context_length, vector_dim)

    def forward(self, batch):
        return self.embedding(batch)


if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GutenbergDataset(
        tokenizer=tokenizer, context_length=CONTEXT_LENGTH, stride=STRIDE
    )
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    inputs, targets = next(iter(dataloader))
    print("Inputs = ", inputs)
    print("Target = ", targets)
