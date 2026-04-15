from torch.nn import Dropout, Linear, Module, Sequential
import tiktoken
from torch.utils.data import DataLoader

from dataset.gutenberg_dataset import GutenbergDataset
from nnets import PositionalEmbedding, TokenEmbedding
from transformer import LayerNorm, Transformer

from torch import arange

from constants import CONFIG_EXP_S


class GPT(Module):
    def __init__(self, config):
        super().__init__()
        # print("  GPT: token embedding")
        self.token_embedding = TokenEmbedding(
            vocab_size=config["vocab_size"], vector_dim=config["dim"]
        )
        # print("  GPT: positional embedding")
        self.positional_embedding = PositionalEmbedding(
            context_length=config["context_length"], vector_dim=config["dim"]
        )
        # print("  GPT: dropout")
        self.dropout = Dropout(config["dropout"])
        # print("  GPT: transformers")
        self.transformers = Sequential(
            *[
                Transformer(
                    context_length=config["context_length"],
                    dim=config["dim"],
                    n_heads=config["n_heads"],
                    dropout=config["dropout"],
                )
                for _ in range(config["n_layers"])
            ]
        )
        # print("  GPT: final norm + out head")
        self.final_norm = LayerNorm(dim=config["dim"])
        self.out_head = Linear(config["dim"], config["vocab_size"], bias=False)
        # print("  GPT: done")

    def forward(self, x):
        b, context_len = x.shape
        x = self.token_embedding(x)
        x_pos = self.positional_embedding(
            arange(context_len, device=x.device),
        )
        input_embedding = x + x_pos
        input_embedding = self.dropout(input_embedding)
        transformer_output = self.transformers(input_embedding)
        normed = self.final_norm(transformer_output)
        logits = self.out_head(normed)
        return logits


if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GutenbergDataset(
        tokenizer=tokenizer,
        context_length=CONFIG_EXP_S["context_length"],
        stride=CONFIG_EXP_S["stride"],
    )
    dataloader = DataLoader(
        dataset=dataset, batch_size=CONFIG_EXP_S["batch_size"], shuffle=True
    )
    CONFIG_EXP_S["vocab_size"] = tokenizer.n_vocab
    model = GPT(CONFIG_EXP_S)
    inputs, targets = next(iter(dataloader))
    prediction = model(inputs)
    print("Shape of Prediction, ", prediction.shape)
