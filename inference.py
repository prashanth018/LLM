import tiktoken
from torch import tensor
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from constants import CONFIG_EXP_M
from dataset import GutenbergDataset
from gpt import GPT


def inference(init_context="I love", max_tokens=10):
    tokenizer = tiktoken.get_encoding("gpt2")
    CONFIG_EXP_M["vocab_size"] = tokenizer.n_vocab
    torch.manual_seed(42)
    model = GPT(CONFIG_EXP_M)
    model.eval()
    input_str = init_context
    for i in range(max_tokens):
        input = tensor(tokenizer.encode(input_str))
        input = input.unsqueeze(dim=0)
        # print(input.shape)
        prediction = model(input)
        last_token = prediction[0, -1, :].argmax().item()
        input_str = input_str + tokenizer.decode([last_token])
        print(input_str)


def calculate_loss_for_single_batch():
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GutenbergDataset(
        tokenizer=tokenizer,
        context_length=CONFIG_EXP_M["context_length"],
        stride=CONFIG_EXP_M["stride"],
    )
    dataloader = DataLoader(
        dataset=dataset, batch_size=CONFIG_EXP_M["batch_size"], shuffle=True
    )
    CONFIG_EXP_M["vocab_size"] = tokenizer.n_vocab
    cross_entropy_loss_fn = nn.CrossEntropyLoss()
    torch.manual_seed(42)
    model = GPT(CONFIG_EXP_M)
    model.eval()
    for input, target in dataloader:
        print("Shape of the Input: ", input.shape)
        # print(input)
        prediction = model(input)
        # print("Shape of the Prediction: ", prediction.shape)
        # flatten batch and token column; resulting shape = (b * context_size, vocab_size)
        prediction = prediction.flatten(0, 1)
        # flatten batch and token column; resulting shape = (b * context_size,)
        target = target.flatten()
        print("Shape of the Prediction: ", prediction.shape)
        print("Shape of the Target: ", target.shape)
        loss = cross_entropy_loss_fn(prediction, target)
        print(loss.item())
        break


if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")
    # print(type(tokenizer.encode("I love")))
    # print(tokenizer.encode("I love"))
    # print(tensor(tokenizer.encode("I love")).argmax().item())
    # print(tokenizer.decode(5044))
    # inference()
    calculate_loss_for_single_batch()
