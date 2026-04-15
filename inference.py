import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import glob
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tiktoken
from torch import tensor
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
import torch.nn as nn
import torch.nn.functional as F

from constants import CONFIG_EXP_M, NUM_EPOCHS
from dataset import GutenbergDataset
from gpt import GPT


def inference(init_context="I love", max_tokens=10):
    tokenizer = tiktoken.get_encoding("gpt2")
    CONFIG_EXP_M["vocab_size"] = tokenizer.n_vocab
    torch.manual_seed(42)
    model = GPT(CONFIG_EXP_M)
    model.eval()
    input_str = init_context
    with torch.no_grad():
        for _ in range(max_tokens):
            # input = tensor(tokenizer.encode(input_str))
            input = tensor(tokenizer.encode(input_str))[
                -CONFIG_EXP_M["context_length"] :
            ]
            input = input.unsqueeze(dim=0)
            # print(input.shape)
            prediction = model(input)
            last_token = prediction[0, -1, :].argmax().item()
            input_str = input_str + tokenizer.decode([last_token])
            print(input_str)


def trained_model_inference(model=None, init_context="I love", max_tokens=10):
    tokenizer = tiktoken.get_encoding("gpt2")
    CONFIG_EXP_M["vocab_size"] = tokenizer.n_vocab
    if model is None:
        model = GPT(config=CONFIG_EXP_M)
        load_model(model)
    model.eval()
    input_str = init_context
    with torch.no_grad():
        for _ in range(max_tokens):
            input = tensor(tokenizer.encode(input_str))[
                -CONFIG_EXP_M["context_length"] :
            ]
            input = input.unsqueeze(dim=0)
            prediction = model(input)
            last_token = prediction[0, -1, :].argmax().item()
            input_str = input_str + tokenizer.decode([last_token])
            print(input_str)


def inference_with_temperature_and_topk(
    model=None, init_context="I love", max_tokens=10, temperature=0.7, topk=3
):
    tokenizer = tiktoken.get_encoding("gpt2")
    CONFIG_EXP_M["vocab_size"] = tokenizer.n_vocab
    if model == None:
        model = GPT(CONFIG_EXP_M)
    model.eval()
    input_str = init_context
    with torch.no_grad():
        input_vec = tensor(
            tokenizer.encode(input_str)[-CONFIG_EXP_M["context_length"] :]
        )
        for _ in range(max_tokens):
            # print("Input size", input_vec.shape)
            input_vec = input_vec.reshape(1, len(input_vec))
            logits = model(input_vec)[0, -1, :]
            # apply temperature
            logits = logits / temperature
            # apply topk
            n_vocab = logits.shape[-1]
            topk_indices = torch.topk(logits, topk).indices
            mask = torch.ones(n_vocab, dtype=bool)
            mask[topk_indices] = 0
            logits = logits.masked_fill(mask, -float("inf"))
            # apply softmax & multinomial
            index = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            # append to input tokens
            input_vec = torch.cat([input_vec, index.reshape(1, 1)], dim=1).squeeze()
            # print tokens
            # print(input_vec)
            # print(input_vec.shape)
            print(tokenizer.decode(input_vec.tolist()))


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


def calculate_avg_loss_for_dataset(model, dataloader):
    cross_entropy_loss_fn = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0
    total_batches = 0
    with torch.no_grad():
        for input, target in dataloader:
            prediction = model(input)
            prediction = prediction.flatten(0, 1)
            target = target.flatten()
            loss = cross_entropy_loss_fn(prediction, target)
            total_loss += loss.item()
            total_batches += 1
    return total_loss / total_batches


def train():
    # print("1. creating tokenizer")
    tokenizer = tiktoken.get_encoding("gpt2")
    # print("2. creating dataset")
    dataset = GutenbergDataset(
        tokenizer=tokenizer,
        context_length=CONFIG_EXP_M["context_length"],
        stride=CONFIG_EXP_M["stride"],
    )
    # print("3. splitting dataset")
    n_train = int(len(dataset) * 0.85)
    n_val = len(dataset) - n_train
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
    # print("4. creating dataloaders")
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=CONFIG_EXP_M["batch_size"], shuffle=True
    )
    validation_dataloader = DataLoader(
        dataset=val_dataset, batch_size=CONFIG_EXP_M["batch_size"], shuffle=True
    )

    # print("5. creating model")
    CONFIG_EXP_M["vocab_size"] = tokenizer.n_vocab
    torch.manual_seed(123)
    model = GPT(config=CONFIG_EXP_M)
    cross_entropy_loss_fn = nn.CrossEntropyLoss()
    optim = AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    validation_loss_per_epoch = []
    print("Starting training loop")
    for epoch in range(NUM_EPOCHS):
        model.train()
        for input, target in train_dataloader:
            # if i == 0: print(f"  epoch {epoch}, batch 0 started")
            optim.zero_grad()
            prediction = model(input)
            prediction = prediction.flatten(0, 1)
            target = target.flatten()
            loss = cross_entropy_loss_fn(prediction, target)
            loss.backward()
            optim.step()
            # if i == 0: print(f"  epoch {epoch}, batch 0 done")
        validation_loss_per_epoch.append(
            calculate_avg_loss_for_dataset(
                model=model, dataloader=validation_dataloader
            )
        )
        print(f"Epoch num {epoch}, loss: {validation_loss_per_epoch[-1]}")
    plot_loss(validation_loss_per_epoch)
    save_model(model=model, optim=optim)
    trained_model_inference(model)


def save_model(model, optim):
    os.makedirs("./weights", exist_ok=True)
    os.makedirs("./optim", exist_ok=True)
    existing = glob.glob("./weights/*.pth")
    next_id = (
        max(
            (int(os.path.splitext(os.path.basename(f))[0]) for f in existing),
            default=-1,
        )
        + 1
    )
    torch.save(model.state_dict(), f"./weights/{next_id:03d}.pth")
    torch.save(optim.state_dict(), f"./optim/{next_id:03d}.pth")


def load_model(model, optim=None):
    existing_weights = glob.glob("./weights/*.pth")
    if not existing_weights:
        raise FileNotFoundError("No weights found in ./weights/")
    latest_weights = max(
        existing_weights, key=lambda f: int(os.path.splitext(os.path.basename(f))[0])
    )
    model.load_state_dict(torch.load(latest_weights, weights_only=True))
    if optim is not None:
        existing_optim = glob.glob("./optim/*.pth")
        if not existing_optim:
            raise FileNotFoundError("No weights found in ./optim/")
        latest_optim = max(
            existing_optim, key=lambda f: int(os.path.splitext(os.path.basename(f))[0])
        )
        optim.load_state_dict(torch.load(latest_optim, weights_only=True))


def plot_loss(losses):
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss")
    os.makedirs("./plots", exist_ok=True)
    existing = glob.glob("./plots/*.png")
    next_id = (
        max(
            (int(os.path.splitext(os.path.basename(f))[0]) for f in existing),
            default=-1,
        )
        + 1
    )
    plt.savefig(f"./plots/{next_id:03d}.png")
    # plt.show()


if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")
    # print(type(tokenizer.encode("I love")))
    # print(tokenizer.encode("I love"))
    # print(tensor(tokenizer.encode("I love")).argmax().item())
    # print(tokenizer.decode(5044))
    # inference()
    # calculate_loss_for_single_batch()
    # train()
    # trained_model_inference()
    inference_with_temperature_and_topk()
