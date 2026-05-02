from functools import partial
import os

from dataset.instructionft_dataset import InstructionFineTuningDataset, collate_fn
from load_gpt2 import load_and_map_gpt2

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import glob
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tiktoken
from torch import multinomial, ones, tensor, topk
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
import torch.nn as nn
import torch.nn.functional as F
import time

from utils.constants import CONFIG_EXP_M, CONFIG_GPT2_124M, NUM_EPOCHS
from dataset.gutenberg_dataset import GutenbergDataset
from models.gpt import GPT


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


def gpt2_inference(model=None, init_context="I love", max_tokens=10, device="mps"):
    tokenizer = tiktoken.get_encoding("gpt2")
    stop_token_id = tokenizer.encode(
        "<|endoftext|>", allowed_special={"<|endoftext|>"}
    )[0]
    model.to(device)
    model.eval()
    input_str = init_context
    with torch.no_grad():
        for _ in range(max_tokens):
            input = tensor(
                tokenizer.encode(input_str, allowed_special={"<|endoftext|>"})
            )[-CONFIG_GPT2_124M["context_length"] :]
            input = input.unsqueeze(dim=0).to(device)
            prediction = model(input)
            last_token = prediction[0, -1, :].argmax().item()
            if last_token == stop_token_id:
                break
            input_str = input_str + tokenizer.decode([last_token])
        print(input_str)


def gpt2_inference_with_temperature_and_topk(
    model=None,
    init_context="I love",
    max_tokens=20,
    temperature=0.7,
    top_k=3,
    device="mps",
):
    tokenizer = tiktoken.get_encoding("gpt2")
    stop_token_id = tokenizer.encode(
        "<|endoftext|>", allowed_special={"<|endoftext|>"}
    )[0]
    model.to(device)
    model.eval()

    # input_str = init_context
    input_ids = tokenizer.encode(init_context, allowed_special={"<|endoftext|>"})
    input_tensor = tensor(input_ids).unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(max_tokens):
            input = input_tensor[:, -CONFIG_GPT2_124M["context_length"] :]
            prediction = model(input)
            last_token_logit = prediction[0, -1, :]
            vocab_size = last_token_logit.shape[-1]
            # apply temperature
            last_token_logit = last_token_logit / temperature
            # topk
            topk_indices = topk(last_token_logit, top_k).indices
            mask = ones(vocab_size, dtype=bool).to(device)
            mask[topk_indices] = 0
            last_token_logit = last_token_logit.masked_fill(mask, -float("inf"))
            # softmax and sample
            next_token = multinomial(
                F.softmax(last_token_logit, dim=-1), num_samples=1
            ).item()

            if next_token == stop_token_id:
                break

            next_token_tensor = tensor([[next_token]]).to(device)
            input_tensor = torch.cat((input_tensor, next_token_tensor), dim=1)

        generated_text = tokenizer.decode(input_tensor.squeeze(0).tolist())
        print(generated_text)


def inference_with_temperature_and_topk(
    model=None, init_context="I love", max_tokens=10, temperature=0.7, top_k=3
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
            topk_indices = torch.topk(logits, top_k).indices
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


def calculate_gpt2_loss_for_single_batch(
    model: GPT, pad_token_id=50256, ignore_index=-100, device="mps", batch_size=8
):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = InstructionFineTuningDataset(tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset=dataset,
        collate_fn=partial(
            collate_fn,
            pad_token_id=pad_token_id,
            ignore_index=ignore_index,
            device=device,
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    model.eval()
    cross_entropy_loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
    with torch.no_grad():
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


def freeze_model_for_ift(model):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.transformers[-1].parameters():
        param.requires_grad = True
    for param in model.final_norm.parameters():
        param.requires_grad = True
    for param in model.out_head.parameters():
        param.requires_grad = True


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


def instruction_fine_tune(
    model, device="mps", batch_size=8, num_epochs=NUM_EPOCHS, lr=1e-4
):
    pad_token_id = 50256
    ignore_index = -100
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = InstructionFineTuningDataset(tokenizer=tokenizer)
    # dataset.inputs = dataset.inputs[:50]
    # dataset.targets = dataset.targets[:50]
    n_train = int(len(dataset) * 0.85)
    n_val = len(dataset) - n_train
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(
            collate_fn,
            pad_token_id=pad_token_id,
            ignore_index=ignore_index,
            device=device,
        ),
    )
    validation_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(
            collate_fn,
            pad_token_id=pad_token_id,
            ignore_index=ignore_index,
            device=device,
        ),
    )
    # freeze_model_for_ift(model)
    model.to(device)
    cross_entropy_loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
    optim = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.1
    )
    validation_loss_per_epoch = [
        calculate_avg_loss_for_dataset(model=model, dataloader=validation_dataloader)
    ]
    print(f"Epoch -1, val loss: {validation_loss_per_epoch[-1]}")
    print("Starting instruction fine-tuning loop")
    for epoch in range(num_epochs):
        start = time.time()
        model.train()
        total_batches = len(train_dataloader)
        for batch_idx, (input, target) in enumerate(train_dataloader):
            optim.zero_grad()
            prediction = model(input)
            prediction = prediction.flatten(0, 1)
            target = target.flatten()
            loss = cross_entropy_loss_fn(prediction, target)
            loss.backward()
            optim.step()
            if (batch_idx + 1) % 10 == 0:
                print(f"  {batch_idx + 1} / {total_batches} batches")
        validation_loss_per_epoch.append(
            calculate_avg_loss_for_dataset(
                model=model, dataloader=validation_dataloader
            )
        )
        print(
            f"Epoch {epoch}, val loss: {validation_loss_per_epoch[-1]}, time: {time.time() - start:.2f}s"
        )
    plot_loss(validation_loss_per_epoch, subfolder="/gpt2/ift")
    save_model(model, optim, subfolder="/gpt2/ift")


def save_model(model, optim, subfolder=""):
    weights_dir = f"./weights{subfolder}"
    optim_dir = f"./optim{subfolder}"
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(optim_dir, exist_ok=True)
    existing = glob.glob(f"{weights_dir}/*.pth")
    next_id = (
        max(
            (int(os.path.splitext(os.path.basename(f))[0]) for f in existing),
            default=-1,
        )
        + 1
    )
    torch.save(model.state_dict(), f"{weights_dir}/{next_id:03d}.pth")
    torch.save(optim.state_dict(), f"{optim_dir}/{next_id:03d}.pth")


def load_model(model, optim=None, subfolder=""):
    weights_dir = f"./weights{subfolder}"
    optim_dir = f"./optim{subfolder}"
    existing_weights = glob.glob(f"{weights_dir}/*.pth")
    if not existing_weights:
        raise FileNotFoundError(f"No weights found in {weights_dir}/")
    latest_weights = max(
        existing_weights, key=lambda f: int(os.path.splitext(os.path.basename(f))[0])
    )
    model.load_state_dict(torch.load(latest_weights, weights_only=True))
    if optim is not None:
        existing_optim = glob.glob(f"{optim_dir}/*.pth")
        if not existing_optim:
            raise FileNotFoundError(f"No weights found in {optim_dir}/")
        latest_optim = max(
            existing_optim, key=lambda f: int(os.path.splitext(os.path.basename(f))[0])
        )
        optim.load_state_dict(torch.load(latest_optim, weights_only=True))


def plot_loss(losses, subfolder=""):
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss")
    plots_dir = f"./plots{subfolder}"
    os.makedirs(plots_dir, exist_ok=True)
    existing = glob.glob(f"{plots_dir}/*.png")
    next_id = (
        max(
            (int(os.path.splitext(os.path.basename(f))[0]) for f in existing),
            default=-1,
        )
        + 1
    )
    plt.savefig(f"{plots_dir}/{next_id:03d}.png")
    # plt.show()


def format_inference_prompt(instruction, input_text=""):
    return f"### Instruction: {instruction}; ### Input: {input_text}; ### Output: "


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
    # inference_with_temperature_and_topk()
    # gpt2_inference(model)
    # gpt2_inference_with_temperature_and_topk(model=model)
    # calculate_gpt2_loss_for_single_batch(model=model, device=device)

    # Code block to do IFT
    # device = "mps"
    # model = load_and_map_gpt2(CONFIG_GPT2_124M)
    # instruction_fine_tune(
    #     model=model, lr=1e-5, batch_size=8, num_epochs=2, device=device
    # )

    # Code block to do post IFT inference
    device = "mps"
    model = load_and_map_gpt2(CONFIG_GPT2_124M)
    # freeze_model_for_ift(model)
    optim = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5, weight_decay=0.1
    )
    load_model(model=model, optim=optim, subfolder="/gpt2/ift")
    model.to(device)
    gpt2_inference_with_temperature_and_topk(
        model,
        init_context=format_inference_prompt("List an antonym of complicated"),
        max_tokens=20,
        temperature=0.7,
        top_k=3,
        device=device,
    )
