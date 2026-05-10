import os
from functools import partial

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset.preference_dataset import PreferenceDataset, preference_collate_fn
from models.rewards import BradleyTerryRewardModel
from utils.constants import DEVICE, QWEN_INSTRUCT_MODEL, resolve_save_path
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F


def load_checkpoint(model, optimizer, save_path, device=DEVICE):
    """Resume training from a checkpoint if one exists at save_path.

    Returns (start_epoch, global_step) — both 0 if no checkpoint found.
    Safe to call unconditionally: missing/empty path is a no-op.
    """
    if not save_path or not os.path.exists(save_path):
        print(f"no checkpoint at {save_path!r} — starting from scratch")
        return 0, 0
    ckpt = torch.load(save_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_epoch = ckpt.get("epoch", 0)
    global_step = ckpt.get("global_step", 0)
    print(f"resumed from {save_path} | epoch={start_epoch} | step={global_step}")
    return start_epoch, global_step


def plot_losses(losses, save_path, title="Bradley-Terry training loss"):
    """Plot per-step training losses and save to disk."""
    if not losses:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(losses, linewidth=1)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"saved loss plot to {save_path}")


def inference(model, dataloader, log_every=10):
    """Evaluate a Bradley-Terry reward model on a preference dataloader.

    Returns a dict with preference accuracy and average reward margin.
    """
    model.eval()

    n_correct = 0
    n_total = 0
    reward_diff_sum = 0.0
    with torch.no_grad():
        for batch_idx, (chosen, _, rejected, _) in enumerate(dataloader):
            chosen_reward = model(chosen)
            rejected_reward = model(rejected)
            n_correct += (chosen_reward > rejected_reward).sum().item()
            n_total += chosen_reward.size(0)
            reward_diff_sum += (chosen_reward - rejected_reward).sum().item()
            if log_every and (batch_idx + 1) % log_every == 0:
                print(
                    f"  batch {batch_idx + 1}/{len(dataloader)} | "
                    f"acc so far: {n_correct / n_total:.4f}"
                )

    accuracy = n_correct / n_total if n_total else 0.0
    avg_margin = reward_diff_sum / n_total if n_total else 0.0
    print(f"\nPreference accuracy: {n_correct}/{n_total} = {accuracy:.4f}")
    print(f"Avg reward margin (chosen - rejected): {avg_margin:.4f}")
    return {
        "n_correct": n_correct,
        "n_total": n_total,
        "accuracy": accuracy,
        "avg_margin": avg_margin,
    }


def train(
    model,
    train_dataloader,
    eval_dataloader=None,
    n_epochs=1,
    lr=0.5 * 1e-5,
    log_every=10,
    eval_every_steps=None,
    save_path=None,
    plot_path=None,
    resume=True,
):
    """Train a Bradley-Terry reward model on a preference dataloader."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    start_epoch, global_step = (0, 0)
    if resume:
        start_epoch, global_step = load_checkpoint(model, optimizer, save_path)

    model.train()
    losses = []
    for epoch in range(start_epoch, n_epochs):
        for batch_idx, (chosen, _, rejected, _) in enumerate(train_dataloader):
            optimizer.zero_grad()
            chosen_reward = model(chosen)
            rejected_reward = model(rejected)
            loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()
            loss.backward()
            optimizer.step()

            global_step += 1
            losses.append(loss.item())
            if log_every and (batch_idx + 1) % log_every == 0:
                print(
                    f"epoch {epoch + 1}/{n_epochs} | "
                    f"batch {batch_idx + 1}/{len(train_dataloader)} | "
                    f"loss: {loss.item():.4f}"
                )

            if (
                eval_dataloader is not None
                and eval_every_steps
                and global_step % eval_every_steps == 0
            ):
                inference(model, eval_dataloader)
                model.train()

        if save_path:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                save_path,
            )
            print(f"saved checkpoint to {save_path}")

        if eval_dataloader is not None:
            inference(model, eval_dataloader)
            model.train()

    if plot_path:
        plot_losses(losses, plot_path)
    return losses


if __name__ == "__main__":
    hh_rlhf = load_dataset("Anthropic/hh-rlhf")
    train_dataset = PreferenceDataset(dataset=hh_rlhf["train"], pct=0.0005)
    test_dataset = PreferenceDataset(dataset=hh_rlhf["test"], pct=0.01)
    tokenizer = AutoTokenizer.from_pretrained(QWEN_INSTRUCT_MODEL)
    collate = partial(
        preference_collate_fn,
        tokenizer=tokenizer,
        context_length=512,
        device=DEVICE,
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        collate_fn=collate,
        batch_size=4,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        collate_fn=collate,
        batch_size=8,
        shuffle=False,
    )
    base_lm = AutoModelForCausalLM.from_pretrained(
        QWEN_INSTRUCT_MODEL, torch_dtype=torch.float32
    )
    model = BradleyTerryRewardModel(base_lm, tokenizer.eos_token_id)
    model.to(device=DEVICE)
    model.base_lm.gradient_checkpointing_enable()

    train(
        model,
        train_dataloader,
        eval_dataloader=test_dataloader,
        n_epochs=1,
        lr=0.5 * 1e-5,
        log_every=2,
        save_path=resolve_save_path("reward_model.pt"),
        plot_path=resolve_save_path("loss.png"),
        resume=True,
    )
