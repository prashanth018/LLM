from functools import partial

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset.preference_dataset import PreferenceDataset, preference_collate_fn
from models.rewards import BradleyTerryRewardModel
from utils.constants import DEVICE, QWEN_INSTRUCT_MODEL
from datasets import load_dataset
from torch.utils.data import DataLoader

if __name__ == "__main__":
    hh_rlhf = load_dataset("Anthropic/hh-rlhf")
    test_dataset = PreferenceDataset(dataset=hh_rlhf["test"])
    tokenizer = AutoTokenizer.from_pretrained(QWEN_INSTRUCT_MODEL)
    dataloader = DataLoader(
        dataset=test_dataset,
        collate_fn=partial(
            preference_collate_fn,
            tokenizer=tokenizer,
            context_length=512,
            device=DEVICE,
        ),
        batch_size=8,
        shuffle=False,
    )
    base_lm = AutoModelForCausalLM.from_pretrained(
        QWEN_INSTRUCT_MODEL, torch_dtype=torch.float32
    )
    model = BradleyTerryRewardModel(base_lm, tokenizer.eos_token_id)
    model.to(device=DEVICE)
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
            if (batch_idx + 1) % 10 == 0:
                print(
                    f"  batch {batch_idx + 1}/{len(dataloader)} | "
                    f"acc so far: {n_correct / n_total:.4f}"
                )

    print(f"\nPreference accuracy: {n_correct}/{n_total} = {n_correct / n_total:.4f}")
    print(f"Avg reward margin (chosen - rejected): {reward_diff_sum / n_total:.4f}")
