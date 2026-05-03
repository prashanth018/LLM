from functools import partial

from datasets import load_dataset
from torch import tensor
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from utils.constants import QWEN_INSTRUCT_MODEL

TEMP = 400


class PreferenceDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset.select(range(TEMP))
        self.chosen, self.rejected = self.create_data()

    def create_data(self):
        chosen, rejected = [], []
        for data in self.dataset:
            chosen.append(data["chosen"])
            rejected.append(data["rejected"])
        return (chosen, rejected)

    def __getitem__(self, index):
        return (self.chosen[index], self.rejected[index])

    def __len__(self):
        return len(self.chosen)


def preference_collate_fn(batch, tokenizer, context_length=512, device="mps"):
    chosen_tensor, rejected_tensor = [], []
    eos_token = tokenizer.eos_token
    for c, r in batch:
        # prompts with size more than context_length
        if (
            len(tokenizer.encode(c + eos_token)) > context_length - 2
            or len(tokenizer.encode(c + eos_token)) > context_length - 2
        ):
            continue
        # append eos to prompts
        chosen_tensor.append(c + eos_token)
        rejected_tensor.append(r + eos_token)
    # Example output like: {'input_ids': [[14990, 151645, 151643, 151643], [14990, 1879, 151645, 151643]],
    # 'attention_mask': [[1, 1, 0, 0], [1, 1, 1, 0]]}, where 151645 is eos, 151643 is pad
    chosen = tokenizer(chosen_tensor, padding=True, return_tensors="pt")
    chosen_tensor, chosen_mask = chosen["input_ids"], chosen["attention_mask"]
    rejected = tokenizer(rejected_tensor, padding=True, return_tensors="pt")
    rejected_tensor, rejected_mask = rejected["input_ids"], rejected["attention_mask"]
    return (
        chosen_tensor.to(device=device),
        chosen_mask.to(device=device),
        rejected_tensor.to(device=device),
        rejected_mask.to(device=device),
    )


if __name__ == "__main__":
    hh_rlhf = load_dataset("Anthropic/hh-rlhf")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_INSTRUCT_MODEL)
    train_dataset = PreferenceDataset(dataset=hh_rlhf["train"])
    dataloader = DataLoader(
        dataset=train_dataset,
        collate_fn=partial(
            preference_collate_fn, tokenizer=tokenizer, context_length=512, device="mps"
        ),
        batch_size=8,
        shuffle=True,
    )
    val = next(iter(dataloader))
    # print(len(hh_rlhf["test"]))
    # print(hh_rlhf["test"][0])
    print(val[0].shape)
    print(val[1].shape)
    print(val[2].shape)
    print(val[3].shape)
