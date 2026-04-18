from functools import partial

from torch.utils.data import Dataset, DataLoader
import json
import tiktoken
from torch import tensor, where, stack
from torch.nn.utils.rnn import pad_sequence
import torch


class InstructionFineTuningDataset(Dataset):

    def __init__(self, tokenizer, context_length=1028):
        super().__init__()
        self.tokenizer = tokenizer
        self.context_length = context_length
        with open("data/instruction-data.json") as f:
            self.corpus = json.load(f)
        self.inputs, self.targets = self.create_data()

    def create_data(self):
        inputs = []
        targets = []
        # i = 0
        for elem in self.corpus:
            input_str = f"### Instruction: {elem['instruction']}; ### Input: {elem['input']}; ### Output: {elem['output']}"
            input_tok = self.tokenizer.encode(input_str)
            # input_tok.append("<|endoftext|>")
            if len(input_tok) <= 1:
                continue
            # print("input_tok 1: ", input_tok)
            input_tok = input_tok[-self.context_length :]
            # print("input_tok 2: ", input_tok)
            target_tok = input_tok.copy()
            target_tok.append(50256)
            target_tok = target_tok[1:]
            inputs.append(input_tok)
            targets.append(target_tok)
            # print(inputs, targets)
            # remove this
            # i += 1
            # if i == 2:
            #     break
        return inputs, targets

    def __getitem__(self, index):
        return (tensor(self.inputs[index]), tensor(self.targets[index]))

    def __len__(self):
        return len(self.inputs)


def collate_fn(batch, pad_token_id=50256, ignore_index=-100, device="cpu"):
    input_list = []
    target_list = []
    for i, t in batch:
        input_list.append(i)
        target_list.append(t)
    input_list = pad_sequence(input_list, padding_value=pad_token_id, batch_first=True)
    # pad_sequence(target_list, padding_value=pad_token_id, batch_first=True)
    max_len = max([len(w) for w in target_list])
    print("max_len: ", max_len)

    new_target_list = []
    for target in target_list:
        target = torch.cat(
            [target, tensor([pad_token_id] * (max_len - len(target)), dtype=torch.long)]
        )
        new_target_list.append(target)

    target_list = new_target_list
    new_target_list = []

    for target in target_list:
        mask = target == pad_token_id
        indices = where(mask)[0]
        indices_except_first = indices[1:]
        target[indices_except_first] = ignore_index
        new_target_list.append(target)
    target_list = new_target_list

    input_list = input_list.to(device=device)
    target_list = stack(target_list).to(device=device)
    return input_list, target_list


if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = InstructionFineTuningDataset(tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset=dataset,
        collate_fn=partial(
            collate_fn, pad_token_id=50256, ignore_index=-100, device="cpu"
        ),
        batch_size=8,
        shuffle=True,
    )
    val = next(iter(dataloader))
    # val = dataset.__getitem__(0)
    print(val[0].shape)
    print(val[1].shape)
    # print(val)
