from torch import arange
from torch.nn import Linear, Module
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

from utils.constants import QWEN_INSTRUCT_MODEL


class BradleyTerryRewardModel(Module):
    def __init__(self, base_lm, eos_token_id):
        super().__init__()
        self.base_lm = base_lm
        self.reward_head = Linear(
            in_features=self.base_lm.config.hidden_size, out_features=1, bias=True
        )
        # eos_token populated from tiktoken
        self.eos_token_id = eos_token_id
        # pad_token_id is not populated by default
        self.base_lm.config.pad_token_id = eos_token_id

    def forward(self, x):
        mask = x == self.eos_token_id
        batch_size, batch_context_len = mask.shape
        num_eos = mask.sum(dim=-1)
        # eos pos is len(batch[i])th index
        eos_pos = batch_context_len - num_eos
        # call the inner transformer directly (skip the LM head + ~5GB logits tensor)
        outputs = self.base_lm.model(x)
        # last hidden state straight from the base model — no need to retain all layers
        x = outputs.last_hidden_state
        # x[eos_pos] directly selects the rows. For example, if eos_pos = [2,5],
        # it selects 2nd and 5th row. We rather want 2nd and 5th column in the
        # respective rows to be selected, so we use row index.
        r_idx = arange(batch_size, device=x.device)
        eos_context_vecs = x[r_idx, eos_pos, :]
        return self.reward_head(eos_context_vecs).squeeze(-1)


if __name__ == "__main__":
    base_lm = AutoModelForCausalLM.from_pretrained(QWEN_INSTRUCT_MODEL)
    # config = AutoConfig.from_pretrained(QWEN_INSTRUCT_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(QWEN_INSTRUCT_MODEL)
    # tokenizer.pad_token = tokenizer.eos_token
    BradleyTerryRewardModel(base_lm, tokenizer.eos_token_id)
    # print(config)
