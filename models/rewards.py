from torch import arange
from torch.nn import Linear

from models.gpt import TransformerBase


class BradleyTerry(TransformerBase):
    def __init__(self, config, eos_token=50256):
        self.eos_token = eos_token
        super().__init__(config)
        self.reward_head = Linear(in_features=config["dim"], out_features=1, bias=True)

    def forward(self, x):
        mask = x == self.eos_token
        batch_size, batch_context_len = mask.shape
        num_eos = mask.sum(dim=-1)
        # eos pos is len(batch[i])th index
        eos_pos = batch_context_len - num_eos
        r_idx = arange(batch_size)
        x = super().forward(x)
        # x[eos_pos] directly selects the rows. For example, if eos_pos = [2,5],
        # it selects 2nd and 5th row. We rather want 2nd and 5th column in the
        # respective rows to be selected, so we use row index.
        eos_context_vecs = x[r_idx, eos_pos, :]
        return self.reward_head(eos_context_vecs).squeeze(-1)
