import torch.nn as nn
import torch
import einops
import numpy as np
from .utils import tiny_value_of_dtype


class PointerNet(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.fc_c = nn.Linear(d_model, d_model, bias=False)
        self.fc_p = nn.Linear(d_model, d_model, bias=False)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, p_seq, c_seq):

        seq_len, batch_size, d_model = p_seq.shape

        p_seq, c_seq = self.fc_p(p_seq), self.fc_c(c_seq)
        p_seq = einops.rearrange(p_seq, "P N E -> N P E")
        c_seq = einops.rearrange(c_seq, "C N E -> N E C")
        prod = torch.matmul(p_seq, c_seq) / np.sqrt(d_model)  # (N P C)

        mask = prod.new_ones((seq_len, seq_len))  # (P C)
        mask = torch.tril(mask, diagonal=-1)
        mask = einops.repeat(mask, "P C -> N P C", N=batch_size)

        iscores = prod + (mask + tiny_value_of_dtype(prod.dtype)).log()
        return einops.rearrange(iscores, "N P C -> P N C")
