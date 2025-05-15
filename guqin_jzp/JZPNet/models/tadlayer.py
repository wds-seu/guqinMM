import torch.nn as nn
from torch import Tensor
from .tsa import TSAttention
from typing import Optional


class TADLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        max_num_hops=2,
        tsa_activation=None,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model,
                                                    nhead,
                                                    dropout=dropout)
        self.gat = TSAttention(
            d_model,
            dim_feedforward // nhead,
            nhead,
            max_num_hops=max_num_hops,
            activation=tsa_activation,
            dropout=dropout,
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        tgt: Tensor,
        mem: Tensor,
        tgt_pos: Tensor,
        mem_pos: Tensor,
        tgt_adj_mat: Tensor,
        tgt_mask: Optional[Tensor] = None,
        mem_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        mem_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        q = tgt + tgt_pos
        tgt2 = self.self_attn(q,
                              q,
                              tgt,
                              attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        k = mem + mem_pos
        tgt2, attn_weights = self.multihead_attn(
            tgt,
            k,
            mem,
            attn_mask=mem_mask,
            key_padding_mask=mem_key_padding_mask,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.gat(tgt, tgt, tgt, tgt_adj_mat, tgt_mask,
                        tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn_weights
