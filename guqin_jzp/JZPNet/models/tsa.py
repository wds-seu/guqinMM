import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange, reduce, repeat
from .utils import get_activation_fn


class TSAttention(nn.Module):
    def __init__(
        self,
        d_model,
        d_head,
        num_heads,
        max_num_hops=2,
        activation="relu",
        dropout=0.1,
    ):
        super().__init__()
        self.attention = AdditiveAttention(d_model, d_head, num_heads)
        self.attn_dropout = nn.Dropout(dropout)
        self.fc_o = nn.Linear(num_heads * d_head, d_model)
        self.neighbor_weights = nn.Parameter(
            torch.Tensor(max_num_hops + 1, num_heads))
        if activation:
            self.activation = get_activation_fn(activation)
            self.out_dropout = nn.Dropout(dropout)
        else:
            self.activation = None

        self.num_heads = num_heads
        self.num_hops = max_num_hops

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        adj_mat: Tensor,
        attn_mask: Tensor,
        key_padding_mask: Tensor = None,
        need_weights: bool = False,
    ):
        ns, batch_size = query.shape[:2]
        device = query.device
        adj_mat_list = [
            repeat(torch.eye(ns, ns, device=device),
                   "nq nk -> b nq nk",
                   b=batch_size),
            adj_mat,
        ]
        for _ in range(self.num_hops - 1):
            adj_mat_list.append(
                torch.clamp(torch.bmm(adj_mat_list[-1], adj_mat), 0, 1))

        adj_mat_list = torch.stack(adj_mat_list)
        adj_bias = torch.einsum("jbqk,jh->bhqk", adj_mat_list,
                                self.neighbor_weights)
        adj_mask = rearrange(
            torch.sum(adj_mat_list, dim=0) == 0, "b nq nk -> b 1 nq nk")
        attn, v = self.attention(query, key, value)
        attn = attn + adj_bias + attn_mask
        attn = torch.masked_fill(attn, adj_mask, float("-inf"))

        if key_padding_mask is not None:
            attn = attn.masked_fill(
                rearrange(key_padding_mask, "b ns -> b 1 1 ns"),
                float("-inf"),
            )

        attn = rearrange(attn, "b nh nq nk -> (b nh) nq nk")

        attn = self.attn_dropout(torch.softmax(attn, -1))
        out = rearrange(torch.bmm(attn, v),
                        "(b nh) ns dv -> ns b (nh dv)",
                        nh=self.num_heads)
        if self.activation:
            out = self.out_dropout(self.activation(out))
        out = self.fc_o(out)
        if need_weights:
            attn = reduce(attn,
                          "(b nh) nq nk -> b nq nk",
                          "mean",
                          nh=self.num_heads)
            return out, attn
        else:
            return out, None


class AdditiveAttention(nn.Module):
    def __init__(self, d_model, d_head, num_heads):
        super().__init__()
        self.fc_q = nn.Linear(d_model, num_heads * d_head)
        self.fc_k = nn.Linear(d_model, num_heads * d_head)
        self.scoring_proj = nn.Parameter(torch.Tensor(num_heads, d_head))
        nn.init.xavier_uniform_(self.scoring_proj)
        self.leaky_relu = nn.LeakyReLU()

        self.num_heads = num_heads

    def forward(self, query: Tensor, *args, **kwargs):
        q = rearrange(self.fc_q(query),
                      "ns b (nh dh) -> b ns 1 nh dh",
                      nh=self.num_heads)
        k = rearrange(self.fc_k(query),
                      "ns b (nh dh) -> b 1 ns nh dh",
                      nh=self.num_heads)
        attn = torch.einsum("bqkhe,he->bhqk", self.leaky_relu(q + k),
                            self.scoring_proj)

        return attn, rearrange(k, "b 1 ns nh dh -> (b nh) ns dh")
