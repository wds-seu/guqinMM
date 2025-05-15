import math
import torch
from torch import nn
from torch import Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, temp=10000.0, normalize=True, alpha=1.0):
        super().__init__()

        self.temp = temp
        self.norm = normalize
        if alpha is None:
            alpha = nn.Parameter(torch.ones(1))
        self.alpha = alpha

    def forward(self, x: Tensor):
        seq_len, _, d_model = x.shape
        pe = torch.zeros(seq_len, d_model, device=x.device)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe * self.alpha


class PositionalEncoding2D(nn.Module):
    def __init__(self, temp=10000.0, normalize=True, scale=None, alpha=1.0):
        super().__init__()
        self.temp = temp
        self.norm = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        if alpha is None:
            alpha = nn.Parameter(torch.ones(1))
        self.alpha = alpha

    def forward(self, x: Tensor, n_channels):
        # x(B,C,H,W)
        max_h, max_w = x.shape[2:]
        half_channels = n_channels // 2
        not_mask = torch.ones(max_h, max_w, device=x.device)
        y_embed = not_mask.cumsum(0, dtype=torch.float32)
        x_embed = not_mask.cumsum(1, dtype=torch.float32)
        if self.norm:
            eps = 1e-6
            y_embed = y_embed / (y_embed[-1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(half_channels, dtype=torch.float, device=x.device)
        inv_feq = 1.0 / (self.temp ** (dim_t / half_channels))

        pos_x = torch.einsum("h w, d -> h w d", x_embed, inv_feq)
        pos_y = torch.einsum("h w, d -> h w d", y_embed, inv_feq)

        pos_x = torch.stack(
            (pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        pos_y = torch.stack(
            (pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        pos = torch.cat((pos_x, pos_y), dim=-1)  # (H,W,C)
        pos = pos.permute(2, 0, 1)
        return pos * self.alpha
