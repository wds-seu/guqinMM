import torch.nn as nn

from .densenet import DenseNet
from .positional import PositionalEncoding2D


class Encoder(nn.Module):
    def __init__(
        self,
        d_in,
        d_model,
        growth_rate=24,
        block_config=16,
        dropout=0.1,
        in_channels=1,
    ):
        super().__init__()
        self.d_model = d_model
        self.densenet = DenseNet(growth_rate, block_config, in_channels=in_channels)
        self.pe2d = PositionalEncoding2D()
        self.in_proj = nn.Linear(d_in, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x, mask):
        # x(B,C,H,W)
        # mask(B,H,W)
        x, mask = self.densenet(x, mask)
        pos_emb = self.pe2d(x, self.d_model).flatten(-2).transpose(0, 1)  # (HW,C)
        x = x.flatten(-2).permute(2, 0, 1)  # (HW,B,C)
        mask = mask.flatten(-2)  # (B,HW)
        x = self.layer_norm(self.dropout(self.relu(self.in_proj(x))))
        x.masked_fill_(mask.transpose(0, 1).unsqueeze(-1), 0)

        return x, mask, pos_emb