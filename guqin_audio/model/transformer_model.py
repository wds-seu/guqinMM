import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as tp

# ====== Positional Embedding：正弦位置编码 ======
def create_sin_embedding(positions: torch.Tensor, dim: int, max_period: float = 10000):
    assert dim % 2 == 0
    half_dim = dim // 2
    adim = torch.arange(half_dim, device=positions.device).view(1, 1, -1)
    phase = positions / (max_period ** (adim / (half_dim - 1)))
    return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)


# ====== Streaming Transformer Encoder Layer ======
class StreamingTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def forward(self, x: torch.Tensor, x_past: torch.Tensor, past_context: int):
        if self.norm_first:
            sa_input = self.norm1(x)
            x = x + self._sa_block(sa_input, x_past, past_context)
            x = x + self._ff_block(self.norm2(x))
        else:
            sa_input = x
            x = self.norm1(x + self._sa_block(sa_input, x_past, past_context))
            x = self.norm2(x + self._ff_block(x))
        return x, sa_input

    def _sa_block(self, x: torch.Tensor, x_past: torch.Tensor, past_context: int):
        _, T, _ = x.shape
        _, H, _ = x_past.shape

        if x_past.size(0) != x.size(0):
            min_B = min(x_past.size(0), x.size(0))
            x_past = x_past[:min_B]
            x = x[:min_B]


        queries = x
        keys = torch.cat([x_past, x], dim=1)
        values = keys

        queries_pos = torch.arange(H, T + H, device=x.device).view(-1, 1)
        keys_pos = torch.arange(T + H, device=x.device).view(1, -1)
        delta = queries_pos - keys_pos
        valid_access = (delta >= 0) & (delta <= past_context)

        x = self.self_attn(queries, keys, values, attn_mask=~valid_access, need_weights=False)[0]
        return self.dropout1(x)


# ====== Streaming Transformer Encoder（主干）======
class StreamingTransformerEncoder(nn.Module):
    def __init__(self, dim, hidden_scale=4., num_heads=8, num_layers=6,
                 max_period=10000, past_context=1000, gelu=True, norm_in=True, dropout=0.):
        super().__init__()
        hidden_dim = int(dim * hidden_scale)
        activation = F.gelu if gelu else F.relu

        self.max_period = max_period
        self.past_context = past_context

        self.norm_in = nn.LayerNorm(dim) if norm_in else nn.Identity()
        self.layers = nn.ModuleList([
            StreamingTransformerEncoderLayer(
                dim, num_heads, hidden_dim,
                activation=activation, batch_first=True, dropout=dropout
            ) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor,
                states: tp.Optional[tp.List[torch.Tensor]] = None,
                offset: tp.Union[int, torch.Tensor] = 0):
        B, T, C = x.shape
        if states is None:
            states = [torch.zeros_like(x[:, :1]) for _ in range(len(self.layers))]

        positions = torch.arange(T, device=x.device).view(1, -1, 1) + offset
        pos_emb = create_sin_embedding(positions, C, max_period=self.max_period)

        x = self.norm_in(x) + pos_emb

        new_states = []
        for layer, past in zip(self.layers, states):
            x, new_past = layer(x, past, self.past_context)
            # 🚀 断开 autograd 跟踪！不然内存越堆越大
            new_states.append(torch.cat([past, new_past], dim=1)[:, -self.past_context:, :].detach())

        return x, new_states, offset + T


# ====== 你的最终模型：带有谱文+风格控制 Streaming Transformer ======
class StreamingTransformerModel(nn.Module):
    def __init__(self, vocab_size_score=128, vocab_size_acoustic=1024,
                 style_emb_dim=256, d_model=256, nhead=8, num_layers=6):
        super().__init__()

        self.score_emb = nn.Embedding(vocab_size_score, d_model)
        self.style_emb = nn.Embedding(4, d_model)  # 4种流派 exam, master, silkqin, other
        self.acoustic_emb = nn.Embedding(vocab_size_acoustic, d_model)

        self.streaming_transformer = StreamingTransformerEncoder(
            dim=d_model, hidden_scale=4., num_heads=nhead, num_layers=num_layers,
            max_period=10000, past_context=1000, gelu=True, norm_in=True, dropout=0.1
        )

        self.output_layer = nn.Linear(d_model, vocab_size_acoustic)

    def forward(self, score_tokens, style_labels, targets, states=None, offset=0):
        B, T = targets.shape

        # ----- 构造条件 context -----
        if score_tokens is not None and style_labels is not None:
            score_feat = self.score_emb(score_tokens)  # (B, T_score, d_model)
            style_feat = self.style_emb(style_labels).unsqueeze(1)  # (B, 1, d_model)
            cond_feat = score_feat.mean(dim=1, keepdim=True) + style_feat  # (B, 1, d_model)
        else:
            cond_feat = None

        # ----- 目标 acoustic token embedding -----
        tgt_feat = self.acoustic_emb(targets)  # (B, T, d_model)

        if cond_feat is not None:
            # 拼接 cond + tgt，一起送进去
            tgt_feat = torch.cat([cond_feat, tgt_feat], dim=1)  # (B, 1+T, d_model)

        # 流式 forward！
        out, states, offset = self.streaming_transformer(tgt_feat, states, offset)

        if cond_feat is not None:
            out = out[:, 1:, :]  # 去掉第一个 cond 的输出

        logits = self.output_layer(out)  # (B, T, vocab_size_acoustic)
        if not isinstance(offset, torch.Tensor):
            offset = torch.tensor([offset], device=targets.device)
        else:
            offset = offset.detach().clone()

        return logits, states, offset