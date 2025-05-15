# models/guqin_from_encodec.py

import torch
import torch.nn as nn
from encodec.model import EncodecModel

class GuqinTransformerFromEncodec(nn.Module):
    def __init__(self, 
                 vocab_size_score=512, 
                 vocab_size_acoustic=1024, 
                 style_emb_dim=256,
                 d_model=256,
                 freeze_transformer=True):
        super().__init__()

        # Load Encodec pretrained transformer (24kHz version)
        base_model = EncodecModel.encodec_model_24khz()
        self.streaming_transformer = base_model.decoder.transformer

        # Optional: freeze encodec transformer
        if freeze_transformer:
            for param in self.streaming_transformer.parameters():
                param.requires_grad = False

        # Add your own control tokens
        self.score_emb = nn.Embedding(vocab_size_score, d_model)
        self.style_emb = nn.Embedding(4, d_model)  # 四种风格标签
        self.pos_emb = nn.Embedding(4096, d_model)  # 位置编码（自己新建的）

        # Output layer to predict acoustic tokens
        self.output_layer = nn.Linear(d_model, vocab_size_acoustic)

    def forward(self, score_tokens, style_labels, targets, states=None, offset=0):
        """
        score_tokens: (B, T_score) or None
        style_labels: (B,) or None
        targets: (B, T_target)
        states: list of past cached states for each transformer layer
        offset: int
        """

        B, T = targets.shape

        # --- 位置编码 ---
        pos_ids = torch.arange(T, device=targets.device).unsqueeze(0).expand(B, T)

        # --- 条件编码 ---
        if score_tokens is not None and style_labels is not None:
            score_feat = self.score_emb(score_tokens).mean(dim=1)  # (B, d_model)
            style_feat = self.style_emb(style_labels)              # (B, d_model)
            cond_feat = score_feat + style_feat                    # (B, d_model)
            cond_feat = cond_feat.unsqueeze(1)                     # (B, 1, d_model)
        else:
            cond_feat = None

        # --- 目标声学token嵌入 ---
        tgt_feat = self.pos_emb(pos_ids)  # (B, T, d_model)

        if cond_feat is not None:
            tgt_feat = tgt_feat + cond_feat

        # --- 传给Streaming Transformer ---
        out, states, offset = self.streaming_transformer(tgt_feat, states=states, offset=offset)

        # --- 输出声学token logits ---
        logits = self.output_layer(out)  # (B, T, vocab_size_acoustic)

        return logits, states, offset
