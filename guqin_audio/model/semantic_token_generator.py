import torch
import torch.nn as nn

class SemanticTokenGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8),
            num_layers=6
        )
        self.output_proj = nn.Linear(embed_dim, vocab_size)
        self.max_seq_len = max_seq_len

    def forward(self, score_tokens, style_emb):
        x = self.embedding(score_tokens)
        style_emb = style_emb.unsqueeze(1).expand(-1, x.size(1), -1)
        x = torch.cat([x, style_emb], dim=-1)
        x = self.transformer(x)
        logits = self.output_proj(x)
        return logits
