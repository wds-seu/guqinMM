import torch
import torch.nn as nn

class AcousticTransformerDecoder(nn.Module):
    def __init__(self, vocab_size, style_emb_dim, d_model, nhead, num_layers, dropout=0.1):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.style_embedding = nn.Embedding(4, style_emb_dim)  # 4个流派

        self.input_proj = nn.Linear(d_model + style_emb_dim, d_model)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, token_seq, style_labels):
        """
        token_seq: (B, T)
        style_labels: (B,)
        """
        token_emb = self.token_embedding(token_seq)  # (B, T, d_model)
        style_emb = self.style_embedding(style_labels)  # (B, style_emb_dim)
        style_emb = style_emb.unsqueeze(1).expand(-1, token_seq.size(1), -1)  # (B, T, style_emb_dim)

        x = torch.cat([token_emb, style_emb], dim=-1)  # (B, T, d_model + style_emb_dim)
        x = self.input_proj(x)  # (B, T, d_model)

        # Transformer expects (T, B, d_model)
        x = x.permute(1, 0, 2)

        output = self.transformer_decoder(x, x)  # (T, B, d_model)
        output = output.permute(1, 0, 2)  # (B, T, d_model)

        logits = self.output_proj(output)  # (B, T, vocab_size)
        return logits
