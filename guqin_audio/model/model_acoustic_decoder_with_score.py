import torch
import torch.nn as nn

class AcousticTransformerDecoderWithScore(nn.Module):
    def __init__(self, vocab_size, style_emb_dim, score_emb_dim, d_model, nhead, num_layers, dropout=0.1):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.style_embedding = nn.Embedding(4, style_emb_dim)
        self.score_embedding = nn.Embedding(5000, score_emb_dim)  # 5000 = 你的谱文vocab大小，可以改

        self.input_proj = nn.Linear(d_model + style_emb_dim + score_emb_dim, d_model)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, token_seq, score_seq, style_labels):
        token_emb = self.token_embedding(token_seq)  # (B, T, d_model)
        score_emb = self.score_embedding(score_seq)  # (B, T_score, score_emb_dim)
        style_emb = self.style_embedding(style_labels)  # (B, style_emb_dim)

        style_emb_expand = style_emb.unsqueeze(1).expand(-1, token_seq.size(1), -1)
        score_emb_expand = score_emb.mean(dim=1).unsqueeze(1).expand(-1, token_seq.size(1), -1)  # mean pooling

        x = torch.cat([token_emb, style_emb_expand, score_emb_expand], dim=-1)
        x = self.input_proj(x)

        x = x.permute(1, 0, 2)
        output = self.transformer_decoder(x, x)
        output = output.permute(1, 0, 2)

        logits = self.output_proj(output)
        return logits

    def load_pretrained_decoder(self, pretrained_dict):
        """ 只加载无监督预训练部分 """
        model_dict = self.state_dict()
        pretrained_keys = [k for k in pretrained_dict.keys() if 'token_embedding' in k or 'transformer_decoder' in k or 'output_proj' in k]
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
