import math
import torch
import torch.nn as nn

from .positional import PositionalEncoding


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def get_weight(self):
        return self.embedding.weight

    def forward(self, tokens: torch.Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class TreeEmbeddingBlock(nn.Module):
    def __init__(self, d_model, node_vocab_size, edge_vocab_size,
                 node_emb_size, edge_emb_size) -> None:
        super().__init__()
        self.node_emb = TokenEmbedding(node_vocab_size, node_emb_size)
        self.edge_emb = TokenEmbedding(edge_vocab_size, edge_emb_size)
        self.fc_ne = nn.Linear((node_emb_size + edge_emb_size), d_model)
        self.pe = PositionalEncoding()

    def _reset_parameters(self):
        for child in self.children():
            if isinstance(child, nn.Linear):
                nn.init.xavier_uniform_(child.weight)
                nn.init.zeros_(child.bias)

    def forward(self, v_list, e_list):
        v_emb = self.node_emb(v_list)
        e_emb = self.edge_emb(e_list)
        h_emb = self.fc_ne(torch.cat([v_emb, e_emb], dim=-1))
        h_pos_emb = self.pe(h_emb)

        return h_emb, h_pos_emb
