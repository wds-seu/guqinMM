import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F

from .tadlayer import TADLayer
from .utils import generate_square_subsequent_mask
from .pointer import PointerNet
from .embedding import TreeEmbeddingBlock


class BeamNode:
    def __init__(self, prob, v_list, e_list, p_list):
        self.prob = prob
        self.v_list = v_list
        self.e_list = e_list
        self.p_list = p_list

    def __len__(self):
        return len(self.v_list)

    def score(self, length_norm=0.7):
        return self.prob / (len(self.v_list)**length_norm)



class TADNet(nn.Module):
    def __init__(
        self,
        d_model,
        node_vocab_size,
        edge_vocab_size,
        node_emb_size,
        edge_emb_size,
        nc=3,
        np=3,
        n_head=8,
        d_ff=1024,
        dropout=0.3,
        activation="relu",
        max_num_hops=2
    ):
        super().__init__()
        self.tree_emb_blk = TreeEmbeddingBlock(
            d_model,
            node_vocab_size,
            edge_vocab_size,
            node_emb_size,
            edge_emb_size,
        )
        self.chld_pred_blk = nn.ModuleList([
            TADLayer(
                d_model,
                n_head,
                d_ff,
                dropout,
                max_num_hops,
                activation,
            ) for _ in range(nc)
        ])
        self.prnt_pred_blk = nn.ModuleList([
            TADLayer(
                d_model,
                n_head,
                d_ff,
                dropout,
                max_num_hops,
                activation,
            ) for _ in range(np)
        ])
        self.pos_pred_blk = PointerNet(d_model)
        self.fc_chld = nn.Linear(d_model, node_vocab_size)
        self.fc_prnt = nn.Linear(d_model, node_vocab_size)
        self.fc_edge = nn.Linear(d_model * 2, edge_vocab_size)

        self._reset_parameters()

    def _reset_parameters(self):
        for child in self.children():
            if isinstance(child, nn.Linear):
                nn.init.xavier_uniform_(child.weight)
                nn.init.zeros_(child.bias)

    def forward(self, x, x_mask, x_pos_emb, v_list, e_list, adj_mat):
        seq_len, _ = v_list.shape

        hc, h_pos_emb = self.tree_emb_blk(v_list, e_list)

        tgt_mask = generate_square_subsequent_mask(seq_len)
        tgt_mask = tgt_mask.to(hc.device)

        for layer in self.chld_pred_blk:
            hc, _ = layer(
                hc,
                x,
                h_pos_emb[:, None, :],
                x_pos_emb[:, None, :],
                adj_mat,
                tgt_mask=tgt_mask,
                mem_key_padding_mask=x_mask,
            )
        v_scores = self.fc_chld(hc)
        hp = hc
        for layer in self.prnt_pred_blk:
            hp, _ = layer(
                hp,
                x,
                h_pos_emb[:, None, :],
                x_pos_emb[:, None, :],
                adj_mat,
                tgt_mask=tgt_mask,
                mem_key_padding_mask=x_mask,
            )
        vp_scores = self.fc_prnt(hp)
        e_scores = self.fc_edge(torch.cat([hc, hp], -1))
        p_scores = self.pos_pred_blk(hp, hc)

        return v_scores, vp_scores, e_scores, p_scores

    def beam_search(
        self,
        x: Tensor,
        x_mask,
        x_pos_emb,
        pad_idx,
        sos_idx,
        eos_idx,
        width,
        max_length=100,
        length_norm=0.7,
    ):
        """
        x(HW,B,E)
        x_mask(B,HW)
        x_pos_emb(HW,E)
        """
        batch_size = x.shape[1]
        result = []

        tgt_mask = generate_square_subsequent_mask(max_length).to(x.device)

        # For each instance in the batch
        for idx in range(batch_size):

            # Pick corresponding inputs
            feature_map = x[:, [idx]]  # (HW, 1, E)
            feature_mask = x_mask[[idx]]  # (1, HW)

            # Ending beams; Current beams; Candidate beams
            end = []
            beam = [BeamNode(0, [sos_idx], [pad_idx], [])]
            cand = []

            # Expand each current beam
            while beam:

                # End condition (1. max length 2. EOS)
                curr = beam.pop()
                curr_len = len(curr)
                if curr_len >= max_length or curr.v_list[-1] == eos_idx:
                    end.append(curr)
                    continue

                # Compute cout and pout
                adj_mat = x.new_zeros(1, curr_len, curr_len)
                for p in range(1, curr_len):
                    if p == 1:
                        adj_mat[0, p, 0] = 1
                    else:
                        adj_mat[0, p, curr.p_list[p - 1] + 1] = 1
                adj_mat = adj_mat + adj_mat.transpose(1, 2)
                v_list = torch.tensor(curr.v_list).unsqueeze(-1).to(
                    x.device)  # (T,1)
                e_list = torch.tensor(curr.e_list).unsqueeze(-1).to(
                    x.device)  # (T,1)
                hc, h_pos_emb = self.tree_emb_blk(v_list, e_list)
                for layer in self.chld_pred_blk:
                    hc, _ = layer(
                        hc,
                        feature_map,
                        h_pos_emb[:, None, :],
                        x_pos_emb[:, None, :],
                        adj_mat,
                        tgt_mask=tgt_mask[:curr_len, :curr_len],
                        mem_key_padding_mask=feature_mask,
                    )
                hp = hc
                for layer in self.prnt_pred_blk:
                    hp, _ = layer(
                        hp,
                        feature_map,
                        h_pos_emb[:, None, :],
                        x_pos_emb[:, None, :],
                        adj_mat,
                        tgt_mask=tgt_mask[:curr_len, :curr_len],
                        mem_key_padding_mask=feature_mask,
                    )

                # Compute scores and add to candidate
                v_scores = F.log_softmax(self.fc_chld(hc)[-1, 0],
                                        dim=-1)  # (CV)
                p_scores = F.log_softmax(self.pos_pred_blk(hp, hc)[-1, 0],
                                        dim=-1)  # (T)
                e_scores = F.log_softmax(self.fc_edge(
                    torch.cat([hc[-1, 0], hp[-1, 0]], -1)),
                                        dim=-1)  # (RV)

                v_scores, v_indices = v_scores.topk(width)  # (width)
                p_scores, p_indices = p_scores.topk(min(width,
                                                      len(p_scores)))  # (width)
                e_scores, e_indices = e_scores.topk(min(width,
                                                      len(e_scores)))  # (width)

                for v, v_scores in zip(v_indices.cpu().numpy(),
                                     v_scores.cpu().numpy()):
                    if len(curr) == 1:
                        prob = v_scores
                        cand.append(
                            BeamNode(
                                curr.prob + prob,
                                curr.v_list + [v],
                                curr.e_list + [pad_idx],
                                curr.p_list + [0],
                            ))
                        continue

                    for p, p_scores in zip(p_indices.cpu().numpy(),
                                         p_scores.cpu().numpy()):
                        for e, e_scores in zip(e_indices.cpu().numpy(),
                                             e_scores.cpu().numpy()):
                            prob = v_scores + p_scores + e_scores
                            cand.append(
                                BeamNode(
                                    curr.prob + prob,
                                    curr.v_list + [v],
                                    curr.e_list + [e],
                                    curr.p_list + [p],
                                ))

                # Cut the beam by beam-width
                if not beam:
                    cand.sort(key=lambda x: x.score(length_norm), reverse=True)
                    cand = cand[:width]
                    beam, cand = cand, beam

            # Pick the best prediction for this instance
            best = max(end, key=lambda x: x.score(length_norm))
            result.append([best.v_list, best.e_list, best.p_list])

        return result
