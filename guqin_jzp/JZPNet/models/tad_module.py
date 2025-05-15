import pytorch_lightning as pl

from .tadnet import TADNet
from .encoder import Encoder

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from .utils import (
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
)
import editdistance as ed
from tqdm import tqdm
import numpy as np


class Recorder(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.add_state("num", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("denom",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")

    def update(self, num, denom, reduced=False):
        if reduced:
            self.num += num * denom
        else:
            self.num += num
        self.denom += denom

    def compute(self):
        return self.num / self.denom


class TSDNetModule(pl.LightningModule):
    def __init__(
        self,
        d_model,
        node_vocab_size,
        edge_vocab_size,
        node_emb_size,
        edge_emb_size,
        pad_idx,
        dropout=0.3,
        enc_d_in=684,
        enc_growth_rate=24,
        enc_block_config=16,
        enc_in_channels=1,
        dec_n_head=8,
        dec_d_ff=1024,
        dec_nc=3,
        dec_np=3,
        dec_max_num_hops=2,
        dec_activation="relu",
        lr=3e-4,
        warmup=20,
        lambda_v=1,
        lambda_vp=0.1,
        lambda_e=1,
        labmda_p=1,
        label_smoothing=0,
        optim="adamw",
        cos_epochs=300,
        restart="hard",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = Encoder(
            enc_d_in,
            d_model,
            enc_growth_rate,
            enc_block_config,
            dropout=dropout,
            in_channels=enc_in_channels,
        )
        self.decoder = TADNet(
            d_model,
            node_vocab_size,
            edge_vocab_size,
            node_emb_size,
            edge_emb_size,
            dec_nc,
            dec_np,
            dec_n_head,
            dec_d_ff,
            dropout,
            dec_activation,
            dec_max_num_hops,
        )
        self.lr = lr
        self.warmup = warmup
        self.pad_idx = pad_idx
        self.lambda_v = lambda_v
        self.lambda_vp = lambda_vp
        self.labmda_p = labmda_p
        self.lambda_e = lambda_e
        self.optim = optim
        self.cos_epochs = cos_epochs
        self.restart = restart
        self.celoss_ve = nn.CrossEntropyLoss(ignore_index=pad_idx,
                                             label_smoothing=label_smoothing)
        self.celoss_pos = nn.CrossEntropyLoss(ignore_index=-1,
                                              label_smoothing=label_smoothing)
        self.v_loss = nn.ModuleDict()
        self.vp_loss = nn.ModuleDict()
        self.e_loss = nn.ModuleDict()
        self.p_loss = nn.ModuleDict()
        self.v_acc = nn.ModuleDict()
        self.vp_acc = nn.ModuleDict()
        self.e_acc = nn.ModuleDict()
        self.p_acc = nn.ModuleDict()
        self.v_seq_acc = nn.ModuleDict()
        self.vp_seq_acc = nn.ModuleDict()
        self.e_seq_acc = nn.ModuleDict()
        self.p_seq_acc = nn.ModuleDict()
        self.seq_acc = nn.ModuleDict()
        for stage in ["trn", "val", "tst"]:
            self.v_loss[stage] = Recorder()
            self.vp_loss[stage] = Recorder()
            self.e_loss[stage] = Recorder()
            self.p_loss[stage] = Recorder()
            self.v_acc[stage] = Recorder()
            self.vp_acc[stage] = Recorder()
            self.e_acc[stage] = Recorder()
            self.p_acc[stage] = Recorder()
            self.v_seq_acc[stage] = Recorder()
            self.vp_seq_acc[stage] = Recorder()
            self.e_seq_acc[stage] = Recorder()
            self.p_seq_acc[stage] = Recorder()
            self.seq_acc[stage] = Recorder()

    def _shared_step(self, batch, stage):
        imgs, img_masks, v_list, vp_list, p_list, e_list, adj_mat, _ = batch

        x, mask, pos_emb = self.encoder(imgs, img_masks)
        v_scores, vp_scores, e_scores, p_scores = self.decoder(
            x, mask, pos_emb, v_list[:-1], e_list[:-1, :], adj_mat)
        
        v_list = v_list[1:]
        e_list = e_list[1:]

        v_loss = self.celoss_ve(v_scores.flatten(0, 1), v_list.flatten())
        vp_loss = self.celoss_ve(vp_scores.flatten(0, 1), vp_list.flatten())
        e_loss = self.celoss_ve(e_scores.flatten(0, 1), e_list.flatten())
        p_loss = self.celoss_pos(p_scores.flatten(0, 1), p_list.flatten())
        loss = (self.lambda_v * v_loss + self.lambda_vp * vp_loss +
                self.lambda_e * e_loss + self.labmda_p * p_loss)

        node_mask = v_list != self.pad_idx
        edge_mask = e_list != self.pad_idx
        pos_mask = p_list != -1
        num_nodes = torch.sum(node_mask)
        num_edges = torch.sum(edge_mask)
        num_pos = torch.sum(pos_mask)
        batch_size = v_list.shape[1]
        v_pred = v_scores.argmax(-1)
        vp_pred = vp_scores.argmax(-1)
        e_pred = e_scores.argmax(-1)
        p_pred = p_scores.argmax(-1)
        v_correct = v_pred == v_list
        vp_correct = vp_pred == vp_list
        e_correct = e_pred == e_list
        p_correct = p_pred == p_list
        v_seq_correct = torch.all(torch.logical_or(v_correct, ~node_mask),
                                  dim=0)
        vp_seq_correct = torch.all(torch.logical_or(vp_correct, ~node_mask),
                                   dim=0)
        e_seq_correct = torch.all(torch.logical_or(e_correct, ~edge_mask),
                                  dim=0)
        p_seq_correct = torch.all(torch.logical_or(p_correct, ~pos_mask),
                                  dim=0)
        self.v_loss[stage](v_loss, num_nodes, True)
        self.vp_loss[stage](vp_loss, num_nodes, True)
        self.e_loss[stage](e_loss, num_edges, True)
        self.p_loss[stage](p_loss, num_pos, True)
        self.v_acc[stage](v_correct[node_mask].sum(), num_nodes)
        self.vp_acc[stage](vp_correct[node_mask].sum(), num_nodes)
        self.e_acc[stage](e_correct[edge_mask].sum(), num_edges)
        self.p_acc[stage](p_correct[pos_mask].sum(), num_pos)
        self.v_seq_acc[stage](v_seq_correct.sum(), batch_size)
        self.vp_seq_acc[stage](vp_seq_correct.sum(), batch_size)
        self.e_seq_acc[stage](e_seq_correct.sum(), batch_size)
        self.p_seq_acc[stage](p_seq_correct.sum(), batch_size)
        self.seq_acc[stage](
            (e_seq_correct * v_seq_correct * p_seq_correct).sum(), batch_size)
        self.log(
            f"{stage}_loss",
            {
                "v": self.v_loss[stage],
                "vp": self.vp_loss[stage],
                "e": self.e_loss[stage],
                "p": self.p_loss[stage],
            },
        )
        self.log(
            f"{stage}_sum_loss",
            loss,
        )
        self.log(
            f"{stage}_acc",
            {
                "v": self.v_acc[stage],
                "vp": self.vp_acc[stage],
                "e": self.e_acc[stage],
                "p": self.p_acc[stage],
            },
        )
        self.log(
            f"{stage}_seq_acc",
            {
                "v": self.v_seq_acc[stage],
                "vp": self.vp_seq_acc[stage],
                "e": self.e_seq_acc[stage],
                "p": self.p_seq_acc[stage],
            },
        )
        self.log(
            f"{stage}_ExpRate",
            self.seq_acc[stage],
        )
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, "trn")
        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "tst")

    def beam_test(self,
                  dl,
                  node_vocab,
                  edge_vocab,
                  width=1,
                  max_len=200,
                  length_norm=0,
                  test_log=None):
        # model.beam_test(dm.test_dataloader(), node_vocab, edge_vocab)
        dists = []
        stru_dists = []

        for batch in tqdm(dl, leave=False):
            imgs, img_masks, v_list, _, p_list, e_list, _, id_list = batch
            
            output = self(imgs, img_masks, node_vocab, edge_vocab, width,
                          max_len, length_norm)            
            
            node_eos = node_vocab.get_eos_idx()
            edge_pad = edge_vocab.get_pad_idx()

            v_list = v_list.cpu().numpy()
            e_list = e_list[1:].cpu().numpy()
            p_list = p_list.cpu().numpy()

            batch_size = imgs.shape[0]
            for i in range(batch_size):

                id = id_list[i]

                v_gt = list(v_list[:, i])
                e_gt = list(e_list[:, i])
                p_gt = list(p_list[:, i])

                v_gt = v_gt[1:v_gt.index(node_eos)]
                e_gt = e_gt[1:e_gt[1:].index(edge_pad) + 1]
                try:
                    p_gt = p_gt[1:p_gt[1:].index(-1)]
                except:
                    p_gt = p_gt[1:-1]

                assert len(v_gt) == len(e_gt) + 1
                assert len(e_gt) == len(p_gt)
                
                
                v_pred, e_pred, p_pred = output[i]

                v_pred = v_pred[1:-1]
                e_pred = e_pred[2:-1]
                p_pred = p_pred[1:-1]

                if len(v_pred) != 0:
                    assert len(v_pred) == len(e_pred) + 1
                    assert len(e_pred) == len(p_pred)   
                dists.append(
                    ed.eval(v_pred, v_gt) + ed.eval(e_pred, e_gt) +
                    ed.eval(p_pred, p_gt))
                stru_dists.append(
                    ed.eval(p_pred, p_gt) + ed.eval(e_pred, e_gt))
                if test_log:
                    with open(test_log, "a") as f:
                        f.write(f"{id}\n")
                        f.write(f"{list(v_gt)}\n{list(v_pred)}\n")
                        f.write(f"{list(e_gt)}\n{list(e_pred)}\n")
                        f.write(f"{list(p_gt)}\n{list(p_pred)}\n")
                        f.write("\n")

        dists = np.array(dists)
    
        
        stru_dists = np.array(stru_dists)
        exp_rate = np.count_nonzero(dists <= 0) / len(dists)
        ed1 = np.count_nonzero(dists <= 1) / len(dists)
        ed2 = np.count_nonzero(dists <= 2) / len(dists)
        stru_rate = np.count_nonzero(stru_dists <= 0) / len(stru_dists)
        print(
            f"exp_rate: {exp_rate}\ned1: {ed1}\ned2: {ed2}\nstru_rate: {stru_rate}"
        )

    def forward(self, imgs, img_masks, node_vocab, edge_vocab, width, max_len,
                length_norm):
        # imgs(B,C,H,W)
        # img_masks(B,H,W)
        x, mask, pos_emb = self.encoder(imgs, img_masks)
        
        dec_rsts = self.decoder.beam_search(
            x,
            mask,
            pos_emb,
            node_vocab.stoi[node_vocab.pad_tok],
            node_vocab.stoi[node_vocab.sos_tok],
            node_vocab.stoi[node_vocab.eos_tok],
            width,
            max_len,
            length_norm,
        )
        return dec_rsts

    def configure_optimizers(self):
        if self.optim == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        elif self.optim == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            raise ValueError
        scheduler_config = {}
        if self.restart == "hard":
            scheduler_config[
                "scheduler"] = get_cosine_with_hard_restarts_schedule_with_warmup(
                    optimizer, self.warmup, self.cos_epochs)
        elif self.restart == "soft":
            scheduler_config["scheduler"] = get_cosine_schedule_with_warmup(
                optimizer, self.warmup, self.cos_epochs)
        else:
            raise ValueError
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config,
        }
