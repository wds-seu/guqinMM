import torch
import torch.nn.functional as F
from torch.utils.data import Sampler, RandomSampler
from torch.nn.utils.rnn import pad_sequence
from typing import Iterator, List
import pandas as pd
import random

class Vocabulary:

    pad_tok = "<PAD>"
    sos_tok = "<SOS>"
    eos_tok = "<EOS>"
    unk_tok = "<UNK>"

    def __init__(self,
                 dict_path,
                 use_pad=True,
                 use_sos=True,
                 use_eos=True,
                 use_unk=1,
                 specials=[]):
        self.itos = {}
        self.stoi = {}
        idx = 0
        if use_pad:
            self.itos.update({idx: self.pad_tok})
            self.stoi.update({self.pad_tok: idx})
            idx += 1
        if use_sos:
            self.itos.update({idx: self.sos_tok})
            self.stoi.update({self.sos_tok: idx})
            idx += 1
        if use_eos:
            self.itos.update({idx: self.eos_tok})
            self.stoi.update({self.eos_tok: idx})
            idx += 1
        if use_unk > 1:
            self.itos.update({idx: self.unk_tok})
            self.stoi.update({self.unk_tok: idx})
            idx += 1
        for tok in specials:
            self.itos.update({idx: tok})
            self.stoi.update({tok: idx})

        self.freq = {}
        self.freq_threshold = use_unk
        self._build_vocab(dict_path)

    def __len__(self):
        return len(self.itos)

    def _build_vocab(self, dict_path):
        idx = len(self.itos)
        with open(dict_path, "r") as fdict:
            for line in fdict.readlines():
                tok = line[:-1]
                if tok not in self.freq:
                    self.freq[tok] = 1
                else:
                    self.freq[tok] += 1

                if self.freq[tok] == self.freq_threshold:
                    self.stoi[tok] = idx
                    self.itos[idx] = tok
                    idx += 1

    def get_pad_idx(self):
        if self.pad_tok in self.stoi:
            return self.stoi[self.pad_tok]
        else:
            return None

    def get_sos_idx(self):
        if self.sos_tok in self.stoi:
            return self.stoi[self.sos_tok]
        else:
            return None

    def get_eos_idx(self):
        if self.eos_tok in self.stoi:
            return self.stoi[self.eos_tok]
        else:
            return None

    def tokenize(self, seq):
        return [
            self.stoi[x] if x in self.stoi else self.stoi["<UNK>"] for x in seq
        ]

    def detokenize(self, seq):
        return [self.itos[x] for x in seq]


class Collate:
    def __init__(self, node_padding=0, edge_padding=0) -> None:
        self.node_padding = node_padding
        self.edge_padding = edge_padding

    def __call__(self, batch):
        raw_h = [item[0].shape[-2] for item in batch]
        raw_w = [item[0].shape[-1] for item in batch]
        max_h = max(raw_h)
        max_w = max(raw_w)
        max_seq_len = max(len(item[2]) for item in batch)
        in_channels = batch[0][0].shape[0]
        imgs = torch.zeros(len(batch), in_channels, max_h, max_w)
        img_masks = torch.ones(len(batch), max_h, max_w, dtype=torch.bool)
        adj_mat = torch.zeros(len(batch),
                              max_seq_len,
                              max_seq_len,
                              dtype=torch.float)
        vp_list = []
        v_list = []
        p_list = []
        e_list = []
        id_list = []
        for i, item in enumerate(batch):
            imgs[i, :, :raw_h[i], :raw_w[i]] = item[0]
            img_masks[i, :raw_h[i], :raw_w[i]] = False
            vp_list.append(torch.tensor(item[2]["src_type"].to_list()))
            v_list.append(torch.tensor(item[1]["type"].to_list()))
            parent_pos = torch.tensor(item[2]["src"].to_list())
            p_list.append(parent_pos)
            for j in range(len(parent_pos) - 1):
                adj_mat[i,
                        torch.tensor(item[2].loc[j, "dst"]), parent_pos[j]] = 1
            adj_mat = adj_mat.transpose(1, 2) + adj_mat
            e_list.append(torch.tensor(item[2]["type"].to_list()))
            id_list.append(item[-1])
        vp_list = pad_sequence(vp_list, padding_value=self.node_padding)
        v_list = pad_sequence(v_list, padding_value=self.node_padding)
        p_list = pad_sequence(p_list, padding_value=-1)
        p_list = ((p_list >= 0) * -1) + p_list
        e_list = pad_sequence(e_list, padding_value=self.edge_padding)
        e_list = F.pad(e_list, (0, 0, 1, 0), value=0)

        return imgs, img_masks, v_list, vp_list, p_list, e_list, adj_mat, id_list


class AdaptiveBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        metadata: pd.DataFrame,
        batch_size: int = 8,
        rand_size: int = 128,
        mem_size=1.6e5,
    ) -> None:
        self.sampler = RandomSampler(metadata)
        self.batch_size = batch_size
        self.rand_size = rand_size
        self.mem_size = mem_size
        self.metadata = metadata
        self.batches = []
        self._sample()

    def _sample(self):
        self.batches = []
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.rand_size:
                batch = sorted(
                    batch,
                    key=lambda x:
                    (self.metadata.loc[x, "h"], self.metadata.loc[x, "w"]))
                max_h = 0
                max_w = 0
                sub_batch = []
                for idy in batch:
                    if self.metadata.loc[idy, "h"] > max_h:
                        max_h = self.metadata.loc[idy, "h"]
                    if self.metadata.loc[idy, "w"] > max_w:
                        max_w = self.metadata.loc[idy, "w"]
                    cur_mem_size = max_h * max_w * len(sub_batch)
                    if (cur_mem_size > self.mem_size
                            or len(sub_batch) == self.batch_size):
                        self.batches.append(sub_batch)
                        sub_batch = [idy]
                        max_h = self.metadata.loc[idy, "h"]
                        max_w = self.metadata.loc[idy, "w"]
                    else:
                        sub_batch.append(idy)
                if len(sub_batch) > 0:
                    self.batches.append(sub_batch)
                batch = []
        random.shuffle(self.batches)

    def __iter__(self) -> Iterator[List[int]]:
        for sub_batch in self.batches:
            yield sub_batch
        self._sample()
