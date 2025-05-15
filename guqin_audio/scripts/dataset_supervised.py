# scripts/dataset_supervised.py

import torch
from torch.utils.data import Dataset
import os

class SupervisedAcousticDataset(Dataset):
    def __init__(self, dataset_path):
        data = torch.load(dataset_path)
        self.score_tokens = data["score_tokens"]     # List of [T_score]
        self.acoustic_tokens = data["acoustic_tokens"] # List of [T_acoustic]
        self.style_labels = data["style_labels"]      # List of int

    def __len__(self):
        return len(self.acoustic_tokens)

    def __getitem__(self, idx):
        return {
            "score_tokens": self.score_tokens[idx],
            "acoustic_tokens": self.acoustic_tokens[idx],
            "style_label": self.style_labels[idx]
        }
