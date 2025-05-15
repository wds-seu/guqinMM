import torch
from torch.utils.data import Dataset
import os

class GuqinDataset(Dataset):
    def __init__(self, data_path, mode="train", inference_mode=False):
        """
        Args:
            data_path: 训练集路径，比如 data/processed/dataset_5pieces/
            mode: "train", "val", "test" 三种子集
            inference_mode: 是否推理模式（推理时只有score_tokens和style_label）
        """
        self.inference_mode = inference_mode
        if not inference_mode:
            data = torch.load(os.path.join(data_path, f"{mode}_data.pt"))
            self.score_tokens = data["score_tokens"]     # List of [T_score]
            self.acoustic_tokens = data["acoustic_tokens"] # List of [T_acoustic]
            self.style_labels = data["style_labels"]      # List of int
        else:
            data = torch.load(data_path)  # 直接传推理用的.pt文件
            self.score_tokens = data["score_tokens"]
            self.style_labels = data["style_labels"]

    def __len__(self):
        return len(self.score_tokens)

    def __getitem__(self, idx):
        if self.inference_mode:
            return {
                "score_tokens": self.score_tokens[idx],
                "style_label": self.style_labels[idx]
            }
        else:
            return {
                "score_tokens": self.score_tokens[idx],
                "acoustic_tokens": self.acoustic_tokens[idx],
                "style_label": self.style_labels[idx]
            }


def guqin_collate_fn(batch, inference_mode=False):
    """
    自适应的collate_fn
    支持 train / val / test / inference
    """
    score_tokens = [sample["score_tokens"] for sample in batch]
    style_labels = torch.tensor([sample["style_label"] for sample in batch], dtype=torch.long)

    max_len_score = max([len(x) for x in score_tokens])
    padded_score = torch.zeros((len(batch), max_len_score), dtype=torch.long)
    for i, tokens in enumerate(score_tokens):
        padded_score[i, :len(tokens)] = tokens

    if inference_mode:
        return {
            "score_tokens": padded_score,
            "style_label": style_labels
        }
    else:
        acoustic_tokens = [sample["acoustic_tokens"] for sample in batch]
        max_len_acoustic = max([len(x) for x in acoustic_tokens])
        padded_acoustic = torch.zeros((len(batch), max_len_acoustic), dtype=torch.long)
        for i, tokens in enumerate(acoustic_tokens):
            padded_acoustic[i, :len(tokens)] = tokens

        return {
            "score_tokens": padded_score,
            "acoustic_tokens": padded_acoustic,
            "style_label": style_labels
        }
