# dataset/dataset_guqin.py

import torch
from torch.utils.data import Dataset
import os

class GuqinDataset(Dataset):
    def __init__(self, data_path, mode="train", inference_mode=False):
        """
        Args:
            data_path: 数据文件夹，'data/processed_75h/'
            mode: "train", "val", "test"
            inference_mode: 推理模式（只需要score_token和style_label）
        """
        self.inference_mode = inference_mode
        if not inference_mode:
            data = torch.load(os.path.join(data_path, f"{mode}_data.pt"))
            self.score_tokens = data["score_tokens"]     # List of [T_score]
            self.acoustic_tokens = data["acoustic_tokens"] # List of [T_acoustic]
            self.style_labels = data["style_labels"]      # List of int
        else:
            data = torch.load(data_path)  # 推理模式直接传.pt文件
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
            if idx == 0:
                print(f"[Dataset] score_tokens shape: {self.score_tokens[idx].shape}")
                print(f"[Dataset] acoustic_tokens shape: {self.acoustic_tokens[idx].shape}")
                print(f"[Dataset] style_label: {self.style_labels[idx]}")
            return {
                "score_tokens": self.score_tokens[idx],
                "acoustic_tokens": self.acoustic_tokens[idx],
                "style_label": self.style_labels[idx]
            }
        
def guqin_collate_fn(batch, inference_mode=False, max_token_index=1023):
    """
    稳定版 collate_fn:
    - 自动裁剪 acoustic_tokens 超index的地方
    - 遇到异常样本直接跳过，打印警告
    """
    safe_batch = []
    for idx, sample in enumerate(batch):
        try:
            score_tokens = sample["score_tokens"]
            style_label = sample["style_label"]

            if not inference_mode:
                acoustic_tokens = sample["acoustic_tokens"]
                if (acoustic_tokens < 0).any() or (acoustic_tokens > max_token_index).any():
                    print(f"[警告] acoustic_tokens 越界：第{idx}个样本 (min={acoustic_tokens.min().item()}, max={acoustic_tokens.max().item()})，已裁剪。")
                    acoustic_tokens = acoustic_tokens.clamp(0, max_token_index)
                sample["acoustic_tokens"] = acoustic_tokens

            safe_batch.append(sample)

        except Exception as e:
            print(f"[跳过异常样本] 第{idx}个样本异常: {e}")
            continue

    if len(safe_batch) == 0:
        raise ValueError("[致命错误] 当前batch无有效样本，请检查数据集！")

    # 正常拼接batch
    score_tokens = [sample["score_tokens"] for sample in safe_batch]
    style_labels = torch.tensor([sample["style_label"] for sample in safe_batch], dtype=torch.long)

    max_len_score = max([len(x) for x in score_tokens])
    padded_score = torch.zeros((len(safe_batch), max_len_score), dtype=torch.long)
    for i, tokens in enumerate(score_tokens):
        padded_score[i, :len(tokens)] = tokens

    if inference_mode:
        return {
            "score_tokens": padded_score,
            "style_label": style_labels
        }
    else:
        acoustic_tokens = [sample["acoustic_tokens"].view(-1) for sample in safe_batch]
        max_len_acoustic = max([len(x) for x in acoustic_tokens])
        padded_acoustic = torch.zeros((len(safe_batch), max_len_acoustic), dtype=torch.long)
        for i, tokens in enumerate(acoustic_tokens):
            padded_acoustic[i, :len(tokens)] = tokens

        return {
            "score_tokens": padded_score,
            "acoustic_tokens": padded_acoustic,
            "style_label": style_labels
        }
