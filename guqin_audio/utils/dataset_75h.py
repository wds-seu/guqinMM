import torch
import os
from torch.utils.data import Dataset

class AcousticTokenDataset75h(Dataset):
    def __init__(self, acoustic_tokens_dir, label_mapping):
        """
        label_mapping: dict { 文件名前缀: 流派类别id }
        """
        self.samples = []
        for filename in sorted(os.listdir(acoustic_tokens_dir)):
            if filename.endswith(".pt"):
                path = os.path.join(acoustic_tokens_dir, filename)
                basename = filename.replace(".pt", "")
                style = label_mapping.get(basename.split("_")[0], 3)  # 默认为other
                self.samples.append((path, style))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        token_path, style_label = self.samples[idx]
        acoustic_tokens = torch.load(token_path)

        return {
            "acoustic_tokens": acoustic_tokens,
            "style_label": style_label
        }
