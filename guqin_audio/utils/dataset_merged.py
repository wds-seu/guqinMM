import torch
from torch.utils.data import Dataset

class MergedGuqinDataset(Dataset):
    """
    支持混合加载：
    - 有谱文、语义、声学、流派标签的完整样本
    - 只有声学、流派标签的无谱文样本（例如75h数据）
    """

    def __init__(self, pt_file_paths, max_score_len=256, max_semantic_len=256, max_acoustic_len=1024, pad_idx=0):
        """
        Args:
            pt_file_paths (list of str): 需要加载的 .pt 数据文件列表
            max_score_len (int): 谱文token最大长度
            max_semantic_len (int): 语义token最大长度
            max_acoustic_len (int): 声学token最大长度
            pad_idx (int): PAD符号在词表中的索引
        """
        self.samples = []
        for path in pt_file_paths:
            data = torch.load(path)
            self.samples.extend(data)

        self.max_score_len = max_score_len
        self.max_semantic_len = max_semantic_len
        self.max_acoustic_len = max_acoustic_len
        self.pad_idx = pad_idx

    def __len__(self):
        return len(self.samples)

    def _pad_or_truncate(self, tokens, max_len):
        """统一长度处理：pad到max_len或者截断"""
        if tokens is None:
            return torch.full((max_len,), self.pad_idx, dtype=torch.long)
        if len(tokens) >= max_len:
            return torch.tensor(tokens[:max_len], dtype=torch.long)
        else:
            padded = tokens + [self.pad_idx] * (max_len - len(tokens))
            return torch.tensor(padded, dtype=torch.long)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        score_tokens = sample.get('score_tokens', None)
        semantic_tokens = sample.get('semantic_tokens', None)
        acoustic_tokens = sample['acoustic_tokens']
        style_label = sample['style_label']

        return {
            'score_tokens': self._pad_or_truncate(score_tokens, self.max_score_len),
            'semantic_tokens': self._pad_or_truncate(semantic_tokens, self.max_semantic_len),
            'acoustic_tokens': self._pad_or_truncate(acoustic_tokens, self.max_acoustic_len),
            'style_label': torch.tensor(style_label, dtype=torch.long)
        }
