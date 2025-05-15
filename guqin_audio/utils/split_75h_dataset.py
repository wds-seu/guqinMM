import torch
import os
import random

def split_dataset(dataset_path, output_dir, train_ratio=0.8, val_ratio=0.1):
    os.makedirs(output_dir, exist_ok=True)

    data = torch.load(dataset_path)
    
    # 打印数据结构，帮助调试
    print(f"Data type: {type(data)}")
    if isinstance(data, list) and len(data) > 0:
        print(f"First element type: {type(data[0])}")
        if isinstance(data[0], dict):
            print(f"Keys in first element: {data[0].keys()}")
    
    # 基于数据实际格式进行处理
    if isinstance(data, dict):
        # 原始预期的字典格式
        acoustic_tokens = data["acoustic_tokens"]
        style_labels = data["style_labels"]
    elif isinstance(data, list):
        # 如果数据是列表，尝试提取acoustic_tokens和style_labels
        if len(data) > 0 and isinstance(data[0], dict):
            # 列表中包含字典元素
            acoustic_tokens = [item.get("acoustic_tokens") for item in data]
            style_labels = [item.get("style_label", 0) for item in data]  # 默认为0
        else:
            # 简单列表，假设每个元素就是acoustic_tokens
            acoustic_tokens = data
            style_labels = [0] * len(data)  # 默认所有样本style为0
    else:
        raise ValueError(f"Unsupported data format: {type(data)}")

    indices = list(range(len(acoustic_tokens)))
    random.shuffle(indices)

    n_total = len(indices)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]

    def make_subset(idxs):
        return {
            "score_tokens": [torch.zeros(1, dtype=torch.long)] * len(idxs),  # 无谱文，所以dummy
            "acoustic_tokens": [acoustic_tokens[i] for i in idxs],
            "style_labels": [style_labels[i] for i in idxs]
        }

    torch.save(make_subset(train_idx), os.path.join(output_dir, "train_data.pt"))
    torch.save(make_subset(val_idx), os.path.join(output_dir, "val_data.pt"))
    torch.save(make_subset(test_idx), os.path.join(output_dir, "test_data.pt"))

    print(f"✅ Dataset split done! Total={n_total} samples")
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

if __name__ == "__main__":
    split_dataset("data/processed_75h/75h_dataset.pt", "data/processed_75h")