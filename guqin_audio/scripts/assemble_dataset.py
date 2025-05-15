import os
import torch
import random
from tqdm import tqdm

def assemble_dataset(score_tokenized_path, acoustic_tokens_dir, semantic_tokens_dir, output_dir):
    print("🔍 Loading score tokens...")
    score_tokenized = torch.load(score_tokenized_path)

    samples = []

    print("🔍 Assembling samples...")

    if isinstance(score_tokenized, dict):
        keys = list(score_tokenized.keys())
    elif isinstance(score_tokenized, list):
        # 新版逻辑：直接按 acoustic_tokens_dir 的文件名来配对
        file_list = sorted([f.replace('.pt', '') for f in os.listdir(acoustic_tokens_dir) if f.endswith('.pt')])
        if len(score_tokenized) != len(file_list):
            raise ValueError(f"Length mismatch: {len(score_tokenized)} tokens vs {len(file_list)} files")
        keys = file_list
    else:
        raise ValueError("Unsupported score_tokenized format.")

    for idx, key in enumerate(tqdm(keys)):
        acoustic_path = os.path.join(acoustic_tokens_dir, key + ".pt")
        semantic_path = os.path.join(semantic_tokens_dir, key + "_semantic.pt")

        if not os.path.exists(acoustic_path) or not os.path.exists(semantic_path):
            continue

        acoustic_tokens = torch.load(acoustic_path)
        semantic_tokens = torch.load(semantic_path)

        if isinstance(score_tokenized, dict):
            score_tokens = score_tokenized[key]["tokens"]
            style_label = score_tokenized[key]["style"]
        else:
            score_tokens = torch.tensor(score_tokenized[idx], dtype=torch.long)
            style_label = 3  # 默认 other流派 (如果没有其他标签)

        sample = {
            "score_tokens": score_tokens,
            "acoustic_tokens": acoustic_tokens,
            "semantic_tokens": semantic_tokens,
            "style_label": style_label
        }
        samples.append(sample)

    print(f"✅ Collected {len(samples)} samples.")

    random_split(samples, output_dir)

def random_split(samples, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    os.makedirs(output_dir, exist_ok=True)
    n = len(samples)
    train_num = int(train_ratio * n)
    val_num = int(val_ratio * n)

    train_samples = samples[:train_num]
    val_samples = samples[train_num:train_num+val_num]
    test_samples = samples[train_num+val_num:]

    torch.save(train_samples, os.path.join(output_dir, "train_data.pt"))
    torch.save(val_samples, os.path.join(output_dir, "val_data.pt"))
    torch.save(test_samples, os.path.join(output_dir, "test_data.pt"))

    print(f"📦 Saved train ({len(train_samples)}), val ({len(val_samples)}), test ({len(test_samples)}) samples.")

if __name__ == "__main__":
    assemble_dataset(
        "./data/score_tokenized.pt",
        "./data/processed/acoustic_tokens",
        "./data/processed/semantic_tokens",
        "./data/processed"
    )
