import torch
import os

def inspect_acoustic_tokens(file_path, num_samples=5):
    assert os.path.exists(file_path), f"文件不存在: {file_path}"

    data = torch.load(file_path)
    acoustic_tokens = data["acoustic_tokens"]

    print(f"✅ 成功加载: {file_path}")
    print(f"📦 样本总数量: {len(acoustic_tokens)}")
    print()

    for idx in range(min(num_samples, len(acoustic_tokens))):
        tokens = acoustic_tokens[idx]
        print(f"--- 第 {idx+1} 个样本 ---")
        print(f"Shape: {tokens.shape}")
        print(f"内容示例: {tokens[:50]}")  # 打印前50个token看看
        print(f"最大值: {tokens.max().item()}, 最小值: {tokens.min().item()}")
        print()

    # 全局统计
    all_tokens = torch.cat([x.view(-1) for x in acoustic_tokens])
    print("📈 全部数据统计:")
    print(f"总 token 数: {all_tokens.numel()}")
    print(f"最大值: {all_tokens.max().item()}, 最小值: {all_tokens.min().item()}")
    print(f"平均值: {all_tokens.float().mean().item():.2f}")
    print()

    if len(acoustic_tokens[0].shape) == 2:
        print("🚀 检测到 2维声学token！疑似 multi-stream 格式。")
    else:
        print("🎯 检测到 1维声学token，标准的单流格式。")

if __name__ == "__main__":
    # 修改成你的数据路径，比如 'data/processed_75h/train_data.pt'
    file_path = "data/processed_75h/train_data.pt"

    inspect_acoustic_tokens(file_path)
