import os
import torch
import argparse

def text_to_tokenized_list(text_file, output_path):
    """
    - text_file: 你的原始总txt文件（5302行）
    - output_path: 输出为list，每行一个token列表
    """
    tokenized = []

    with open(text_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 按英文逗号切分
        tokens = line.split(',')
        # 清理每个token：去除引号和前后空格
        tokens = [t.strip().strip("'").strip('"') for t in tokens if t.strip()]
        
        if tokens:
            tokenized.append(tokens)

    print(f"✅ Parsed {len(tokenized)} lines from {text_file}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(tokenized, output_path)
    print(f"✅ Saved tokenized data to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_file', type=str, required=True, help="原始总txt文件路径")
    parser.add_argument('--output_path', type=str, required=True, help="保存的.pt文件路径")
    args = parser.parse_args()

    text_to_tokenized_list(
        text_file=args.text_file,
        output_path=args.output_path
    )
