import torch
import os
import argparse 

def prepare_score_tokenized(score_tokenized_path, output_path, n_pieces=5):
    """
    - score_tokenized_path: 原始2651个小片段组成的list
    - output_path: 保存的新版本.pt
    - n_pieces: 要切成几首曲子，默认5
    """

    score_tokenized = torch.load(score_tokenized_path)  # 加载2651条token片段

    if not isinstance(score_tokenized, list):
        raise ValueError("Input score_tokenized should be a list.")

    total = len(score_tokenized)
    per_piece = total // n_pieces  # 每首曲子大约多少条
    samples = {}

    print(f"🔍 Total {total} segments, splitting into {n_pieces} pieces, about {per_piece} per piece...")

    for i in range(n_pieces):
        start_idx = i * per_piece
        end_idx = (i + 1) * per_piece if i != n_pieces - 1 else total  # 最后一首拿剩下所有

        merged_tokens = []
        for j in range(start_idx, end_idx):
            line_tokens = score_tokenized[j]
            merged_tokens.extend(line_tokens)  # 把这一行的所有token加入

        piece_name = f"piece_{i+1}"  # 统一命名：piece_1, piece_2, ...

        samples[piece_name] = {
            "tokens": merged_tokens,
            "style": 3  # 默认流派标签 other=3
        }

        print(f"✅ {piece_name}: {len(merged_tokens)} tokens.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(samples, output_path)
    print(f"✅ Saved new score_tokenized to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help="原始小片段谱文tokens路径")
    parser.add_argument('--output_path', type=str, required=True, help="输出的合并后谱文tokens路径")
    parser.add_argument('--n_pieces', type=int, default=5, help="要切成多少首曲子")
    args = parser.parse_args()

    prepare_score_tokenized(
        score_tokenized_path=args.input_path,
        output_path=args.output_path,
        n_pieces=args.n_pieces
    )