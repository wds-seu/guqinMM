import argparse
import torch
from inference_generate import generate_music
from model.transformer_model import MultiStreamTransformer  # 你的新模型

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载模型
    model = MultiStreamTransformer()
    checkpoint = torch.load(args.model_ckpt, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()

    # 加载推理输入
    score_tokens = torch.load(args.score_tokens_path)
    style_label = args.style_label

    # 生成
    tokens = generate_music(
        model=model,
        score_tokens=score_tokens,
        style_label=style_label,
        method=args.method,
        max_length=args.max_length,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        beam_width=args.beam_width
    )

    # 保存
    torch.save(tokens, f"{args.save_dir}/generated_tokens.pt")
    print(f"Saved generated tokens at {args.save_dir}/generated_tokens.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_ckpt', type=str, required=True)
    parser.add_argument('--score_tokens_path', type=str, required=True)
    parser.add_argument('--style_label', type=int, required=True)
    parser.add_argument('--method', type=str, choices=["sampling", "beam"], default="sampling")
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--top_k', type=int, default=None)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--beam_width', type=int, default=5)
    parser.add_argument('--save_dir', type=str, default="./results")
    args = parser.parse_args()

    main(args)
