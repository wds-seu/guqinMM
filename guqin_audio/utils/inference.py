# inference.py
import os
import torch
from model.transformer_model import StreamingTransformerModel

def load_model(checkpoint_path, device):
    model = StreamingTransformerModel(
        vocab_size_score=128,
        vocab_size_acoustic=1024,
        style_emb_dim=256,
        d_model=256,
        nhead=8,
        num_layers=6
    )
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model

@torch.no_grad()
def inference(model, score_tokens, style_labels, max_length=2048, device="cuda"):
    """
    Args:
        model: 训练好的模型
        score_tokens: (B, T_score)  谱文token，LongTensor
        style_labels: (B)           风格标签，LongTensor
        max_length: 生成的声学token最大长度
    Returns:
        generated_acoustic_tokens: (B, max_length)
    """
    B = score_tokens.size(0)
    generated_tokens = []
    states = None
    offset = 0

    prev_tokens = torch.zeros((B, 1), dtype=torch.long, device=device)  # start token（你可以换成别的特殊token）
    
    for step in range(max_length):
        logits, states, offset = model(score_tokens, style_labels, prev_tokens, states, offset)
        logits = logits[:, -1, :]  # 取最后一个step的预测 (B, vocab_size)

        next_token = torch.argmax(logits, dim=-1, keepdim=True)  # (B, 1)

        generated_tokens.append(next_token)

        prev_tokens = next_token  # 下一步继续喂

    generated_tokens = torch.cat(generated_tokens, dim=1)  # (B, max_length)
    return generated_tokens

def main():
    # ==== 配置 ====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "exp/finetune/checkpoint_epoch50.pt"  # 你的最终微调模型
    output_dir = "exp/inference_results"
    os.makedirs(output_dir, exist_ok=True)

    # ==== 准备输入 ====
    # TODO: 这里你要换成自己的谱文tokens和style_labels，比如5首测试曲子
    # 举个例子：
    score_tokens = torch.randint(0, 128, (2, 100)).to(device)  # 假设2首曲子，每首100个谱文token
    style_labels = torch.tensor([0, 2], dtype=torch.long).to(device)  # 假设一首 exam，一首 silkqin

    # ==== 加载模型 ====
    model = load_model(checkpoint_path, device)

    # ==== 开始推理 ====
    print("🚀 Start inference...")
    generated_acoustic_tokens = inference(model, score_tokens, style_labels, max_length=2048, device=device)
    print("✅ Inference done!")

    # ==== 保存输出 ====
    save_path = os.path.join(output_dir, "generated_tokens.pt")
    torch.save({
        "score_tokens": score_tokens.cpu(),
        "style_labels": style_labels.cpu(),
        "generated_acoustic_tokens": generated_acoustic_tokens.cpu()
    }, save_path)

    print(f"🎵 Generated acoustic tokens saved at {save_path}")

if __name__ == "__main__":
    main()
