# inference.py

import torch
from model.guqin_from_encodec import GuqinTransformerFromEncodec
from torch import nn
def inference(model, score_tokens, style_label, max_length=2048, device="cuda"):
    model.eval()
    states = None
    offset = 0

    score_tokens = score_tokens.to(device).unsqueeze(0)   # (1, T_score)
    style_label = torch.tensor([style_label], device=device)

    generated = []

    input_token = torch.zeros(1, 1, dtype=torch.long, device=device)

    for _ in range(max_length):
        logits, states, offset = model(
            score_tokens=score_tokens,
            style_labels=style_label,
            targets=input_token,
            states=states,
            offset=offset
        )
        next_token = logits.argmax(dim=-1)[:, -1]  # (1,)
        generated.append(next_token.item())

        input_token = next_token.unsqueeze(1)  # (1, 1)

    return generated

def load_model(ckpt_path, device="cuda"):
    model = GuqinTransformerFromEncodec(
        vocab_size_score=128,
        vocab_size_acoustic=1024,
        style_emb_dim=256,
        d_model=256,
        freeze_transformer=True
    )
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    model.to(device)
    model.eval()
    return model

if __name__ == "__main__":
    model = load_model("./finetune_ckpt/finetune_epoch19.pt")

    # 假设你的谱文是一个token列表
    score_tokens = torch.randint(0, 127, (16,))  # 这里随便模拟了16个字谱tokens
    style_label = 1  # 选择流派标签，比如 master=1

    acoustic_tokens = inference(model, score_tokens, style_label)

    print("Generated Acoustic Tokens:", acoustic_tokens)
