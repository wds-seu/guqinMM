# inference_and_decode.py

import torch
from model.guqin_from_encodec import GuqinTransformerFromEncodec
from decode_audio_from_tokens import decode_tokens_to_audio

def load_finetuned_model(ckpt_path, device="cuda"):
    model = GuqinTransformerFromEncodec(
        vocab_size_score=128,
        vocab_size_acoustic=1024,
        style_emb_dim=256,
        d_model=256,
        freeze_transformer=True
    )
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    model.to(device)
    model.eval()
    return model

def generate_music(score_tokens, style_label, model, device="cuda", max_length=2048):
    model.eval()
    states = None
    offset = 0

    score_tokens = score_tokens.to(device).unsqueeze(0)
    style_label = torch.tensor([style_label], device=device)

    input_token = torch.zeros(1, 1, dtype=torch.long, device=device)
    generated_tokens = []

    for _ in range(max_length):
        logits, states, offset = model(
            score_tokens=score_tokens,
            style_labels=style_label,
            targets=input_token,
            states=states,
            offset=offset
        )
        next_token = logits.argmax(dim=-1)[:, -1]
        generated_tokens.append(next_token.item())
        input_token = next_token.unsqueeze(1)

    return generated_tokens

if __name__ == "__main__":
    device = "cuda"

    model = load_finetuned_model("./finetune_ckpt/finetune_epoch19.pt", device)

    score_tokens = torch.randint(0, 127, (32,))
    style_label = 1

    # 生成声学 token
    acoustic_tokens = generate_music(score_tokens, style_label, model, device=device, max_length=1000)

    print("生成的acoustic token数量：", len(acoustic_tokens))

    # 声学 token -> 音频
    audio = decode_tokens_to_audio(acoustic_tokens, device=device)

    print("生成音频shape:", audio.shape)  # (1, 1, T)

    # 保存为wav
    import torchaudio
    torchaudio.save("generated.wav", audio.cpu(), sample_rate=24000)
