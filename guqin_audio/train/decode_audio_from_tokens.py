# decode_audio_from_tokens.py

import torch
from encodec import EncodecModel

def decode_tokens_to_audio(tokens, model=None, device="cuda"):
    """
    Args:
        tokens: list or tensor of acoustic tokens (1D, 单流 or 多流）
        model: 已加载好的 EncodecModel
    Returns:
        audio: [1, 1, T] float32 waveform, 24kHz
    """
    if model is None:
        model = EncodecModel.encodec_model_24khz()
        model.set_target_bandwidth(6.0)  # 6kbps, 默认
        model.to(device)
        model.eval()

    if isinstance(tokens, list):
        tokens = torch.tensor(tokens, device=device)
    elif isinstance(tokens, torch.Tensor):
        tokens = tokens.to(device)

    # 还原成 EnCodec 需要的多流输入格式
    if tokens.ndim == 1:
        # 单个流，强制变成(1, n_q, T)
        tokens = tokens.unsqueeze(0).unsqueeze(0)
    elif tokens.ndim == 2:
        tokens = tokens.unsqueeze(0)

    with torch.no_grad():
        audio = model.decode(tokens)

    return audio
