class EncodecTokenizer:
    def __init__(self):
        from encodec import EncodecModel
        self.model = EncodecModel().to("cpu")  # 或指定device
        self.model.set_target_bandwidth(6.0)

    def decode(self, acoustic_tokens):
        # acoustic_tokens shape: (num_tokens,)
        decoded_audio = self.model.decode(acoustic_tokens.unsqueeze(0))
        return decoded_audio.squeeze(0)

def decode_acoustic_tokens_to_audio(tokenizer, acoustic_tokens, sample_rate=24000):
    audio = tokenizer.decode(acoustic_tokens)
    # 简单封装成保存
    import torchaudio
    torchaudio.save("generated.wav", audio.unsqueeze(0).cpu(), sample_rate)
    return audio
