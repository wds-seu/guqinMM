import os
import torch
from encodec import EncodecModel
import torchaudio
from tqdm import tqdm

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

device = get_device()
print(f"🚀 Using device: {device}")

def extract_encodec_tokens(input_dir, output_dir):
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)
    model = model.to(device)
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    for file_name in tqdm(os.listdir(input_dir)):
        if not file_name.endswith(".wav"):
            continue
        file_path = os.path.join(input_dir, file_name)
        waveform, sr = torchaudio.load(file_path)
        waveform = waveform.to(device)

        with torch.no_grad():
            encoded_frames = model.encode(waveform.unsqueeze(0))
            acoustic_tokens = torch.cat([code[0] for code in encoded_frames], dim=-1)

        save_path = os.path.join(output_dir, file_name.replace('.wav', '.pt'))
        torch.save(acoustic_tokens.cpu(), save_path)

if __name__ == "__main__":
    extract_encodec_tokens("./data/processed_wav", "./data/processed/acoustic_tokens")
