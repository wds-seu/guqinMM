import os
import torch
import torchaudio
import argparse
from encodec import EncodecModel
from tqdm import tqdm

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:1")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
    

def resample_audio(input_dir, output_dir):
    """Step 1: 重采样为24kHz单声道wav"""
    print("🔄 Step 1: Resampling audio to 24kHz mono...")

    os.makedirs(output_dir, exist_ok=True)

    for style_folder in os.listdir(input_dir):
        style_path = os.path.join(input_dir, style_folder)
        if not os.path.isdir(style_path):
            continue
            
        output_style_path = os.path.join(output_dir, style_folder)
        os.makedirs(output_style_path, exist_ok=True)
        
        # 使用os.walk递归获取所有MP3/WAV文件
        audio_files = []
        for root, _, files in os.walk(style_path):
            for file in files:
                if file.endswith('.mp3') or file.endswith('.wav'):
                    audio_files.append((os.path.join(root, file), file))
        
        for input_path, orig_filename in tqdm(audio_files, desc=f"Resampling {style_folder}"):
            # 生成唯一文件名，避免冲突
            rel_path = os.path.relpath(os.path.dirname(input_path), style_path)
            if rel_path != '.':
                unique_prefix = rel_path.replace('/', '_').replace('\\', '_') + '_'
            else:
                unique_prefix = ''
                
            output_filename = unique_prefix + orig_filename
            output_path = os.path.join(output_style_path, 
                                      output_filename.replace('.mp3', '.wav').replace('.flac', '.wav'))
            
            waveform, sr = torchaudio.load(input_path)
            if sr != 24000:
                waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=24000)
            # 保证单声道
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # 修正：保存为24000采样率，与重采样一致
            torchaudio.save(output_path, waveform, 24000)

    print("✅ Resampling finished.")


def extract_acoustic_tokens_with_chunking(resampled_dir, token_output_dir, device, chunk_seconds=30):
    """
    提取声学token，并且将超长音频自动切成chunk，每个chunk长度控制在chunk_seconds秒以内。
    """
    print("🔄 Step 2 (Updated): Extracting acoustic tokens with chunking...")

    os.makedirs(token_output_dir, exist_ok=True)

    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)
    model = model.to(device).eval()

    style_mapping = {"exam": 0, "master": 1, "silkqin": 2, "other": 3}
    sample_rate = 24000
    chunk_samples = chunk_seconds * sample_rate

    for style_folder in os.listdir(resampled_dir):
        style_path = os.path.join(resampled_dir, style_folder)
        if not os.path.isdir(style_path):
            continue

        for file_name in tqdm(os.listdir(style_path), desc=f"Encoding {style_folder}"):
            if not file_name.endswith(".wav"):
                continue
            input_path = os.path.join(style_path, file_name)

            try:
                waveform, sr = torchaudio.load(input_path)
                if sr != sample_rate:
                    waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=sample_rate)

                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)

                waveform = waveform.float().contiguous()

                # waveform shape: [1, time]
                total_samples = waveform.shape[1]

                if total_samples < 16000:
                    print(f"Skipping {file_name}: too short ({total_samples} samples)")
                    continue

                # 切片处理
                num_chunks = (total_samples + chunk_samples - 1) // chunk_samples
                for chunk_idx in range(num_chunks):
                    start = chunk_idx * chunk_samples
                    end = min((chunk_idx + 1) * chunk_samples, total_samples)
                    chunk = waveform[:, start:end]

                    if chunk.shape[1] < 16000:
                        continue

                    chunk = chunk.unsqueeze(0).to(device)  # [1, 1, time]

                    with torch.no_grad():
                        encoded_frames = model.encode(chunk)
                        tokens = torch.cat([code[0] for code in encoded_frames], dim=-1)

                    sample = {
                        "acoustic_tokens": tokens.cpu(),
                        "style_label": style_mapping.get(style_folder, 3)
                    }

                    save_name = f"{style_folder}_{file_name.replace('.wav', '')}_chunk{chunk_idx}.pt"
                    torch.save(sample, os.path.join(token_output_dir, save_name))

            except Exception as e:
                print(f"Error processing {input_path}: {e}")
                continue

    print("✅ Acoustic tokens extraction with chunking finished.")


def assemble_dataset(token_dir, output_path):
    """Step 3: 整合成一个.pt数据文件"""
    print("🔄 Step 3: Assembling dataset...")

    samples = []
    for file_name in tqdm(os.listdir(token_dir), desc="Merging tokens"):
        if file_name.endswith(".pt"):
            sample = torch.load(os.path.join(token_dir, file_name))
            samples.append(sample)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(samples, output_path)

    print(f"✅ Dataset assembled. Saved to {output_path} ({len(samples)} samples)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', type=str, default="./data/raw_75h", help="原始mp3文件夹")
    parser.add_argument('--resampled_dir', type=str, default="./data/processed_wav_75h", help="重采样后的wav输出文件夹")
    parser.add_argument('--acoustic_token_dir', type=str, default="./data/acoustic_tokens_75h", help="声学tokens输出文件夹")
    parser.add_argument('--output_dataset', type=str, default="./data/processed_75h/75h_dataset.pt", help="最终合并输出文件")
    parser.add_argument("--chunk_seconds", type=int, default=30, help="每段音频最大秒数（默认30秒）")
    args = parser.parse_args()

    device = get_device()
    print(f"🚀 Using device: {device}")

    # resample_audio(args.raw_dir, args.resampled_dir)
    extract_acoustic_tokens_with_chunking(
        args.resampled_dir,
        args.acoustic_token_dir,
        device=device,
        chunk_seconds=args.chunk_seconds
    )
    assemble_dataset(args.acoustic_token_dir, args.output_dataset)

if __name__ == "__main__":
    main()
