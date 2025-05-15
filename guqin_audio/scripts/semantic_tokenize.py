import os
import random
import torch
import torchaudio
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model

def extract_semantic_tokens(input_dir, output_dir, model_name="facebook/wav2vec2-base-960h", n_clusters=4096, sample_rate=0.05):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name).to(device).eval()

    os.makedirs(output_dir, exist_ok=True)

    all_samples = []

    print("🔍 Step 1: Collecting sampled features for clustering...")

    for file_name in tqdm(os.listdir(input_dir)):
        if not file_name.endswith(".wav"):
            continue
        path = os.path.join(input_dir, file_name)
        wav, sr = torchaudio.load(path)
        wav = wav.to(device)

        with torch.no_grad():
            inputs = processor(wav.squeeze(0), sampling_rate=sr, return_tensors="pt", padding=True).input_values
            inputs = inputs.to(device)
            features = model(inputs).last_hidden_state.squeeze(0).cpu().numpy()

        # 随机下采样
        num_frames = features.shape[0]
        sample_size = int(num_frames * sample_rate)
        idx = np.random.choice(num_frames, sample_size, replace=False)
        sampled_features = features[idx]

        all_samples.append(sampled_features)

    all_samples = np.vstack(all_samples)
    print(f"Collected {all_samples.shape[0]} frames for clustering.")

    effective_clusters = min(n_clusters, all_samples.shape[0] // 2)
    if effective_clusters < n_clusters:
        print(f"⚠️ Warning: not enough samples. Adjusting clusters to {effective_clusters}.")


    # Step 2: 聚类
    print(f"🔍 Step 2: Running MiniBatchKMeans (clusters={n_clusters})...")
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=4096*10, random_state=0)
    kmeans.fit(all_samples)

    # Step 3: 量化每个文件
    print(f"🔍 Step 3: Quantizing and saving semantic tokens...")

    for file_name in tqdm(os.listdir(input_dir)):
        if not file_name.endswith(".wav"):
            continue
        path = os.path.join(input_dir, file_name)
        wav, sr = torchaudio.load(path)
        wav = wav.to(device)

        with torch.no_grad():
            inputs = processor(wav.squeeze(0), sampling_rate=sr, return_tensors="pt", padding=True).input_values
            inputs = inputs.to(device)
            features = model(inputs).last_hidden_state.squeeze(0).cpu().numpy()

        tokens = kmeans.predict(features)

        save_path = os.path.join(output_dir, file_name.replace('.wav', '_semantic.pt'))
        torch.save(tokens, save_path)

    print("✅ Semantic token extraction done.")

if __name__ == "__main__":
    extract_semantic_tokens("data/processed_wav", "data/processed/semantic_tokens")