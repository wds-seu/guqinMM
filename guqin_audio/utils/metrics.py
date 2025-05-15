import os
import torch
import torch.nn.functional as F
import numpy as np
import torchaudio
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from scipy.stats import entropy
from encodec import EncodecModel
from tqdm import tqdm

def compute_mcc(generated_audios, conditions):
    """
    MCC: 计算生成音频和谱文条件的一致性
    方法：简单模拟，取mel特征和条件token的余弦相似度
    """
    mcc_scores = []
    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=24000, n_mels=80)

    for audio, cond in zip(generated_audios, conditions):
        mel = mel_transform(audio.unsqueeze(0))  # (1, 80, T)
        mel_feat = mel.mean(dim=-1).squeeze()    # (80,)
        cond_feat = torch.tensor(cond).float().mean().unsqueeze(0)  # 简单地以条件平均值作为特征
        cos_sim = F.cosine_similarity(mel_feat, cond_feat, dim=0)
        mcc_scores.append(cos_sim.item())

    return np.mean(mcc_scores)

def compute_fad(generated_audio_dir, real_audio_dir):
    """
    FAD: Fréchet Audio Distance
    使用 EnCodec encoder作为音频特征提取器
    """
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)
    model = model.eval().cpu()

    def extract_features(directory):
        feats = []
        for file in os.listdir(directory):
            if file.endswith(".wav"):
                path = os.path.join(directory, file)
                wav, sr = torchaudio.load(path)
                with torch.no_grad():
                    encoded_frames = model.encode(wav.unsqueeze(0))
                    latent = torch.cat([code[0] for code in encoded_frames], dim=-1)
                feats.append(latent.flatten().cpu().numpy())
        return np.vstack(feats)

    gen_feats = extract_features(generated_audio_dir)
    real_feats = extract_features(real_audio_dir)

    mu_gen = np.mean(gen_feats, axis=0)
    sigma_gen = np.cov(gen_feats, rowvar=False)
    mu_real = np.mean(real_feats, axis=0)
    sigma_real = np.cov(real_feats, rowvar=False)

    diff = mu_gen - mu_real
    covmean = np.sqrt(np.dot(sigma_gen, sigma_real) + 1e-6)

    fad = np.sum(diff ** 2) + np.trace(sigma_gen + sigma_real - 2 * covmean)
    return float(fad)

def compute_token_diversity(token_lists):
    """
    Token Diversity & Entropy
    - Diversity: unique token ratio
    - Entropy: token分布熵
    """
    all_tokens = np.concatenate(token_lists)
    total = len(all_tokens)
    unique = len(np.unique(all_tokens))

    diversity = unique / total

    counts = np.bincount(all_tokens)
    probs = counts / counts.sum()
    entropy_val = entropy(probs)

    return diversity, entropy_val

def compute_style_accuracy(generated_audio_dir, style_labels):
    """
    风格分类准确率：
    简单模拟分类器，用平均频谱特征做分类
    """
    features = []
    for file in sorted(os.listdir(generated_audio_dir), key=lambda x: int(x.split('.')[0])):
        if file.endswith(".wav"):
            path = os.path.join(generated_audio_dir, file)
            wav, sr = torchaudio.load(path)
            spec = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_mels=64)(wav)
            feat = spec.mean(dim=-1).mean(dim=-1)  # (batch, mel_dim) -> 平均
            features.append(feat.squeeze().cpu().numpy())

    features = np.stack(features)

    # 聚类模拟分类器
    kmeans = KMeans(n_clusters=4, random_state=0).fit(features)
    preds = kmeans.labels_

    acc = accuracy_score(style_labels, preds)
    return acc
