import os
import torch
from torch.utils.data import DataLoader
from utils.inference import GuqinInference
from utils.dataset_merged import MergedGuqinDataset
from utils.metrics import compute_mcc, compute_fad, compute_token_diversity, compute_style_accuracy

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def evaluate_experiment(exp_name):
    """
    exp_name: str, 需要评估的实验名称
    """

    device = get_device()
    print(f"\n🚀 Evaluating experiment: {exp_name} on device {device}")

    config = {
        "device": device,
        "semantic_vocab_size": 4096,  # 可以根据不同实验适当调整
        "acoustic_vocab_size": 2048,
        "embedding_dim": 256,
        "max_seq_len": 1024
    }

    # 加载模型
    inference_model = GuqinInference(
        config,
        semantic_ckpt_path=f"checkpoints/{exp_name}/semantic_best.pth",
        acoustic_ckpt_path=f"checkpoints/{exp_name}/acoustic_best.pth"
    )

    # 加载测试数据
    test_dataset = MergedGuqinDataset([
        "data/processed/val_data.pt"
    ])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    os.makedirs(f"outputs/{exp_name}", exist_ok=True)

    generated_audios = []
    real_audios = []
    conditions = []
    styles = []
    generated_tokens = []

    for idx, batch in enumerate(test_loader):
        score_tokens = batch['score_tokens'].squeeze(0).tolist()
        style_label = batch['style_label'].item()

        # 推理生成
        waveform = inference_model.generate(score_tokens, style_label)
        out_path = f"outputs/{exp_name}/{idx}.wav"
        inference_model.save_waveform(waveform, out_path)

        generated_audios.append(waveform.squeeze(0))
        real_audios.append(batch['acoustic_tokens'].squeeze(0))
        conditions.append(score_tokens)
        styles.append(style_label)

        # 收集用于 diversity 计算
        generated_tokens.append(batch['acoustic_tokens'].cpu().numpy())

    # --- 评估各项指标 ---
    print(f"🔍 Computing metrics for {exp_name}...")

    mcc_score = compute_mcc(generated_audios, conditions)
    fad_score = compute_fad(f"outputs/{exp_name}/", "data/processed_wav/")  # 真实音频路径
    token_diversity, token_entropy = compute_token_diversity(generated_tokens)
    style_acc = compute_style_accuracy(f"outputs/{exp_name}/", styles)

    print(f"✅ {exp_name} results:")
    print(f"MCC: {mcc_score:.4f}")
    print(f"FAD: {fad_score:.4f}")
    print(f"Token Diversity: {token_diversity:.4f}")
    print(f"Token Entropy: {token_entropy:.4f}")
    print(f"Style Classification Accuracy: {style_acc:.4f}")

    return {
        "mcc": mcc_score,
        "fad": fad_score,
        "diversity": token_diversity,
        "entropy": token_entropy,
        "style_acc": style_acc
    }

if __name__ == "__main__":
    experiments = [
        "baseline_encodec_w2vbert",
        "soundstream_w2vbert",
        "encodec_hubert",
        "specvqvae_cpc",
        "small_vocab_1024",
    ]

    all_results = {}

    for exp_name in experiments:
        result = evaluate_experiment(exp_name)
        all_results[exp_name] = result

    torch.save(all_results, "./logs/evaluation_results.pt")
    print("\n🎯 All experiment evaluations finished.")
