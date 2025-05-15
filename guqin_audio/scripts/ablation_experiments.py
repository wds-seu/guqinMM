import os
import torch
from torch.utils.data import DataLoader
from model.semantic_token_generator import SemanticTokenGenerator
from model.acoustic_token_generator import AcousticTokenGenerator
from utils.trainer import Trainer
from utils.dataset_merged import MergedGuqinDataset

def get_device():
    """自动检测设备（CUDA优先，MPS其次，否则CPU）"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def build_models(config):
    """根据config构建模型"""
    semantic_model = SemanticTokenGenerator(
        vocab_size=config["semantic_vocab_size"],
        embed_dim=config["embedding_dim"],
        max_seq_len=config["max_seq_len"]
    ).to(config["device"])

    acoustic_model = AcousticTokenGenerator(
        vocab_size=config["acoustic_vocab_size"],
        embed_dim=config["embedding_dim"],
        max_seq_len=config["max_seq_len"]
    ).to(config["device"])

    return semantic_model, acoustic_model

def build_dataloader(config, train=True):
    """加载训练/验证数据集"""
    if train:
        dataset = MergedGuqinDataset([
            "data/processed/train_data.pt",
            "data/processed_75h/75h_dataset.pt"
        ])
    else:
        dataset = MergedGuqinDataset([
            "data/processed/val_data.pt"
        ])

    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=train, drop_last=train)
    return loader

def run_experiment(exp_name, semantic_vocab_size, acoustic_vocab_size):
    """单组实验执行流程"""
    device = get_device()
    config = {
        "device": device,
        "batch_size": 8,
        "learning_rate": 1e-4,
        "num_epochs": 20,
        "max_seq_len": 1024,
        "semantic_vocab_size": semantic_vocab_size,
        "acoustic_vocab_size": acoustic_vocab_size,
        "score_vocab_size": 5000,  # 根据项目设定
        "style_vocab_size": 4,
        "embedding_dim": 256,
        "checkpoint_dir": f"checkpoints/{exp_name}",
        "log_dir": f"logs/{exp_name}"
    }

    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["log_dir"], exist_ok=True)

    print(f"\n🚀 Starting experiment: {exp_name} on device {device}")

    semantic_model, acoustic_model = build_models(config)
    train_loader = build_dataloader(config, train=True)
    val_loader = build_dataloader(config, train=False)

    # 初始化Trainer
    trainer = Trainer(semantic_model, acoustic_model, train_loader, val_loader, config)

    # 开始训练并保存每轮loss记录
    train_losses, val_losses = trainer.train()

    # 保存loss日志
    torch.save({
        "train_losses": train_losses,
        "val_losses": val_losses
    }, os.path.join(config["log_dir"], "loss_log.pt"))

    print(f"✅ Experiment {exp_name} finished.\n")

if __name__ == "__main__":
    # 定义对比实验列表（可以随意增减）
    experiments = [
        # baseline
        {"exp_name": "baseline_encodec_w2vbert", "semantic_vocab_size": 4096, "acoustic_vocab_size": 2048},

        # 声学tokenizer替换
        {"exp_name": "soundstream_w2vbert", "semantic_vocab_size": 4096, "acoustic_vocab_size": 2048},

        # 语义tokenizer替换
        {"exp_name": "encodec_hubert", "semantic_vocab_size": 4096, "acoustic_vocab_size": 2048},

        # 双替换
        {"exp_name": "specvqvae_cpc", "semantic_vocab_size": 2048, "acoustic_vocab_size": 2048},

        # 小词表压缩
        {"exp_name": "small_vocab_1024", "semantic_vocab_size": 1024, "acoustic_vocab_size": 1024},
    ]

    # 执行所有实验
    for exp in experiments:
        run_experiment(**exp)
