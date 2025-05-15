import os
import torch
from torch.utils.data import DataLoader
from model.semantic_token_generator import SemanticTokenGenerator
from model.acoustic_token_generator import AcousticTokenGenerator
from utils.trainer import Trainer
from utils.dataset_merged import MergedGuqinDataset

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

config = {
    "device": get_device(),
    "batch_size": 8,
    "learning_rate": 1e-4,
    "num_epochs": 50,
    "max_seq_len": 1024,
    "semantic_vocab_size": 4096,
    "acoustic_vocab_size": 2048,
    "score_vocab_size": 5000,
    "style_vocab_size": 4,
    "embedding_dim": 256,
    "checkpoint_dir": "checkpoints"
}

def build_models(config):
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

def build_dataloaders(config):
    train_dataset = MergedGuqinDataset([
        "data/processed/train_data.pt",
        "data/processed_75h/75h_dataset.pt"
    ])
    val_dataset = MergedGuqinDataset([
        "data/processed/val_data.pt"
    ])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, drop_last=False)
    return train_loader, val_loader

def main():
    print(f"🚀 Using device: {config['device']}")

    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    semantic_model, acoustic_model = build_models(config)
    train_loader, val_loader = build_dataloaders(config)

    trainer = Trainer(semantic_model, acoustic_model, train_loader, val_loader, config)
    trainer.train()

    print("✅ Training completed.")

if __name__ == "__main__":
    main()
