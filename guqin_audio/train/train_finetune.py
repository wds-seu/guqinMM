# train/train_finetune.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

from model.guqin_from_encodec import GuqinTransformerFromEncodec
from dataset.dataset_guqin import GuqinDataset, guqin_collate_fn

def train_one_epoch(model, dataloader, optimizer, device, scaler, accumulation_steps=4):
    model.train()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    pbar = tqdm(dataloader, desc="Finetuning")

    for step, batch in enumerate(pbar):
        score_tokens = batch["score_tokens"].to(device)
        acoustic_tokens = batch["acoustic_tokens"].to(device)
        style_labels = batch["style_label"].to(device)

        B, T = acoustic_tokens.shape

        optimizer.zero_grad()

        # 小分块（比如 chunk=1024）
        chunk_size = min(1024, T)
        states = None
        offset = 0

        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            tgt_chunk = acoustic_tokens[:, start:end]

            with autocast():
                logits, states, offset = model(
                    score_tokens=score_tokens,
                    style_labels=style_labels,
                    targets=tgt_chunk,
                    states=states,
                    offset=offset
                )
                logits = logits.transpose(1, 2)
                loss = criterion(logits, tgt_chunk) / accumulation_steps

            scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * accumulation_steps
        avg_loss = total_loss / (step + 1)
        pbar.set_postfix({"loss": avg_loss})

    return total_loss / len(dataloader)

def main():
    # 参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4
    epochs = 20
    lr = 2e-4

    # 加载模型
    model = GuqinTransformerFromEncodec(
        vocab_size_score=128,
        vocab_size_acoustic=1024,
        style_emb_dim=256,
        d_model=256,
        freeze_transformer=True
    )
    model = nn.DataParallel(model)
    model.to(device)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scaler = GradScaler()

    # 加载数据
    dataset = GuqinDataset(
        data_path="./data/finetune/",  # 放你的5首谱文小数据
        mode="train",
        inference_mode=False
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: guqin_collate_fn(x, inference_mode=False))

    os.makedirs("./finetune_ckpt", exist_ok=True)

    # 训练
    for epoch in range(epochs):
        loss = train_one_epoch(model, dataloader, optimizer, device, scaler)
        print(f"[Epoch {epoch}] Loss: {loss:.4f}")

        # 保存checkpoint
        torch.save({
            "model": model.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }, f"./finetune_ckpt/finetune_epoch{epoch}.pt")

if __name__ == "__main__":
    main()
