import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.dataset_75h import AcousticTokenDataset75h
from model.model_acoustic_decoder import AcousticTransformerDecoder
from tqdm import tqdm
import argparse

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # 流派标签映射
    style_mapping = {
        "exam": 0,
        "master": 1,
        "silkqin": 2,
        "other": 3
    }

    # 加载数据集
    dataset = AcousticTokenDataset75h(args.acoustic_tokens_dir, style_mapping)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # 创建模型
    model = AcousticTransformerDecoder(
        vocab_size=args.vocab_size,
        style_emb_dim=args.style_emb_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=0.1
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(args.num_epochs):
        total_loss = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}"):
            acoustic_tokens = batch["acoustic_tokens"].to(device)  # (B, T)
            style_labels = batch["style_label"].to(device)          # (B,)

            optimizer.zero_grad()

            # 训练时输入 acoustic_tokens[:, :-1]，目标是 acoustic_tokens[:, 1:]
            input_tokens = acoustic_tokens[:, :-1]
            target_tokens = acoustic_tokens[:, 1:]

            output_logits = model(input_tokens, style_labels)  # (B, T-1, vocab_size)
            output_logits = output_logits.reshape(-1, args.vocab_size)
            target_tokens = target_tokens.reshape(-1)

            loss = criterion(output_logits, target_tokens)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"✅ Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

        # 保存模型
        torch.save(model.state_dict(), os.path.join(args.save_dir, f"pretrain_epoch{epoch+1}.pt"))

def collate_fn(batch):
    """
    自定义collate_fn，进行动态padding
    """
    acoustic_tokens = [sample["acoustic_tokens"] for sample in batch]
    style_labels = torch.tensor([sample["style_label"] for sample in batch], dtype=torch.long)

    # 动态padding到同一长度
    max_len = max([len(x) for x in acoustic_tokens])
    padded_tokens = torch.zeros((len(batch), max_len), dtype=torch.long)

    for i, tokens in enumerate(acoustic_tokens):
        padded_tokens[i, :len(tokens)] = tokens

    return {
        "acoustic_tokens": padded_tokens,
        "style_label": style_labels
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--acoustic_tokens_dir', type=str, required=True, help="75h声学tokens目录")
    parser.add_argument('--save_dir', type=str, default="checkpoints_pretrain_75h", help="保存模型的路径")
    parser.add_argument('--vocab_size', type=int, default=1024, help="声学token词表大小")
    parser.add_argument('--style_emb_dim', type=int, default=256, help="流派嵌入向量维度")
    parser.add_argument('--d_model', type=int, default=512, help="Transformer隐藏层维度")
    parser.add_argument('--nhead', type=int, default=8, help="多头注意力头数")
    parser.add_argument('--num_layers', type=int, default=6, help="Transformer层数")
    parser.add_argument('--batch_size', type=int, default=16, help="batch大小")
    parser.add_argument('--lr', type=float, default=1e-4, help="学习率")
    parser.add_argument('--num_epochs', type=int, default=20, help="训练轮数")

    args = parser.parse_args()
    train(args)
