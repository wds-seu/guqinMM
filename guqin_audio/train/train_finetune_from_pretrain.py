import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_supervised import SupervisedAcousticDataset
from model.model_acoustic_decoder_with_score import AcousticTransformerDecoderWithScore
from tqdm import tqdm
import argparse

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # 加载有监督数据集
    dataset = SupervisedAcousticDataset(
        dataset_path=args.supervised_dataset_path
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # 初始化模型
    model = AcousticTransformerDecoderWithScore(
        vocab_size=args.vocab_size,
        style_emb_dim=args.style_emb_dim,
        score_emb_dim=args.score_emb_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=0.1
    ).to(device)

    # 加载无监督预训练权重
    if args.pretrained_path:
        print(f"🔄 Loading pretrained weights from {args.pretrained_path}")
        pretrained_dict = torch.load(args.pretrained_path)
        model.load_pretrained_decoder(pretrained_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(args.num_epochs):
        total_loss = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}"):
            acoustic_tokens = batch["acoustic_tokens"].to(device)  # (B, T)
            score_tokens = batch["score_tokens"].to(device)        # (B, T_score)
            style_labels = batch["style_label"].to(device)

            optimizer.zero_grad()

            input_tokens = acoustic_tokens[:, :-1]
            target_tokens = acoustic_tokens[:, 1:]

            output_logits = model(input_tokens, score_tokens, style_labels)

            output_logits = output_logits.reshape(-1, args.vocab_size)
            target_tokens = target_tokens.reshape(-1)

            loss = criterion(output_logits, target_tokens)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"✅ Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

        torch.save(model.state_dict(), os.path.join(args.save_dir, f"finetune_epoch{epoch+1}.pt"))

def collate_fn(batch):
    acoustic_tokens = [sample["acoustic_tokens"] for sample in batch]
    score_tokens = [sample["score_tokens"] for sample in batch]
    style_labels = torch.tensor([sample["style_label"] for sample in batch], dtype=torch.long)

    max_len = max([len(x) for x in acoustic_tokens])
    padded_acoustic = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, tokens in enumerate(acoustic_tokens):
        padded_acoustic[i, :len(tokens)] = tokens

    max_len_score = max([len(x) for x in score_tokens])
    padded_score = torch.zeros((len(batch), max_len_score), dtype=torch.long)
    for i, tokens in enumerate(score_tokens):
        padded_score[i, :len(tokens)] = tokens

    return {
        "acoustic_tokens": padded_acoustic,
        "score_tokens": padded_score,
        "style_label": style_labels
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--supervised_dataset_path', type=str, required=True, help="监督训练集路径")
    parser.add_argument('--pretrained_path', type=str, required=True, help="无监督预训练模型路径")
    parser.add_argument('--save_dir', type=str, default="checkpoints_finetune", help="微调保存路径")
    parser.add_argument('--vocab_size', type=int, default=1024)
    parser.add_argument('--style_emb_dim', type=int, default=256)
    parser.add_argument('--score_emb_dim', type=int, default=256)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=50)
    args = parser.parse_args()
    train(args)
