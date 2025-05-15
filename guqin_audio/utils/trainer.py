import torch
import torch.nn.functional as F
import os

class Trainer:
    def __init__(self, semantic_model, acoustic_model, train_loader, val_loader, config):
        self.semantic_model = semantic_model
        self.acoustic_model = acoustic_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = config["device"]
        self.semantic_optimizer = torch.optim.Adam(self.semantic_model.parameters(), lr=config["learning_rate"])
        self.acoustic_optimizer = torch.optim.Adam(self.acoustic_model.parameters(), lr=config["learning_rate"])
        self.config = config

    def train(self):
        best_val_loss = float('inf')
        for epoch in range(self.config["num_epochs"]):
            self.semantic_model.train()
            self.acoustic_model.train()

            for batch in self.train_loader:
                score_tokens = batch['score_tokens'].to(self.device)
                style_label = batch['style_label'].to(self.device)
                semantic_tokens = batch['semantic_tokens'].to(self.device)
                acoustic_tokens = batch['acoustic_tokens'].to(self.device)

                # 检查哪些是有谱文数据的样本（有谱文的话score_tokens非全0）
                has_score = (score_tokens.sum(dim=1) != 0)

                loss_semantic = torch.tensor(0.0, device=self.device)
                if has_score.any():
                    # 只在有谱文样本上训练Semantic Generator
                    selected_score = score_tokens[has_score]
                    selected_style = style_label[has_score]
                    selected_semantic = semantic_tokens[has_score]

                    semantic_logits = self.semantic_model(selected_score, selected_style)
                    loss_semantic = F.cross_entropy(semantic_logits.view(-1, semantic_logits.size(-1)), selected_semantic.view(-1))

                    self.semantic_optimizer.zero_grad()
                    loss_semantic.backward()
                    self.semantic_optimizer.step()

                # 声学生成器总是需要训练（所有样本都有acoustic_tokens）
                acoustic_logits = self.acoustic_model(semantic_tokens, style_label)
                loss_acoustic = F.cross_entropy(acoustic_logits.view(-1, acoustic_logits.size(-1)), acoustic_tokens.view(-1))

                self.acoustic_optimizer.zero_grad()
                loss_acoustic.backward()
                self.acoustic_optimizer.step()

            val_loss = self.evaluate()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.semantic_model.state_dict(), os.path.join(self.config["checkpoint_dir"], "semantic_best.pth"))
                torch.save(self.acoustic_model.state_dict(), os.path.join(self.config["checkpoint_dir"], "acoustic_best.pth"))

            print(f"Epoch {epoch+1}: Val Loss = {val_loss:.4f} (Semantic Loss = {loss_semantic.item():.4f})")

    def evaluate(self):
        self.semantic_model.eval()
        self.acoustic_model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                score_tokens = batch['score_tokens'].to(self.device)
                style_label = batch['style_label'].to(self.device)
                semantic_tokens = batch['semantic_tokens'].to(self.device)
                acoustic_tokens = batch['acoustic_tokens'].to(self.device)

                has_score = (score_tokens.sum(dim=1) != 0)

                if has_score.any():
                    selected_score = score_tokens[has_score]
                    selected_style = style_label[has_score]
                    selected_semantic = semantic_tokens[has_score]

                    semantic_logits = self.semantic_model(selected_score, selected_style)
                    loss_semantic = F.cross_entropy(semantic_logits.view(-1, semantic_logits.size(-1)), selected_semantic.view(-1))
                else:
                    loss_semantic = torch.tensor(0.0, device=self.device)

                acoustic_logits = self.acoustic_model(semantic_tokens, style_label)
                loss_acoustic = F.cross_entropy(acoustic_logits.view(-1, acoustic_logits.size(-1)), acoustic_tokens.view(-1))

                total_loss += (loss_semantic.item() + loss_acoustic.item())

        return total_loss / len(self.val_loader)
