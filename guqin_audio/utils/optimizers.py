import torch.optim as optim

def build_optimizer(model, lr=1e-4, weight_decay=1e-2):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer
