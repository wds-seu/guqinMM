import torch.nn as nn

def build_loss(loss_type='ce'):
    if loss_type == 'ce':
        return nn.CrossEntropyLoss()
    elif loss_type == 'smooth_ce':
        return nn.CrossEntropyLoss(label_smoothing=0.1)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
