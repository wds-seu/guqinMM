import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.optim import Optimizer
import math


def tiny_value_of_dtype(dtype: torch.dtype):
    """
    Returns a moderately tiny value for a given PyTorch data type that is used to avoid numerical
    issues such as division by zero.
    This is different from `info_value_of_dtype(dtype).tiny` because it causes some NaN bugs.
    Only supports floating point dtypes.
    """
    if not dtype.is_floating_point:
        raise TypeError("Only supports floating point dtypes.")
    if dtype == torch.float or dtype == torch.double:
        return 1e-13
    elif dtype == torch.half:
        return 1e-4
    else:
        raise TypeError("Does not support dtype " + str(dtype))


def get_activation_fn(activation, layer=False):
    if activation == "relu":
        if layer:
            return nn.ReLU()
        else:
            return F.relu
    elif activation == "gelu":
        if layer:
            return nn.GELU()
        else:
            return F.gelu

    raise RuntimeError(
        "activation should be relu/gelu, not {}".format(activation))


def generate_square_subsequent_mask(sz: int, diag=0) -> Tensor:
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz), diag) == 1).transpose(0, 1)
    mask = (mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(
        mask == 1, float(0.0)))
    return mask


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps))
        return max(
            0.0, 0.5 *
            (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: int = 1,
    last_epoch: int = -1,
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps))
        return max(
            0.0,
            0.5 * (1.0 + math.cos(math.pi *
                                  ((float(num_cycles) * progress) % 1.0))),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class ReduceLROnPlateauWithWarmup(ReduceLROnPlateau):
    def __init__(
        self,
        optimizer,
        num_warmup_steps,
        mode="min",
        factor=0.1,
        patience=10,
        threshold=1e-4,
        threshold_mode="rel",
        cooldown=0,
        min_lr=0,
        eps=1e-8,
        verbose=False,
        last_epoch=0,
    ):
        super(ReduceLROnPlateauWithWarmup, self).__init__(
            optimizer,
            mode,
            factor,
            patience,
            threshold,
            threshold_mode,
            cooldown,
            min_lr,
            eps,
            verbose,
        )
        self.num_warmup_steps = num_warmup_steps
        # Initialize epoch and base learning rates
        if last_epoch == 0:
            for group in optimizer.param_groups:
                group.setdefault("initial_lr", group["lr"])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if "initial_lr" not in group:
                    raise KeyError(
                        "param 'initial_lr' is not specified "
                        "in param_groups[{}] when resuming an optimizer".
                        format(i))
        self.base_lrs = [
            group["initial_lr"] for group in optimizer.param_groups
        ]
        self.last_epoch = last_epoch
        if self.last_epoch <= self.num_warmup_steps:
            for group, base_lr in zip(self.optimizer.param_groups,
                                      self.base_lrs):
                group["lr"] = (base_lr * float(self.last_epoch) /
                               float(max(1, self.num_warmup_steps)))

    def step(self, metrics, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if epoch <= self.num_warmup_steps:
            for group, base_lr in zip(self.optimizer.param_groups,
                                      self.base_lrs):
                group["lr"] = (base_lr * float(epoch) /
                               float(max(1, self.num_warmup_steps)))
        else:
            if self.is_better(current, self.best):
                self.best = current
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1

            if self.in_cooldown:
                self.cooldown_counter -= 1
                self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

            if self.num_bad_epochs > self.patience:
                self._reduce_lr(epoch)
                self.cooldown_counter = self.cooldown
                self.num_bad_epochs = 0

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]
