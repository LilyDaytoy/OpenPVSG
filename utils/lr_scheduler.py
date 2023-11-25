import torch
from torch.optim.lr_scheduler import _LRScheduler
import math


class CosineAnnealingLRwithWarmUp(_LRScheduler):
    def __init__(self,
                 optimizer,
                 warmup_epochs=10,
                 max_lr=0.1,
                 min_lr=1e-7,
                 num_epochs=100):
        self.warmup_epochs = warmup_epochs
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.T_max = num_epochs - warmup_epochs
        self.end_annealing_epoch = warmup_epochs + self.T_max

        super(CosineAnnealingLRwithWarmUp, self).__init__(optimizer,
                                                          last_epoch=-1)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warm-up phase
            lr = (self.max_lr - self.base_lrs[0]
                  ) / self.warmup_epochs * self.last_epoch + self.base_lrs[0]
        else:
            # Cosine annealing phase
            cosine_decay = 0.5 * (1 + torch.cos(
                torch.tensor(self.last_epoch - self.warmup_epochs,
                             dtype=torch.float64) / self.T_max * math.pi))
            lr = (self.max_lr - self.min_lr) * cosine_decay + self.min_lr

        return [lr for _ in self.optimizer.param_groups]
