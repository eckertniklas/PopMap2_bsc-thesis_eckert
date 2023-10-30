


from torch.optim.lr_scheduler import _LRScheduler

class CustomLRScheduler(_LRScheduler):
    def __init__(self, optimizer, drop_epochs, gamma=0.1, last_epoch=-1):
        self.drop_epochs = drop_epochs
        self.gamma = gamma
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch in self.drop_epochs:
            self.base_lrs = [base_lr * self.gamma for base_lr in self.base_lrs]
            return self.base_lrs
        return self.base_lrs

