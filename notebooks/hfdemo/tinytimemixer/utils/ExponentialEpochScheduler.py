from torch.optim.lr_scheduler import _LRScheduler

class ExponentialEpochScheduler(_LRScheduler):
    """
    与 OneCycleLR 类似接口的指数衰减调度器
    但在每个 epoch 后更新，而不是每个 step 后更新
    """
    def __init__(self, optimizer, gamma, epochs, steps_per_epoch, last_epoch=-1):
        """
        Args:
            optimizer: 优化器对象
            gamma: 每 epoch 的衰减系数
            epochs: 总 epoch 数
            steps_per_epoch: 每个 epoch 的步数 (batch 数)
            last_epoch: 最后一次 epoch 索引
        """
        self.gamma = gamma
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        super(ExponentialEpochScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """
        返回当前步数的学习率
        """
        # 计算当前 epoch (从0开始)
        current_epoch = self.last_epoch
        
        # 应用指数衰减
        new_lr = [base_lr * (self.gamma ** current_epoch) 
                  for base_lr in self.base_lrs]
        
        return new_lr

    def step(self, epoch=None):
        """
        更新调度器状态
        注意：我们将此设计为在 Trainer 每次调用 .step() 时更新，
        但实际只在每个 epoch 结束时才计算新学习率
        """
        if epoch is None:
            epoch = self.last_epoch + 1
        
        # 只在实际新 epoch 开始时更新 lr
        if epoch % self.steps_per_epoch == 0:
            super().step(epoch)
    
    @property
    def total_steps(self):
        """总步数，用于与 OneCycleLR 接口兼容"""
        return self.epochs * self.steps_per_epoch