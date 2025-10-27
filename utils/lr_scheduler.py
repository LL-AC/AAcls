import math


def build_scheduler(args, optimizer):
    
    scheduler = args.lr_schedule.lower()
    
    if scheduler == 'harfcosine':
        lr_scheduler = HalfCosLRScheduler(optimizer=optimizer, epochs=args.epochs, 
                warmup_epochs=args.warmup_epochs, base_lr=args.lr, min_lr=args.min_lr)
    else:
        raise NotImplementedError(f'Unkown optimizer:{scheduler}')
    
    return lr_scheduler
    
class HalfCosLRScheduler:
    def __init__(self, optimizer, epochs, warmup_epochs, base_lr, min_lr) -> None:
        self.optimizer = optimizer
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
    
    def update_lr(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * epoch / self.warmup_epochs
        else:
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1. + math.cos(math.pi * (epoch - self.warmup_epochs) / (self.epochs - self.warmup_epochs)))
        for param_group in self.optimizer.param_groups:
            if 'lr_scale' in param_group:
                param_group['lr'] = lr * param_group['lr_scale']
            else:
                param_group['lr'] = lr
        return lr
            
        
    
    