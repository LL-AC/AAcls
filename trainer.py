from datetime import datetime
from tqdm import tqdm
from utils import *
import sys
from timm.utils import AverageMeter
from Baser import Baser

class Trainer(Baser):
    def __init__(self, args, model, model_without_ddp, model_params, 
                 ema_params, criterion, train_loader, valid_loader, 
                 optimizer, lr_scheduler, loss_scaler, log_writer, metric_fn=[accuracy]):
        super().__init__(args, model, criterion, metric_fn)
        #self.model = model
        self.model_without_ddp = model_without_ddp
        self.model_params = model_params
        self.ema_params = ema_params
        self.criterion = criterion
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_scaler = loss_scaler
        self.log_writer = log_writer
        self.accum_iter = args.accum_iter
    
    def train_one_epoch(self, epoch):
        self.model.train(True)
        metric_logger = MetricLogger(delimiter='  ')
        metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('acc', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('loss', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)
        print_freq = 20
        
        self.optimizer.zero_grad()
        accum_iter = self.args.accum_iter
        
        if self.log_writer is not None:
            print('log_dir: {}'.format(self.log_writer.log_dir))

        for data_iter_step, (samples) in enumerate(metric_logger.log_every(self.train_loader, print_freq, header)):

            # we use a per iteration (instead of per epoch) lr scheduler
            if data_iter_step % accum_iter == 0:
                self.lr_scheduler.update_lr(data_iter_step / len(self.train_loader) + epoch)

            # 任何输入都不用改train和valid函数，直接改forward
            output = self.forward(samples)
            labels = samples['labels'].cuda(non_blocking=True)
            loss = self.compute_loss(output, labels)

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            loss /= accum_iter
            
            self.loss_scaler(loss, self.optimizer, parameters=self.model.parameters(),
                     update_grad=(data_iter_step + 1) % accum_iter == 0)
            if (data_iter_step + 1) % accum_iter == 0:
                self.optimizer.zero_grad()

            torch.cuda.synchronize()
            
            self.update_ema(self.ema_params, self.model_params, rate=self.args.ema_rate)

            metric_logger.update(loss=loss_value)

            lr = self.optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)
            
            
            acc = self.metric(output['predicted'], labels)['accuracy']
            metric_logger.update(acc=acc)

            loss_value_reduce = all_reduce_mean(loss_value)
            if self.log_writer is not None:
                """ We use epoch_1000x as the x-axis in tensorboard.
                This calibrates different curves when batch size changes.
                """
                epoch_1000x = int((data_iter_step / len(self.train_loader) + epoch) * 1000)
                self.log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                self.log_writer.add_scalar('lr', lr, epoch_1000x)
                self.log_writer.add_scalar('accuracy', acc, epoch_1000x)
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        
    def train(self):
        print(f'Start training for {self.args.epochs} epochs')
        start_time = time.time()
        acc = 0
        for epoch in range(self.args.start_epoch, self.args.epochs):
            if self.args.distributed:
                self.train_loader.sampler.set_epoch(epoch)
            
            self.train_one_epoch(epoch)
            
             #save checkpoint
            if is_main_process():   
                acc1 = self.evaluate(epoch=epoch)  
                if acc <= acc1:
                    acc = acc1
                    print('saving best_model----')
                    save_model(args=self.args, model=self.model, model_without_ddp=self.model_without_ddp, optimizer=self.optimizer,
                                    loss_scaler=self.loss_scaler, epoch=epoch, ema_params=self.ema_params, epoch_name='best')
                
                if epoch % self.args.save_last_freq == 0 or epoch + 1 == self.args.epochs:
                    print('saving model----')
                    save_model(args=self.args, model=self.model, model_without_ddp=self.model_without_ddp, optimizer=self.optimizer,
                                    loss_scaler=self.loss_scaler, epoch=epoch, ema_params=self.ema_params, epoch_name='last')
               # self.evaluate(epoch=epoch)
                if self.log_writer is not None:
                    self.log_writer.flush()
            
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
        
    def evaluate(self, epoch):
        acc = AverageMeter()
        l = AverageMeter()
        
        predicteds = []
        targets = []
        self.model.eval()
        print('Epoch:[{}] validating.......'.format(epoch))
        for samples in tqdm(self.valid_loader):
           # with torch.no_grad:
            output = self.forward(samples, require_grad=False)
            labels = samples['labels'].cuda(non_blocking=True)
            loss = self.compute_loss(output, labels)

            l.update(loss.item())

            predicted = output['predicted'].cpu()
            labels = labels.cpu()
            predicteds.append(predicted)
            targets.append(labels)

        predicteds = torch.cat(predicteds, dim=0)
        targets = torch.cat(targets, dim=0)
        acc = self.metric(predicteds, targets)['accuracy']
        
        print('Valid loss is {}, accuracy is {}'.format(l.avg, acc))
        if self.log_writer is not None:
            self.log_writer.add_scalar('valid_loss', l.avg, epoch)
            self.log_writer.add_scalar('Valid_accuracy', acc, epoch)
        return acc
    
    def update_ema(self, target_params, source_params, rate=0.99):
        """
        Update target parameters to be closer to those of source parameters using
        an exponential moving average.

        :param target_params: the target parameter sequence.
        :param source_params: the source parameter sequence.
        :param rate: the EMA rate (closer to 1 means slower).
        """
        for targ, src in zip(target_params, source_params):
            targ.detach().mul_(rate).add_(src, alpha=1 -rate)

