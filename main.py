import numpy as np
import torch
from utils import *
from utils import NativeScalerWithGradNormCount as NativeScaler
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import argparse
from models import build_model
from datasets import build_loader
import copy
from trainer import Trainer
from tester import Tester
import torch.nn as nn
from losses import build_criterion

def get_args_parser():
    parser = argparse.ArgumentParser('framework', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * # gpus')
    parser.add_argument('--epochs', default=100, type=int)

    # Model parameters
    parser.add_argument('--model', type=str, metavar='MODEL', default='resnet',
                        help='Name of model to train')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--grad_checkpointing', action='store_true')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=1e-8, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--lr_schedule', type=str, default='harfcosine',
                        help='learning rate schedule')
    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--ema_rate', default=0.9999, type=float)
    parser.add_argument('--optimizer', default='adam', type=str, help='optimizer')
    parser.add_argument('--accum_iter', default=32, type=int)
    
    #losses criterion
    parser.add_argument('--ctiterion', default='CrossEntropy', type=str,
                        help='objetive function')

    
    #train_trick
    parser.add_argument('--grad_clip', type=float, default=3.0,
                        help='Gradient clip')

    # Dataset parameters
    parser.add_argument('--data_path', default='../Premature/crop_data', type=str,
                        help='dataset path')
    parser.add_argument('--dataset', default='premature_multi', type=str,
                        help='dataset path')
    parser.add_argument('--num_classes', default=5, type=int,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--evaluate', action='store_true',
                    help='just test')
    parser.add_argument('--save_last_freq', default=10, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_memory', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    
    #save_only_model
    parser.add_argument('--save_only_model', default='all', type=str)

    # distributed training parameters
    # parser.add_argument('--world_size', default=1, type=int,
    #                     help='number of distributed processes')
    #parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    init_distributed_mode(args)

    # fix the seed for reproducibility
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = get_world_size()
    global_rank = get_rank()
    
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None
        
        
    model = build_model(args)
    
    model.cuda()
    model_without_ddp = model
    
    eff_batch_size = args.batch_size * get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("effective batch size: %d" % eff_batch_size)

    freeze_by_names(model, ['clip_model', 'text_encoder'])

    if args.distributed:
        #model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], broadcast_buffers=False)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], broadcast_buffers=False, find_unused_parameters=True)
        model_without_ddp = model.module
        
        
    optimizer = build_optimizer(args, model)
    
    print(optimizer)
    loss_scaler = NativeScaler()
    
    # resume training
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        #missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model_ema'], strict=False)
        model_params = list(model_without_ddp.parameters())
       
        print("Resume checkpoint %s" % args.resume)
        print('missing keys:', missing_keys)
        print('unexpected keys:', unexpected_keys)

        if 'model_ema' in checkpoint:
            ema_state_dict = checkpoint['model_ema']
            ema_params = [ema_state_dict[name].cuda() for name, _ in model_without_ddp.named_parameters()]
        else:
            model_params = list(model_without_ddp.parameters())
            ema_params = copy.deepcopy(model_params)

        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")
        del checkpoint
    else:
        model_params = list(model_without_ddp.parameters())
        ema_params = copy.deepcopy(model_params)
        print("Training from scratch")

    print("Model = %s" % str(model))
    # following timm: set wd as 0 for bias and norm layers
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {}M".format(n_params / 1e6))
    
    criterion = build_criterion(args)
    
    train_set, train_loader = build_loader(args=args)
    valid_set, valid_loader = build_loader(args=args, mode='test')

    # evaluate
    if args.evaluate:
        if is_main_process():
            torch.cuda.empty_cache()
            print('evaluating........')
            tester = Tester(args, model_without_ddp, valid_loader, criterion, [accuracy, sensitivity_multi, specificity_multi], save_results=[plot_result, plot_matrix])
            tester.test(mode='in')
        return

    lr_scheduler = build_scheduler(args=args, optimizer=optimizer)
    
    trainer = Trainer(args, model, model_without_ddp, model_params, 
                      ema_params, criterion, train_loader, valid_loader,
                      optimizer, lr_scheduler, loss_scaler, log_writer)
    
    trainer.train()

    if is_main_process():
        tester = Tester(args, model_without_ddp, valid_loader, criterion, [accuracy, sensitivity_multi, specificity_multi], save_results=[plot_result, plot_matrix])
        tester.test()

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

    