import builtins
import copy
import datetime
import os
from pathlib import Path
import torch.distributed as dist

import torch


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'gloo'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    dist.barrier()
    setup_for_distributed(args.rank == 0)

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)
        
def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x
    
    
def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, ema_params=None, epoch_name=None):
    if epoch_name is None:
        epoch_name = str(epoch)
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / ('checkpoint-%s.pth' % epoch_name)

    # ema
    if ema_params is not None:
            ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
            for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
                assert name in ema_state_dict
                ema_state_dict[name] = ema_params[i]
    else:
        ema_state_dict = None

    if args.save_only_model == 'all':
        model_dict =  model_without_ddp.state_dict()
    else:
        model_dict = save_excluding_modules(model_without_ddp, exclude_modules=args.save_only_model.split(','))

    if epoch_name == 'best':
            # 只存模型
        to_save = {
            'model': model_dict,
            'args': args,
            }
    else:
        to_save = {
            'model': model_dict,
           # 'model_ema': ema_state_dict,
           # 'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'scaler': loss_scaler.state_dict(),
            'args': args,
        }
    save_on_master(to_save, checkpoint_path)

def save_specific_modules(model, module_names):
    """保存指定模块的参数（原有函数）"""
    full_state_dict = model.state_dict()
    specific_state_dict = {}
    for name, param in full_state_dict.items():
        for module in module_names:
            if name.startswith(f"{module}."):
                specific_state_dict[name] = param
                break
    return specific_state_dict

def save_excluding_modules(model, exclude_modules):
    """
    保存模型时排除特定模块的权重
    
    参数:
    model: 完整模型
    exclude_modules: 需要排除的模块名称列表（如['dropout', 'fc2']）
    save_path: 保存路径
    """
    # 获取完整状态字典
    full_state_dict = model.state_dict()
    
    # 筛选：排除以exclude_modules中模块名开头的参数
    filtered_state_dict = {}
    for name, param in full_state_dict.items():
        # 检查当前参数是否属于需要排除的模块
        exclude = False
        for module in exclude_modules:
            if name.startswith(f"{module}."):
                exclude = True
                break
        # 不排除的参数才保留
        if not exclude:
            filtered_state_dict[name] = param
    
    return filtered_state_dict