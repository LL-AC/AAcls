from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch.distributed as dist
import torch
from torch.utils.data import DataLoader, DistributedSampler
from utils import constant
import albumentations as A
import cv2
from .abdominal import Abdominal_View, Abdominal_View_Pred, Abdominal

def build_transform(is_train):
    if is_train:
        t = A.Compose([A.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
                A.OneOf([
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Rotate(20, p=0.5)
                ]),
                A.OneOf([
                    A.AdvancedBlur(
                        blur_limit=[5, 9],
                        sigma_x_limit=[0.7, 1],
                        sigma_y_limit=[0.7, 1],
                        rotate_limit=[-90, 90],
                        beta_limit=[0.5, 8],
                        noise_limit=[0.9, 1.1], p=0.3),
                   #A.AutoContrast(cutoff=0, method="pil", p=0.3),
                    A.ColorJitter(
                        brightness=[0.8, 1.2],
                        contrast=[0.8, 1.2],
                        saturation=[0.8, 1.2],
                        hue=[-0.5, 0.5]
                    )
                ], p=0.3),
                A.OneOf([
                    A.ChannelDropout(channel_drop_range=[1, 1], fill=0, p=0.3),
                    A.ChannelShuffle(p=0.3),
                    A.ColorJitter(
                        brightness=[0.8, 1.2],
                        contrast=[0.8, 1.2],
                        saturation=[0.8, 1.2],
                        hue=[-0.5, 0.5], p=0.3)], p=0.3),
                A.OneOf([
                    A.MedianBlur(blur_limit=[3, 5], p=0.3),
                    A.MotionBlur(blur_limit=[3, 5],
                        allow_shifted=False,
                        angle_range=[0, 20],
                        direction_range=[0, 0], p=0.3)
                ], p=0.3),
                A.Affine(
                    scale=[0.5, 1.5],
                    translate_percent=[-0.05, 0.05],
                    rotate=[-45, 45],
                    shear=[-15, 15],
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_NEAREST,
                    fit_output=False,
                    keep_ratio=False,
                    rotate_method="ellipse",
                    balanced_scale=True,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=0,
                    fill_mask=0, p=0.1),
                A.Normalize(mean=constant.IMAGE_MEAN, std=constant.IMAGE_STD),
                #A.ToTensorV2()
                ])
    else:
        t = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=constant.IMAGE_MEAN, std=constant.IMAGE_STD),
            #A.ToTensorV2()
            ])
        
    return t


def build_loader(args, mode='train', transform=None):
    dataset, collate_fn = build_dataset(args=args, mode=mode, transform=transform)
    # 只有训练集用拆分
    if args.distributed and 'train' in mode:
        print(f"local rank {args.gpu} / global rank {dist.get_rank()} successfully build {mode} dataset")

        num_tasks = dist.get_world_size()
        global_rank = dist.get_rank()
        sampler = DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

        loader = DataLoader(
            dataset, sampler=sampler,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            drop_last=True,
            collate_fn=collate_fn
        )
    else:
        print(f"successfully build {mode} dataset")
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            shuffle='train' in mode,
            collate_fn=collate_fn
        )

    print(args.pin_memory)

    return dataset, loader


def build_dataset(args, mode='train', transform=None):
    if transform is None:
        transform = build_transform('train' in mode)
    if args.dataset == 'your dataset':
        dataset = None
        collate_fn = collate_fn_a
    else:
        raise NotImplementedError("dataset error.")
    return dataset, collate_fn

def collate_fn_b(datas):
    imgs = torch.stack([batch['img'] for batch in datas], dim=0)
    labels = torch.tensor([batch['label'] for batch in datas])
    paths = [batch['path'] for batch in datas]
    return {'imgs':imgs, 'labels':labels, 'paths':paths}

def collate_fn_a(datas):
    pass